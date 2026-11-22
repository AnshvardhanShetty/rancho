// Choose an admission cutoff from earlier offers and the remaining write budget.
//
// record_priced_offer keeps the last 20,000 offers and refreshes the cutoffs
// every 1,000 offers. The current decision has already been made.
// recalculate_admission_cutoffs estimates future demand from recent offers.
// update_cutoff finds the score needed to fit the budget. The helpers at the
// bottom repeat that calculation for groups of similar write sizes.
// select_admission_cutoff chooses which cutoff to apply to the next offer.

#pragma once

#include "cache.hpp"
#include <array>
#include <cmath>

// Score cutoff and admission probability for exactly equal scores.
struct AdmissionCutoff {
  double score_cutoff = 0, tie_probability = 0;
  bool is_ready = false;
};

// One observed offer's score, write cost, and arrival time.
struct PricedOffer {
  double score = 0;
  int64_t write_cost_blocks = 0, arrival_time_us = 0;
};

// Mix a value for deterministic equal-score decisions.
static uint64_t mix_hash(uint64_t value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33;
  return value;
}

// Get a reproducible value in [0, 1) from the request and seed.
static double random_value_for_request(uint64_t request_index, uint64_t seed) {
  return double(mix_hash(request_index * 0x9E3779B97F4A7C15ULL ^ seed) >> 11) /
         9007199254740992.0;
}

// Track spending and the last 20,000 offers; update cutoffs every 1,000 offers.
struct WriteController {
  static constexpr int64_t history_capacity = 20000;

  int64_t write_budget_blocks, evaluation_start_us, evaluation_end_us;

  int64_t blocks_written = 0, history_count = 0, next_history_slot = 0, offers_seen = 0;
  double global_score_cutoff = -INFINITY, tie_probability = 1;

  bool global_cutoff_ready = false;
  std::array<AdmissionCutoff, 64> cutoffs_by_cost_band{};
  std::vector<PricedOffer> recent_offers;

  // Start with the full budget and no learned cutoff.
  WriteController(int64_t write_allowance_blocks, int64_t start_us, int64_t end_us)
      : write_budget_blocks(write_allowance_blocks), evaluation_start_us(start_us),
        evaluation_end_us(end_us), recent_offers(history_capacity) {}

  // Save the offer after its decision and before charging any write.
  void record_priced_offer(double score, int64_t write_cost_blocks, int64_t arrival_us) {
    recent_offers[next_history_slot] = {score, write_cost_blocks, arrival_us};
    next_history_slot = (next_history_slot + 1) % history_capacity;
    if (history_count < history_capacity)
      history_count++;
    offers_seen++;
    if (offers_seen % 1000 == 0 && history_count != 0)
      recalculate_admission_cutoffs(arrival_us);
  }

  // Project recent write demand to the end of the run and compare it with the budget.
  void recalculate_admission_cutoffs(int64_t arrival_us) {
    if (history_count == 0)
      return;

    // Collect positive-cost, finite-score offers and the history time span.
    std::vector<PricedOffer> priced_offers;
    priced_offers.reserve(history_count);
    double total_write_cost = 0;
    int64_t earliest_time = recent_offers[0].arrival_time_us;
    int64_t latest_time = earliest_time;

    for (int64_t history_slot = 0; history_slot < history_count; history_slot++) {
      const PricedOffer &offer = recent_offers[history_slot];
      earliest_time = std::min(earliest_time, offer.arrival_time_us);
      latest_time = std::max(latest_time, offer.arrival_time_us);
      if (offer.write_cost_blocks > 0 && std::isfinite(offer.score)) {
        priced_offers.push_back({offer.score, offer.write_cost_blocks});
        total_write_cost += offer.write_cost_blocks;
      }
    }

    // Keep the existing cutoffs until there are 100 usable offers.
    if (priced_offers.size() < 100)
      return;

    // Measure the recent offer rate and average write cost.
    double elapsed_seconds = std::max(double(latest_time - earliest_time) / 1e6, 1.0);
    double offers_per_second = double(priced_offers.size()) / elapsed_seconds;
    double average_write_cost = total_write_cost / priced_offers.size();

    // Calculate the time and write allowance remaining.
    double remaining_seconds =
        std::max((double(evaluation_end_us) - double(arrival_us)) / 1e6, 1.0);
    double remaining_blocks = std::max(double(write_budget_blocks - blocks_written), 0.0);

    // If future offers cost 100 blocks and 30 remain, aim to pay for 30%.
    double estimated_write_demand = offers_per_second * remaining_seconds * average_write_cost;
    double budget_share = remaining_blocks / std::max(estimated_write_demand, 1e-9);
    double affordable_fraction = std::max(0.0, std::min(1.0, budget_share));

    // Recalculate stored cutoffs for subsequent decisions.
    update_cost_band_cutoffs(priced_offers, affordable_fraction);
    update_cutoff(priced_offers, affordable_fraction, total_write_cost);
    global_cutoff_ready = true;
  }

  // Work down the score list until we reach the target number of writes.
  void update_cutoff(std::vector<PricedOffer> &priced_offers, double affordable_fraction,
                     double total_write_cost) {
    if (affordable_fraction <= 0) {
      global_score_cutoff = INFINITY;
      tie_probability = 0;
    } else if (affordable_fraction >= 1) {
      global_score_cutoff = -INFINITY;
      tie_probability = 1;
    } else {
      std::sort(priced_offers.begin(), priced_offers.end(),
                [](const PricedOffer &left_offer, const PricedOffer &right_offer) {
                  return left_offer.score > right_offer.score;
                });
      double desired_write_cost = affordable_fraction * total_write_cost;
      double cost_above = 0;
      double cost_at_cutoff = 0;
      global_score_cutoff = priced_offers.back().score;
      for (size_t score_index = 0; score_index < priced_offers.size();) {
        size_t next_score_index = score_index;
        cost_at_cutoff = 0;
        while (next_score_index < priced_offers.size() &&
               priced_offers[next_score_index].score == priced_offers[score_index].score) {
          cost_at_cutoff += priced_offers[next_score_index].write_cost_blocks;
          next_score_index++;
        }
        if (cost_above + cost_at_cutoff >= desired_write_cost) {
          global_score_cutoff = priced_offers[score_index].score;
          break;
        }
        cost_above += cost_at_cutoff;
        score_index = next_score_index;
      }
      // If tied offers cross the boundary, admit only the fraction that fits.
      tie_probability =
          cost_at_cutoff > 0 ? (desired_write_cost - cost_above) / cost_at_cutoff : 0;
      tie_probability = std::max(0.0, std::min(1.0, tie_probability));
    }
  }

  // Use the cost-group cutoff when available, otherwise the global cutoff.
  // Halve it when the fraction spent is below the fraction of evaluation time
  // elapsed.
  AdmissionCutoff select_admission_cutoff(int64_t write_cost_blocks, int64_t arrival_us) const {
    AdmissionCutoff cutoff{global_score_cutoff, tie_probability, global_cutoff_ready};
    if (write_cost_blocks > 0 && global_cutoff_ready) {
      const auto &cost_band_index = cutoffs_by_cost_band[write_cost_band(write_cost_blocks)];
      if (cost_band_index.is_ready)
        cutoff = cost_band_index;
    }
    if (write_cost_blocks > 0 && cutoff.is_ready &&
        (__int128)blocks_written * (evaluation_end_us - evaluation_start_us) <
            (__int128)write_budget_blocks * (arrival_us - evaluation_start_us))
      cutoff.score_cutoff *= 0.5;
    return cutoff;
  }

  static int write_cost_band(int64_t write_cost_blocks);
  void update_cost_band_cutoffs(const std::vector<PricedOffer> &priced_offers,
                                double affordable_fraction);
};

// Cost-group cutoffs: use a separate cutoff for each sufficiently populated
// price range.
inline int WriteController::write_cost_band(int64_t write_cost_blocks) {
  return write_cost_blocks > 0
             ? 63 - __builtin_clzll(static_cast<unsigned long long>(write_cost_blocks))
             : 0;
}

inline void
WriteController::update_cost_band_cutoffs(const std::vector<PricedOffer> &priced_offers,
                                          double affordable_fraction) {
  cutoffs_by_cost_band = {};
  std::array<std::vector<PricedOffer>, 64> offers_by_cost_band;
  for (const auto &offer : priced_offers)
    offers_by_cost_band[write_cost_band(offer.write_cost_blocks)].push_back(offer);
  for (int cost_band_index = 0; cost_band_index < 64; ++cost_band_index) {
    auto &band_offers = offers_by_cost_band[cost_band_index];
    if (band_offers.size() < 100)
      continue;
    auto &band_cutoff = cutoffs_by_cost_band[cost_band_index];
    band_cutoff.is_ready = true;
    if (affordable_fraction <= 0) {
      band_cutoff.score_cutoff = INFINITY;
      band_cutoff.tie_probability = 0;
      continue;
    }
    if (affordable_fraction >= 1) {
      band_cutoff.score_cutoff = -INFINITY;
      band_cutoff.tie_probability = 1;
      continue;
    }
    std::sort(band_offers.begin(), band_offers.end(),
              [](const PricedOffer &left_offer, const PricedOffer &right_offer) {
                return left_offer.score > right_offer.score;
              });
    int64_t band_write_cost = 0, cost_above = 0;
    for (const auto &offer : band_offers)
      band_write_cost += offer.write_cost_blocks;
    double desired_write_cost = affordable_fraction * band_write_cost;
    for (size_t score_index = 0; score_index < band_offers.size();) {
      size_t next_score_index = score_index;
      int64_t cost_at_cutoff = 0;
      while (next_score_index < band_offers.size() &&
             band_offers[next_score_index].score == band_offers[score_index].score) {
        cost_at_cutoff += band_offers[next_score_index].write_cost_blocks;
        ++next_score_index;
      }
      if (cost_above + cost_at_cutoff >= desired_write_cost) {
        band_cutoff.score_cutoff = band_offers[score_index].score;
        band_cutoff.tie_probability =
            std::clamp((desired_write_cost - cost_above) / cost_at_cutoff, 0.0, 1.0);
        break;
      }
      cost_above += cost_at_cutoff;
      score_index = next_score_index;
    }
  }
}
