// Decide how selective to be with the remaining flash write allowance
// Save recent offers including ones RANCHO declined
// Estimate how much it would cost to write future offers at the recent rate
// Pick a score cutoff that fits what we can still afford
// The current offer is recorded after its decision so it affects later cutoffs

#pragma once

#include "cache.hpp"
#include <array>
#include <cmath>

// Score cutoff and admission probability for exactly equal scores.
struct AdmissionCutoff {
  double score = 0, tie_probability = 0;
  bool ready = false;
};

// One observed offer's score, write cost, and arrival time.
struct PricedOffer {
  double score = 0;
  int64_t cost = 0, time = 0;
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
static double random_value_for_request(uint64_t row, uint64_t seed) {
  return double(mix_hash(row * 0x9E3779B97F4A7C15ULL ^ seed) >> 11) /
         9007199254740992.0;
}

// Track spending and the last 20,000 offers; update cutoffs every 1,000 offers.
struct WriteController {
  static constexpr int64_t history_capacity = 20000;

  int64_t budget, start_time, end_time;

  int64_t spent = 0, history_count = 0, next_history_slot = 0, observations = 0;
  double threshold = -INFINITY, tie_probability = 1;

  bool has_threshold = false;
  std::array<AdmissionCutoff, 64> bands{};
  std::vector<PricedOffer> history;

  // Start with the full budget and no learned cutoff.
  WriteController(int64_t allowance, int64_t start, int64_t end)
      : budget(allowance), start_time(start), end_time(end),
        history(history_capacity) {}

  // Save each decided offer; refresh cutoffs every 1,000 offers before charging
  // its write.
  void record_priced_offer(double score, int64_t cost, int64_t now) {
    history[next_history_slot] = {score, cost, now};
    next_history_slot = (next_history_slot + 1) % history_capacity;
    if (history_count < history_capacity)
      history_count++;
    observations++;
    if (observations % 1000 == 0 && history_count != 0)
      recalculate_admission_cutoffs(now);
  }

  // Estimate future write demand and the fraction the remaining budget can
  // afford.
  void recalculate_admission_cutoffs(int64_t now) {
    if (history_count == 0)
      return;

    // Collect positive-cost, finite-score offers and the history time span.
    std::vector<PricedOffer> priced_offers;
    priced_offers.reserve(history_count);
    double total_write_cost = 0;
    int64_t earliest_time = history[0].time;
    int64_t latest_time = earliest_time;

    for (int64_t i = 0; i < history_count; i++) {
      earliest_time = std::min(earliest_time, history[i].time);
      latest_time = std::max(latest_time, history[i].time);
      if (history[i].cost > 0 && std::isfinite(history[i].score)) {
        priced_offers.push_back({history[i].score, history[i].cost});
        total_write_cost += history[i].cost;
      }
    }

    // Keep the existing cutoffs until there are 100 usable offers.
    if (priced_offers.size() < 100)
      return;

    // Measure the recent offer rate and average write cost.
    double elapsed_seconds =
        std::max(double(latest_time - earliest_time) / 1e6, 1.0);
    double offers_per_second = double(priced_offers.size()) / elapsed_seconds;
    double average_write_cost = total_write_cost / priced_offers.size();

    // Calculate the time and write allowance remaining.
    double remaining_seconds =
        std::max((double(end_time) - double(now)) / 1e6, 1.0);
    double remaining_blocks = std::max(double(budget - spent), 0.0);

    // Forecast write demand and limit the affordable fraction to [0, 1].
    double estimated_write_demand =
        offers_per_second * remaining_seconds * average_write_cost;
    double affordable_fraction = std::max(
        0.0, std::min(1.0, remaining_blocks /
                               std::max(estimated_write_demand, 1e-9)));

    // Recalculate stored cutoffs for subsequent decisions.
    update_cost_band_cutoffs(priced_offers, affordable_fraction);
    update_cutoff(priced_offers, affordable_fraction, total_write_cost);
    has_threshold = true;
  }

  // Set the global cutoff by write volume, sharing equal scores
  // probabilistically.
  void update_cutoff(std::vector<PricedOffer> &priced_offers,
                     double affordable_fraction, double total_write_cost) {
    if (affordable_fraction <= 0) {
      threshold = INFINITY;
      tie_probability = 0;
    } else if (affordable_fraction >= 1) {
      threshold = -INFINITY;
      tie_probability = 1;
    } else {
      std::sort(
          priced_offers.begin(), priced_offers.end(),
          [](const PricedOffer &left_offer, const PricedOffer &right_offer) {
            return left_offer.score > right_offer.score;
          });
      double desired_write_cost = affordable_fraction * total_write_cost,
             cost_above = 0, cost_at_cutoff = 0;
      threshold = priced_offers.back().score;
      for (size_t score_index = 0; score_index < priced_offers.size();) {
        size_t next_score_index = score_index;
        cost_at_cutoff = 0;
        while (next_score_index < priced_offers.size() &&
               priced_offers[next_score_index].score ==
                   priced_offers[score_index].score) {
          cost_at_cutoff += priced_offers[next_score_index].cost;
          next_score_index++;
        }
        if (cost_above + cost_at_cutoff >= desired_write_cost) {
          threshold = priced_offers[score_index].score;
          break;
        }
        cost_above += cost_at_cutoff;
        score_index = next_score_index;
      }
      tie_probability = cost_at_cutoff > 0
                            ? (desired_write_cost - cost_above) / cost_at_cutoff
                            : 0;
      tie_probability = std::max(0.0, std::min(1.0, tie_probability));
    }
  }

  // Use the cost-group cutoff when available, otherwise the global cutoff.
  // Halve it when the fraction spent is below the fraction of evaluation time
  // elapsed.
  AdmissionCutoff select_admission_cutoff(int64_t cost, int64_t now) const {
    AdmissionCutoff cutoff{threshold, tie_probability, has_threshold};
    if (cost > 0 && has_threshold) {
      const auto &band = bands[write_cost_band(cost)];
      if (band.ready)
        cutoff = band;
    }
    if (cost > 0 && cutoff.ready &&
        (__int128)spent * (end_time - start_time) <
            (__int128)budget * (now - start_time))
      cutoff.score *= 0.5;
    return cutoff;
  }

  static int write_cost_band(int64_t cost);
  void update_cost_band_cutoffs(const std::vector<PricedOffer> &priced_offers,
                                double affordable_fraction);
};

// Cost-group cutoffs: use a separate cutoff for each sufficiently populated
// price range.
inline int WriteController::write_cost_band(int64_t cost) {
  return cost > 0 ? 63 - __builtin_clzll(static_cast<unsigned long long>(cost))
                  : 0;
}

inline void WriteController::update_cost_band_cutoffs(
    const std::vector<PricedOffer> &priced_offers, double affordable_fraction) {
  bands = {};
  std::array<std::vector<PricedOffer>, 64> grouped;
  for (const auto &v : priced_offers)
    grouped[write_cost_band(v.cost)].push_back(v);
  for (int band = 0; band < 64; ++band) {
    auto &group = grouped[band];
    if (group.size() < 100)
      continue;
    auto &cut = bands[band];
    cut.ready = true;
    if (affordable_fraction <= 0) {
      cut.score = INFINITY;
      cut.tie_probability = 0;
      continue;
    }
    if (affordable_fraction >= 1) {
      cut.score = -INFINITY;
      cut.tie_probability = 1;
      continue;
    }
    std::sort(
        group.begin(), group.end(),
        [](const PricedOffer &left_offer, const PricedOffer &right_offer) {
          return left_offer.score > right_offer.score;
        });
    int64_t band_write_cost = 0, cost_above = 0;
    for (const auto &v : group)
      band_write_cost += v.cost;
    double desired_write_cost = affordable_fraction * band_write_cost;
    for (size_t score_index = 0; score_index < group.size();) {
      size_t next_score_index = score_index;
      int64_t cost_at_cutoff = 0;
      while (next_score_index < group.size() &&
             group[next_score_index].score == group[score_index].score) {
        cost_at_cutoff += group[next_score_index].cost;
        ++next_score_index;
      }
      if (cost_above + cost_at_cutoff >= desired_write_cost) {
        cut.score = group[score_index].score;
        cut.tie_probability = std::clamp(
            (desired_write_cost - cost_above) / cost_at_cutoff, 0.0, 1.0);
        break;
      }
      cost_above += cost_at_cutoff;
      score_index = next_score_index;
    }
  }
}
