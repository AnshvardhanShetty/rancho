// Replay requests using the predictions saved by Python.
//
// Start at Simulator::process_request to follow one request:
// 1. Read the cached prefix and count same-session hits.
// 2. Price the remaining suffix and decide whether to write it to flash.
// 3. Update the caches, charge the write and save reuse for later decisions.
//
// replay_requests runs these steps in arrival order. Before the test starts,
// it builds the reuse history while the caches and write controller stay empty.

#include "simulation_io.hpp"
#include "write_controller.hpp"

enum class HitSource : uint8_t { Missing, Flash, Hbm };

// Read until the first miss. Its position is where the suffix starts.
static size_t read_cached_prefix(const std::vector<int32_t> &blocks, Cache &flash, Cache &hbm,
                                 std::vector<HitSource> &prefix_hit_sources) {
  prefix_hit_sources.assign(blocks.size(), HitSource::Missing);
  size_t first_missing = 0;
  for (; first_missing < blocks.size(); first_missing++) {
    int32_t block_id = blocks[first_missing];
    if (flash.contains_block(block_id)) {
      prefix_hit_sources[first_missing] = HitSource::Flash;
      flash.mark_block_used(block_id);
    } else if (hbm.contains_block(block_id)) {
      prefix_hit_sources[first_missing] = HitSource::Hbm;
      hbm.mark_block_used(block_id);
    } else
      break;
  }
  return first_missing;
}

// Within-session eligible reads and the hits served from each tier.
struct RequestHits {
  int64_t flash_hits = 0, hbm_hits = 0, eligible_reads = 0;
};

// Mark each block that was last seen in this session. Only mark it once.
// Keep repeats beyond the HBM-only prefix for the next reuse-history update.
static void identify_session_reuse(const std::vector<int32_t> &blocks, size_t request_index,
                                   int32_t session, size_t hbm_only_readable_blocks,
                                   std::vector<int32_t> &last_request_by_block,
                                   std::vector<int32_t> &last_session_by_block,
                                   std::vector<uint8_t> &session_repeats,
                                   std::vector<int32_t> &blocks_to_credit) {
  session_repeats.assign(blocks.size(), 0);
  blocks_to_credit.clear();
  for (size_t position = 0; position < blocks.size(); position++) {
    int32_t block_id = blocks[position];
    if (last_request_by_block[block_id] == int32_t(request_index))
      continue;
    bool repeat = last_session_by_block[block_id] == session;
    session_repeats[position] = repeat;
    if (repeat && position >= hbm_only_readable_blocks)
      blocks_to_credit.push_back(block_id);
    last_request_by_block[block_id] = int32_t(request_index);
    last_session_by_block[block_id] = session;
  }
}

// Count eligible reads and cache hits using the previously marked repeats.
static RequestHits count_session_hits(const std::vector<uint8_t> &session_repeats,
                                      size_t first_missing,
                                      const std::vector<HitSource> &prefix_hit_sources) {
  RequestHits hits;
  for (size_t position = 0; position < session_repeats.size(); position++) {
    if (!session_repeats[position])
      continue;
    ++hits.eligible_reads;
    if (position < first_missing) {
      hits.flash_hits += prefix_hit_sources[position] == HitSource::Flash;
      hits.hbm_hits += prefix_hit_sources[position] == HitSource::Hbm;
    }
  }
  return hits;
}

// The suffix block count, exact write charge, and prior reuse.
struct SuffixOffer {
  int64_t distinct_blocks = 0, write_cost = 0, earlier_reuses = 0;
};

// Count distinct blocks and past reads. Try the write and roll it back to measure its cost.
static SuffixOffer price_suffix(const std::vector<int32_t> &blocks, size_t first_missing,
                                size_t request_index, Cache &flash,
                                std::vector<int32_t> &last_offer_by_block,
                                const std::vector<uint32_t> &eligible_reuses_by_block) {
  SuffixOffer offer;
  for (size_t position = first_missing; position < blocks.size(); position++) {
    int32_t block_id = blocks[position];
    if (last_offer_by_block[block_id] != int32_t(request_index)) {
      last_offer_by_block[block_id] = int32_t(request_index);
      offer.distinct_blocks++;
      offer.earlier_reuses += eligible_reuses_by_block[block_id];
    }
  }
  offer.write_cost =
      first_missing < blocks.size() ? flash.measure_suffix_write_cost(blocks, first_missing) : 0;
  return offer;
}

// Admission result and the score and cutoff used for it.
struct AdmissionDecision {
  bool admitted = false, budget_denied = false;
  AdmissionReason reason = AdmissionReason::NoSuffix;
  double score = INFINITY;
  AdmissionCutoff cutoff;

  bool requested_flash_write() const { return admitted || budget_denied; }
};

// Score the offer using predicted reuse and the history bonus. Then check the budget and cutoff.
static AdmissionDecision decide_admission(const SuffixOffer &offer,
                                          double predicted_reads_per_block, size_t request_index,
                                          int64_t arrival_us, uint64_t seed,
                                          const WriteController &controller) {
  AdmissionDecision decision;
  decision.score =
      offer.write_cost > 0
          ? predicted_reads_per_block * double(offer.distinct_blocks) / double(offer.write_cost)
          : INFINITY;
  if (offer.write_cost > 0 && offer.distinct_blocks > 0) {
    const double boost = 1.0 + double(offer.earlier_reuses) / double(offer.distinct_blocks);
    decision.score *= boost;
  }
  decision.cutoff = controller.select_admission_cutoff(offer.write_cost, arrival_us);
  if (offer.distinct_blocks == 0)
    decision.reason = AdmissionReason::NoSuffix;
  else if (offer.write_cost == 0) {
    decision.reason = AdmissionReason::FreeWrite;
    decision.admitted = true;
  } else if (offer.write_cost > controller.write_budget_blocks - controller.blocks_written) {
    decision.reason = AdmissionReason::BudgetDenied;
    decision.budget_denied = true;
  } else if (!controller.global_cutoff_ready) {
    decision.reason = AdmissionReason::StartupWrite;
    decision.admitted = true;
  } else if (decision.score > decision.cutoff.score_cutoff ||
             (decision.score == decision.cutoff.score_cutoff &&
              random_value_for_request(request_index, seed ^ 0x5bf03635ULL) <
                  decision.cutoff.tie_probability)) {
    decision.reason = AdmissionReason::AboveThreshold;
    decision.admitted = true;
  } else
    decision.reason = AdmissionReason::BelowThreshold;
  return decision;
}

// Apply the decision to both caches and return the flash write charge.
static int64_t store_suffix_in_cache_tiers(const std::vector<int32_t> &blocks,
                                           size_t first_missing, bool admitted, Cache &flash,
                                           Cache &hbm) {
  int64_t blocks_written_for_request = 0;
  for (size_t position = first_missing; position < blocks.size(); position++) {
    int32_t block_id = blocks[position];
    if (admitted) {
      blocks_written_for_request += flash.store_block(block_id);
      hbm.remove_block(block_id);
    }
    if (!flash.contains_block(block_id))
      hbm.store_block(block_id);
  }
  return blocks_written_for_request;
}

// Own the caches, controller, histories, and results for one replay.
class Simulator {
  const SimulationSettings &settings;
  const TraceInputs &trace_inputs;

  Cache flash, hbm;
  WriteController controller;

  ReplayOutput output;
  Metrics evaluation;

  std::vector<int32_t> last_session_by_block, last_request_by_block, last_offer_by_block;
  std::vector<uint32_t> eligible_reuses_by_block;
  std::vector<OfferHistory> offer_history;

  // Scratch space for the request being replayed.
  std::vector<int32_t> blocks, blocks_to_credit;
  std::vector<HitSource> prefix_hit_sources;
  std::vector<uint8_t> session_repeats;
  std::ifstream block_file;

  int64_t blocks_written = 0, block_cursor = 0;
  size_t next_request = 0;
  bool evaluation_started = false;

  // Collect earlier reuse history while keeping physical caches empty.
  void observe_before_evaluation(size_t request_index, size_t hbm_only_readable_blocks) {
    for (size_t position = 0; position < blocks.size(); position++) {
      int32_t block_id = blocks[position];
      if (last_request_by_block[block_id] == int32_t(request_index))
        continue;
      if (position >= hbm_only_readable_blocks &&
          trace_inputs.arrival_times_us[request_index] >= 0 &&
          last_session_by_block[block_id] == trace_inputs.session_numbers[request_index]) {
        require_condition(eligible_reuses_by_block[block_id] < UINT32_MAX, "reuse overflow");
        ++eligible_reuses_by_block[block_id];
      }
      last_request_by_block[block_id] = int32_t(request_index);
      last_session_by_block[block_id] = trace_inputs.session_numbers[request_index];
    }
    output.record_request(request_index, trace_inputs.arrival_times_us[request_index],
                          int64_t(blocks.size()), 0, 0, 0, 0, false, false, 0);
  }

  // Read and check the next request's block IDs.
  void read_request(size_t request_index) {
    blocks.resize(size_t(trace_inputs.block_counts[request_index]));
    block_file.read(reinterpret_cast<char *>(blocks.data()), std::streamsize(blocks.size() * 4));
    require_condition(bool(block_file), "short row");
    for (int32_t block_id : blocks)
      require_condition(block_id >= 0 && block_id < settings.block_id_count,
                        "invalid block identity");
  }

  // Add one request's hit, write, and admission counts to the totals.
  void record_request_metrics(size_t first_missing, const RequestHits &hits,
                              const AdmissionDecision &decision,
                              int64_t blocks_written_for_request) {
    evaluation.request_count++;
    evaluation.block_positions += blocks.size();
    evaluation.cached_prefix_positions += first_missing;
    evaluation.hbf_hits += hits.flash_hits;
    evaluation.hbm_hits += hits.hbm_hits;
    evaluation.eligible_reads += hits.eligible_reads;
    evaluation.blocks_written += blocks_written_for_request;
    evaluation.requested_admissions += decision.requested_flash_write();
    evaluation.admitted_requests += decision.admitted;
    evaluation.budget_rejections += decision.budget_denied;
    evaluation.no_offer_requests += first_missing == blocks.size();
  }

  void process_request(size_t request_index, size_t hbm_only_readable_blocks) {
    const int64_t arrival_us = trace_inputs.arrival_times_us[request_index];

    // Read from the caches and count eligible hits.
    size_t first_missing = read_cached_prefix(blocks, flash, hbm, prefix_hit_sources);
    identify_session_reuse(blocks, request_index, trace_inputs.session_numbers[request_index],
                           hbm_only_readable_blocks, last_request_by_block, last_session_by_block,
                           session_repeats, blocks_to_credit);
    RequestHits hits = count_session_hits(session_repeats, first_missing, prefix_hit_sources);

    // Score the write and record the offer, even if we decline it.
    AdmissionDecision decision;
    SuffixOffer offer;
    if (trace_inputs.eligible_for_admission[request_index]) {
      offer = price_suffix(blocks, first_missing, request_index, flash, last_offer_by_block,
                           eligible_reuses_by_block);
      require_condition(controller.blocks_written == blocks_written,
                        "wallet/cache spending mismatch");
      decision = decide_admission(offer, trace_inputs.predicted_reads_per_block[request_index],
                                  request_index, arrival_us, settings.seed, controller);
      output.record_decision({int64_t(request_index), arrival_us, offer.distinct_blocks,
                              offer.write_cost, blocks_written,
                              trace_inputs.predicted_reads_per_block[request_index],
                              decision.score, decision.cutoff.score_cutoff,
                              decision.cutoff.tie_probability, int64_t(decision.cutoff.is_ready),
                              int64_t(decision.reason), int64_t(decision.admitted)});
      offer_history[controller.next_history_slot] = {
          int64_t(request_index), offer.distinct_blocks, offer.write_cost, arrival_us};
      controller.record_priced_offer(decision.score, offer.write_cost, arrival_us);
    } else
      output.reason_counts[size_t(AdmissionReason::Ineligible)]++;

    // Update the caches and check that the actual charge matches the trial write.
    int64_t blocks_written_for_request =
        store_suffix_in_cache_tiers(blocks, first_missing, decision.admitted, flash, hbm);
    require_condition(blocks_written_for_request == (decision.admitted ? offer.write_cost : 0),
                      "dry/commit write mismatch");
    blocks_written += blocks_written_for_request;
    require_condition(blocks_written <= settings.write_budget_blocks, "hard wallet exceeded");
    controller.blocks_written = blocks_written;

    // This request's reuse can affect the next decision.
    for (int32_t block_id : blocks_to_credit) {
      require_condition(eligible_reuses_by_block[block_id] < UINT32_MAX, "reuse overflow");
      ++eligible_reuses_by_block[block_id];
    }

    record_request_metrics(first_missing, hits, decision, blocks_written_for_request);
    output.record_request(request_index, arrival_us, int64_t(blocks.size()), first_missing,
                          hits.flash_hits, hits.hbm_hits, blocks_written_for_request,
                          decision.requested_flash_write(), decision.admitted, blocks_written);
    output.record_debug(request_index, first_missing, hits.flash_hits, hits.hbm_hits,
                        blocks_written_for_request, blocks_written,
                        decision.requested_flash_write(), decision.admitted, flash, hbm);
  }

public:
  // Create independent caches and accounting for this replay.
  Simulator(const SimulationSettings &settings, const TraceInputs &trace_inputs)
      : settings(settings), trace_inputs(trace_inputs),
        flash(int32_t(settings.flash_capacity_blocks), int32_t(settings.block_id_count)),
        hbm(int32_t(settings.hbm_capacity_blocks), int32_t(settings.block_id_count)),
        controller(settings.write_budget_blocks, settings.evaluation_start_us,
                   settings.evaluation_end_us),
        output(settings, trace_inputs.arrival_times_us.size()),
        last_session_by_block(size_t(settings.block_id_count), -1),
        last_request_by_block(size_t(settings.block_id_count), -1),
        last_offer_by_block(size_t(settings.block_id_count), -1),
        eligible_reuses_by_block(size_t(settings.block_id_count), 0),
        offer_history(WriteController::history_capacity),
        block_file(trace_inputs.block_ids_path, std::ios::binary) {
    require_condition(bool(block_file), "block stream open");
  }

  // Process requests in arrival order and save the final results.
  void replay_requests() {
    for (size_t request_index = 0; request_index < trace_inputs.arrival_times_us.size();
         request_index++) {
      if (trace_inputs.arrival_times_us[request_index] >= settings.evaluation_end_us) {
        next_request = request_index;
        break;
      }
      if (!evaluation_started &&
          trace_inputs.arrival_times_us[request_index] >= settings.evaluation_start_us) {
        output.save_cache_snapshot("warm", flash, hbm, request_index, block_cursor,
                                   blocks_written);
        output.record_cold_start();
        evaluation_started = true;
      }
      read_request(request_index);
      size_t hbm_only_readable_blocks =
          size_t(trace_inputs.hbm_only_prefix_lengths[request_index]);
      require_condition(hbm_only_readable_blocks <= blocks.size(), "shadow prefix outside row");
      if (trace_inputs.arrival_times_us[request_index] < settings.evaluation_start_us)
        observe_before_evaluation(request_index, hbm_only_readable_blocks);
      else
        process_request(request_index, hbm_only_readable_blocks);
      block_cursor += blocks.size();
      next_request = request_index + 1;
    }
    if (!evaluation_started)
      output.save_cache_snapshot("warm", flash, hbm, next_request, block_cursor, blocks_written);
    output.save_cache_snapshot("final", flash, hbm, next_request, block_cursor, blocks_written);
    std::vector<OfferHistory> offers_in_arrival_order;
    for (int64_t history_offset = 0; history_offset < controller.history_count; history_offset++)
      offers_in_arrival_order.push_back(
          offer_history[(controller.next_history_slot + WriteController::history_capacity -
                         controller.history_count + history_offset) %
                        WriteController::history_capacity]);
    output.write_replay_results(settings, evaluation, offers_in_arrival_order, blocks_written,
                                controller.blocks_written, controller.offers_seen, next_request,
                                block_cursor);
  }
};

// Load settings and inputs, then start the replay.
int main(int argument_count, char **arguments) {
  try {
    SimulationSettings settings = parse_simulation_settings(argument_count, arguments);
    TraceInputs trace_inputs(settings);
    Simulator(settings, trace_inputs).replay_requests();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
