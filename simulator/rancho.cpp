// Replay the recorded RANCHO HBF policy using predictions fitted in Python.
// Walkthrough: read prefix -> identify reuse -> count hits -> price suffix
// -> decide -> write -> apply reuse updates.
// process_request() connects these steps; run() repeats them in arrival order.
// write_controller.hpp holds cutoffs, cache.hpp holds storage, simulation_io.hpp holds files.
// Capacities and costs are blocks; times are microseconds.
#include "simulation_io.hpp"
#include "write_controller.hpp"

// Read flash then HBM until the first missing block; return the suffix start.
static size_t read_cached_prefix(const std::vector<int32_t> &blocks, Cache &flash,
                                 Cache &hbm, std::vector<uint8_t> &sources) {
  sources.assign(blocks.size(), 0);
  size_t first_missing = 0;
  for (; first_missing < blocks.size(); first_missing++) {
    int32_t block = blocks[first_missing];
    if (flash.has(block)) {
      sources[first_missing] = 1;
      flash.touch(block);
    } else if (hbm.has(block)) {
      sources[first_missing] = 2;
      hbm.touch(block);
    } else
      break;
  }
  return first_missing;
}

// Within-session eligible reads and the hits served from each tier.
struct RequestHits {
  int64_t flash = 0, hbm = 0, eligible = 0;
};

// Identify same-session reuse and queue bonus updates for after admission.
static void identify_session_reuse(const std::vector<int32_t> &blocks, size_t request,
                                   int32_t session, size_t hbm_only_prefix,
                                   std::vector<int32_t> &seen_in_request,
                                   std::vector<int32_t> &last_session,
                                   std::vector<uint8_t> &session_repeats,
                                   std::vector<int32_t> &pending_reuse) {
  session_repeats.assign(blocks.size(), 0);
  pending_reuse.clear();
  for (size_t position = 0; position < blocks.size(); position++) {
    int32_t block = blocks[position];
    if (seen_in_request[block] == int32_t(request))
      continue;
    bool repeat = last_session[block] == session;
    session_repeats[position] = repeat;
    if (repeat && position >= hbm_only_prefix)
      pending_reuse.push_back(block);
    seen_in_request[block] = int32_t(request);
    last_session[block] = session;
  }
}

// Count eligible reads and cache hits using the previously marked repeats.
static RequestHits count_session_hits(const std::vector<uint8_t> &session_repeats,
                                      size_t first_missing,
                                      const std::vector<uint8_t> &sources) {
  RequestHits hits;
  for (size_t position = 0; position < session_repeats.size(); position++) {
    if (!session_repeats[position])
      continue;
    ++hits.eligible;
    if (position < first_missing) {
      hits.flash += sources[position] == 1;
      hits.hbm += sources[position] == 2;
    }
  }
  return hits;
}

// The suffix block count, exact write charge, and prior reuse.
struct SuffixOffer {
  int64_t distinct_blocks = 0, write_cost = 0, earlier_reuses = 0;
};

// Price the whole suffix without changing the cache.
static SuffixOffer price_suffix(const std::vector<int32_t> &blocks,
                                size_t first_missing, size_t request, Cache &flash,
                                std::vector<int32_t> &seen_in_suffix,
                                const std::vector<uint32_t> &reuse_counts) {
  SuffixOffer offer;
  for (size_t position = first_missing; position < blocks.size(); position++) {
    int32_t block = blocks[position];
    if (seen_in_suffix[block] != int32_t(request)) {
      seen_in_suffix[block] = int32_t(request);
      offer.distinct_blocks++;
      offer.earlier_reuses += reuse_counts[block];
    }
  }
  offer.write_cost =
      first_missing < blocks.size() ? flash.dry_cost(blocks, first_missing) : 0;
  return offer;
}

// Admission result and the score and cutoff used for it.
struct AdmissionDecision {
  bool admitted = false, budget_denied = false;
  AdmissionReason reason = AdmissionReason::NoSuffix;
  double score = INFINITY;
  AdmissionCutoff cutoff;
  bool requested() const { return admitted || budget_denied; }
};

// Score the suffix, enforce the budget, and compare against the cutoff.
static AdmissionDecision decide_admission(const SuffixOffer &offer, double prediction,
                                          size_t request, int64_t now, uint64_t seed,
                                          const WriteController &controller) {
  AdmissionDecision decision;
  decision.score = offer.write_cost > 0 ? prediction * double(offer.distinct_blocks) /
                                              double(offer.write_cost)
                                        : INFINITY;
  if (offer.write_cost > 0 && offer.distinct_blocks > 0) {
    const double boost =
        1.0 + double(offer.earlier_reuses) / double(offer.distinct_blocks);
    decision.score *= boost;
  }
  decision.cutoff = controller.cutoff_for(offer.write_cost, now);
  if (offer.distinct_blocks == 0)
    decision.reason = AdmissionReason::NoSuffix;
  else if (offer.write_cost == 0) {
    decision.reason = AdmissionReason::FreeWrite;
    decision.admitted = true;
  } else if (offer.write_cost > controller.budget - controller.spent) {
    decision.reason = AdmissionReason::BudgetDenied;
    decision.budget_denied = true;
  } else if (!controller.has_threshold) {
    decision.reason = AdmissionReason::StartupWrite;
    decision.admitted = true;
  } else if (decision.score > decision.cutoff.score ||
             (decision.score == decision.cutoff.score &&
              random_value_for_request(request, seed ^ 0x5bf03635ULL) <
                  decision.cutoff.tie_probability)) {
    decision.reason = AdmissionReason::AboveThreshold;
    decision.admitted = true;
  } else
    decision.reason = AdmissionReason::BelowThreshold;
  return decision;
}

// Apply the decision to both caches and return the flash write charge.
static int64_t place_suffix(const std::vector<int32_t> &blocks, size_t first_missing,
                            bool admitted, Cache &flash, Cache &hbm) {
  int64_t charged = 0;
  for (size_t position = first_missing; position < blocks.size(); position++) {
    int32_t block = blocks[position];
    if (admitted) {
      charged += flash.put(block);
      hbm.erase(block);
    }
    if (!flash.has(block))
      hbm.put(block);
  }
  return charged;
}

// Own the caches, controller, histories, and results for one replay.
class Simulator {
  const SimulationSettings &settings;
  const TraceInputs &input;
  Cache flash, hbm;
  WriteController controller;
  ReplayOutput output;
  Metrics evaluation;

  std::vector<int32_t> last_session, seen_in_request, seen_in_suffix;
  std::vector<uint32_t> reuse_counts;
  std::vector<OfferHistory> offer_history;
  std::vector<int32_t> blocks, pending_reuse;
  std::vector<uint8_t> sources, session_repeats;
  std::ifstream block_file;
  int64_t spent = 0, block_cursor = 0;
  size_t next_request = 0;
  bool evaluation_started = false;

  // Collect earlier reuse history while keeping physical caches empty.
  void observe_before_evaluation(size_t request, size_t hbm_only_prefix) {
    for (size_t position = 0; position < blocks.size(); position++) {
      int32_t block = blocks[position];
      if (seen_in_request[block] == int32_t(request))
        continue;
      if (position >= hbm_only_prefix && input.times[request] >= 0 &&
          last_session[block] == input.sessions[request]) {
        require_condition(reuse_counts[block] < UINT32_MAX, "reuse overflow");
        ++reuse_counts[block];
      }
      seen_in_request[block] = int32_t(request);
      last_session[block] = input.sessions[request];
    }
    output.record_request(request, input.times[request], int64_t(blocks.size()), 0, 0,
                          0, 0, false, false, 0);
  }

  // Read and check the next request's block IDs.
  void read_request(size_t request) {
    blocks.resize(size_t(input.block_counts[request]));
    block_file.read(reinterpret_cast<char *>(blocks.data()),
                    std::streamsize(blocks.size() * 4));
    require_condition(bool(block_file), "short row");
    for (int32_t block : blocks)
      require_condition(block >= 0 && block < settings.block_id_limit,
                        "invalid block identity");
  }

  // Add one request's hit, write, and admission counts to the totals.
  void add_metrics(size_t first_missing, const RequestHits &hits,
                   const AdmissionDecision &decision, int64_t charged) {
    evaluation.requests++;
    evaluation.positions += blocks.size();
    evaluation.fetched += first_missing;
    evaluation.hbf_hits += hits.flash;
    evaluation.hbm_hits += hits.hbm;
    evaluation.eligible += hits.eligible;
    evaluation.writes += charged;
    evaluation.requested += decision.requested();
    evaluation.admitted += decision.admitted;
    evaluation.budget_rejected += decision.budget_denied;
    evaluation.no_offer += first_missing == blocks.size();
  }

  // Read, identify reuse, count hits, decide/write, then apply history updates.
  void process_request(size_t request, size_t hbm_only_prefix) {
    size_t first_missing = read_cached_prefix(blocks, flash, hbm, sources);
    identify_session_reuse(blocks, request, input.sessions[request], hbm_only_prefix,
                           seen_in_request, last_session, session_repeats,
                           pending_reuse);
    RequestHits hits = count_session_hits(session_repeats, first_missing, sources);

    AdmissionDecision decision;
    SuffixOffer offer;
    if (input.eligible[request]) {
      offer = price_suffix(blocks, first_missing, request, flash, seen_in_suffix,
                           reuse_counts);
      require_condition(controller.spent == spent, "wallet/cache spending mismatch");
      decision = decide_admission(offer, input.predicted_reuse[request], request,
                                  input.times[request], settings.seed, controller);
      output.record_decision({int64_t(request), input.times[request],
                              offer.distinct_blocks, offer.write_cost, spent,
                              input.predicted_reuse[request], decision.score,
                              decision.cutoff.score, decision.cutoff.tie_probability,
                              int64_t(decision.cutoff.ready), int64_t(decision.reason),
                              int64_t(decision.admitted)});
      offer_history[controller.next_history_slot] = {
          int64_t(request), offer.distinct_blocks, offer.write_cost,
          input.times[request]};
      controller.observe(decision.score, offer.write_cost, input.times[request]);
    } else
      output.reason_counts[size_t(AdmissionReason::Ineligible)]++;

    int64_t charged =
        place_suffix(blocks, first_missing, decision.admitted, flash, hbm);
    require_condition(charged == (decision.admitted ? offer.write_cost : 0),
                      "dry/commit write mismatch");
    spent += charged;
    require_condition(spent <= settings.budget, "hard wallet exceeded");
    controller.spent = spent;
    for (int32_t block : pending_reuse) {
      require_condition(reuse_counts[block] < UINT32_MAX, "reuse overflow");
      ++reuse_counts[block];
    }
    add_metrics(first_missing, hits, decision, charged);
    output.record_request(request, input.times[request], int64_t(blocks.size()),
                          first_missing, hits.flash, hits.hbm, charged,
                          decision.requested(), decision.admitted, spent);
    output.record_debug(request, first_missing, hits.flash, hits.hbm, charged, spent,
                        decision.requested(), decision.admitted, flash, hbm);
  }

public:
  // Create independent caches and accounting for this replay.
  Simulator(const SimulationSettings &settings, const TraceInputs &input)
      : settings(settings), input(input),
        flash(int32_t(settings.flash_capacity), int32_t(settings.block_id_limit)),
        hbm(int32_t(settings.hbm_capacity), int32_t(settings.block_id_limit)),
        controller(settings.budget, settings.start, settings.end),
        output(settings, input.times.size()),
        last_session(size_t(settings.block_id_limit), -1),
        seen_in_request(size_t(settings.block_id_limit), -1),
        seen_in_suffix(size_t(settings.block_id_limit), -1),
        reuse_counts(size_t(settings.block_id_limit), 0),
        offer_history(WriteController::history_capacity),
        block_file(settings.stream + "_ids.i32", std::ios::binary) {
    require_condition(bool(block_file), "block stream open");
  }

  // Process requests in arrival order and save the final results.
  void run() {
    for (size_t request = 0; request < input.times.size(); request++) {
      if (input.times[request] >= settings.end) {
        next_request = request;
        break;
      }
      if (!evaluation_started && input.times[request] >= settings.start) {
        output.snapshot("warm", flash, hbm, request, block_cursor, spent);
        output.record_cold_start();
        evaluation_started = true;
      }
      read_request(request);
      size_t hbm_only_prefix = size_t(input.hbm_prefix[request]);
      require_condition(hbm_only_prefix <= blocks.size(), "shadow prefix outside row");
      if (input.times[request] < settings.start)
        observe_before_evaluation(request, hbm_only_prefix);
      else
        process_request(request, hbm_only_prefix);
      block_cursor += blocks.size();
      next_request = request + 1;
    }
    if (!evaluation_started)
      output.snapshot("warm", flash, hbm, next_request, block_cursor, spent);
    output.snapshot("final", flash, hbm, next_request, block_cursor, spent);
    std::vector<OfferHistory> chronological_history;
    for (int64_t j = 0; j < controller.history_count; j++)
      chronological_history.push_back(offer_history[(controller.next_history_slot +
                                                     WriteController::history_capacity -
                                                     controller.history_count + j) %
                                                    WriteController::history_capacity]);
    output.finish(settings, evaluation, chronological_history, spent, controller.spent,
                  controller.observations, next_request, block_cursor);
  }
};

// Load settings and inputs, then start the replay.
int main(int argc, char **argv) {
  try {
    SimulationSettings settings = read_settings(argc, argv);
    TraceInputs input(settings);
    Simulator(settings, input).run();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
