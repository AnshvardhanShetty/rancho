// Files used by the HBF replay.
//
// parse_simulation_settings reads the paths, sizes and write budget.
// TraceInputs loads the request arrays, predictions and HBM-only prefix lengths.
// ReplayOutput saves decisions during replay and writes the totals at the end.

#pragma once

#include "cache.hpp"
#include <array>
#include <cmath>
#include <iomanip>

// The four arrays share the prefix passed in by Python.
struct TraceArrayPaths {
  std::string block_counts;
  std::string session_numbers;
  std::string arrival_times_us;
  std::string block_ids;

  explicit TraceArrayPaths(const std::string &stream_prefix)
      : block_counts(stream_prefix + "_block_counts.i32"),
        session_numbers(stream_prefix + "_session_numbers.i32"),
        arrival_times_us(stream_prefix + "_timestamps_us.i64"),
        block_ids(stream_prefix + "_block_ids.i32") {}
};

// Replay paths and limits; capacities are blocks and times are microseconds.
struct SimulationSettings {
  std::string stream_prefix;
  int64_t block_id_count, flash_capacity_blocks, hbm_capacity_blocks, write_budget_blocks,
      evaluation_start_us, evaluation_end_us;
  fs::path results_directory;
  uint64_t seed;
  std::string pre_evaluation_flags_path, evaluation_flags_path, eligible_requests_path,
      predictions_path, hbm_only_prefix_path;
  bool save_debug_log = false;
};

// Read the recorded policy arguments supplied by run_simulation.py.
static SimulationSettings parse_simulation_settings(int argument_count, char **arguments) {
  require_condition(argument_count >= 4 &&
                        std::string(arguments[argument_count - 3]) == "--shadow" &&
                        std::string(arguments[argument_count - 2]) == "shadow-linear",
                    "expected --shadow shadow-linear PREFIX");
  std::string hbm_only_prefix_path = arguments[argument_count - 1];
  argument_count -= 3;
  for (const auto &flag : {"--half-behind", "--cost-bands", "--reuse-bonus"}) {
    require_condition(argument_count > 1 && std::string(arguments[argument_count - 1]) == flag,
                      "expected recorded policy flag " + std::string(flag));
    --argument_count;
  }
  require_condition(argument_count == 16 || argument_count == 17,
                    "arguments: use experiments/run_simulation.py");
  bool save_debug_log = argument_count == 17;
  require_condition(!save_debug_log || std::string(arguments[16]) == "--debug", "debug flag");
  require_condition(std::string(arguments[10]) == "v1", "format version");
  require_condition(std::string(arguments[12]) == "--live-rancho", "live flag");

  SimulationSettings settings;
  settings.stream_prefix = arguments[1];
  settings.block_id_count = std::stoll(arguments[2]);
  settings.flash_capacity_blocks = std::stoll(arguments[3]);
  settings.hbm_capacity_blocks = std::stoll(arguments[4]);
  settings.pre_evaluation_flags_path = arguments[5];
  settings.evaluation_flags_path = arguments[6];
  settings.write_budget_blocks = std::stoll(arguments[7]);
  settings.evaluation_start_us = std::stoll(arguments[8]);
  settings.evaluation_end_us = std::stoll(arguments[9]);
  settings.results_directory = arguments[11];
  settings.seed = std::stoull(arguments[13]);
  settings.eligible_requests_path = arguments[14];
  settings.predictions_path = arguments[15];
  settings.hbm_only_prefix_path = hbm_only_prefix_path;
  settings.save_debug_log = save_debug_log;
  require_condition(
      settings.block_id_count > 0 && settings.block_id_count <= INT32_MAX &&
          settings.flash_capacity_blocks > 0 && settings.flash_capacity_blocks <= INT32_MAX &&
          settings.hbm_capacity_blocks >= 0 && settings.hbm_capacity_blocks <= INT32_MAX,
      "geometry");
  require_condition(settings.write_budget_blocks >= 0 && settings.evaluation_start_us > 0 &&
                        settings.evaluation_end_us > settings.evaluation_start_us,
                    "time or budget");
  return settings;
}

// Aligned request metadata, predictions, and HBM-only prefix lengths.
struct TraceInputs {
  std::string block_ids_path;
  std::vector<int32_t> block_counts, session_numbers, hbm_only_prefix_lengths;
  std::vector<int64_t> arrival_times_us;
  std::vector<uint8_t> eligible_for_admission;
  std::vector<double> predicted_reads_per_block;

  // Load input arrays and check lengths, ordering, and values.
  explicit TraceInputs(const SimulationSettings &settings) {
    const TraceArrayPaths trace_files(settings.stream_prefix);
    block_ids_path = trace_files.block_ids;
    block_counts = read_binary_array<int32_t>(trace_files.block_counts);
    session_numbers = read_binary_array<int32_t>(trace_files.session_numbers);
    arrival_times_us = read_binary_array<int64_t>(trace_files.arrival_times_us);
    hbm_only_prefix_lengths = read_binary_array<int32_t>(settings.hbm_only_prefix_path);
    eligible_for_admission = read_binary_array<uint8_t>(settings.eligible_requests_path);
    predicted_reads_per_block = read_binary_array<double>(settings.predictions_path);
    auto pre_evaluation_flags = read_binary_array<uint8_t>(settings.pre_evaluation_flags_path);
    auto evaluation_flags = read_binary_array<uint8_t>(settings.evaluation_flags_path);
    size_t request_count = block_counts.size();
    require_condition(request_count <= INT32_MAX && session_numbers.size() == request_count &&
                          arrival_times_us.size() == request_count &&
                          pre_evaluation_flags.size() == request_count &&
                          evaluation_flags.size() == request_count &&
                          hbm_only_prefix_lengths.size() == request_count &&
                          eligible_for_admission.size() == request_count &&
                          predicted_reads_per_block.size() == request_count,
                      "metadata lengths");
    require_condition(std::is_sorted(arrival_times_us.begin(), arrival_times_us.end()),
                      "timestamp order");
    int64_t total_block_positions = 0;
    for (size_t request_index = 0; request_index < request_count; request_index++) {
      require_condition(block_counts[request_index] >= 0 && session_numbers[request_index] >= 0 &&
                            pre_evaluation_flags[request_index] == 0 &&
                            evaluation_flags[request_index] == 0 &&
                            eligible_for_admission[request_index] <= 1 &&
                            std::isfinite(predicted_reads_per_block[request_index]) &&
                            predicted_reads_per_block[request_index] >= 0,
                        "invalid input: this recipe requires the two zero placeholder "
                        "arrays");
      total_block_positions += block_counts[request_index];
      require_condition(total_block_positions >= 0, "count overflow");
    }
    require_condition(fs::file_size(trace_files.block_ids) == uint64_t(total_block_positions) * 4,
                      "full block stream length");
  }
};

// Aggregate request, hit, write, and admission counters.
struct Metrics {
  int64_t request_count = 0, block_positions = 0, cached_prefix_positions = 0, hbf_hits = 0,
          hbm_hits = 0, eligible_reads = 0, blocks_written = 0;
  int64_t requested_admissions = 0, admitted_requests = 0, budget_rejections = 0,
          no_offer_requests = 0;

  // Write the counters in the existing results format.
  void write_metrics_json(std::ostream &output_stream) const {
    output_stream << "{\"requests\":" << request_count << ",\"positions\":" << block_positions
                  << ",\"fetched_prefix_positions\":" << cached_prefix_positions
                  << ",\"hbf_hits\":" << hbf_hits << ",\"hbm_hits\":" << hbm_hits
                  << ",\"credited_hits\":" << hbf_hits + hbm_hits
                  << ",\"eligible_first_distinct\":" << eligible_reads
                  << ",\"writes\":" << blocks_written
                  << ",\"requested_admissions\":" << requested_admissions
                  << ",\"admitted\":" << admitted_requests
                  << ",\"budget_rejections\":" << budget_rejections
                  << ",\"no_offer_requests\":" << no_offer_requests << "}";
  }
};

// Original binary layout for a recorded suffix offer.
struct OfferHistory {
  int64_t request_index = -1, distinct_block_count = 0, write_cost_blocks = 0,
          arrival_time_us = 0;
};

// Original binary layout for a scored admission decision.
struct DecisionRecord {
  int64_t request_index, arrival_time_us, distinct_block_count, write_cost_blocks,
      blocks_written_so_far;
  double predicted_reads_per_block, score, score_cutoff, tie_probability;
  int64_t cutoff_ready, decision_reason, admitted;
};

static_assert(sizeof(DecisionRecord) == 96, "decision record layout");
static_assert(sizeof(OfferHistory) == 32, "offer history layout");

// Diagnostic reason codes; code 4 is reserved for the removed reserve mode.
enum class AdmissionReason : int64_t {
  Ineligible = 0,
  NoSuffix = 1,
  FreeWrite = 2,
  BudgetDenied = 3,
  BelowThreshold = 5,
  StartupWrite = 6,
  AboveThreshold = 7
};

// Own the replay logs, masks, snapshots, and output counters.
struct ReplayOutput {
  fs::path results_directory;
  std::ofstream request_log, decision_log, debug_log;
  std::vector<uint8_t> admitted_mask, requested_mask;
  std::array<int64_t, 8> reason_counts{};
  int64_t total_offered_write_cost = 0, cutoff_rejected_write_cost = 0;

  // Create a new results folder and open the logs.
  ReplayOutput(const SimulationSettings &settings, size_t request_count)
      : results_directory(settings.results_directory), admitted_mask(request_count, 0),
        requested_mask(request_count, 0) {
    require_condition(!fs::exists(results_directory), "output already exists");
    require_condition(fs::create_directory(results_directory), "create output");
    request_log.open(results_directory / "rows.i64", std::ios::binary);
    decision_log.open(results_directory / "decisions.bin", std::ios::binary);
    require_condition(bool(request_log) && bool(decision_log), "open replay logs");
    if (settings.save_debug_log) {
      debug_log.open(results_directory / "debug.jsonl");
      require_condition(bool(debug_log), "debug open");
    }
  }

  // Save both cache orders and the current trace position.
  void save_cache_snapshot(const std::string &snapshot_name, const Cache &flash, const Cache &hbm,
                           size_t next_request, int64_t block_cursor,
                           int64_t blocks_written_so_far) {
    auto flash_order = flash.blocks_most_recent_first(),
         hbm_order = hbm.blocks_most_recent_first();
    for (int32_t block : hbm_order)
      require_condition(!flash.contains_block(block), "duplicate retained tier copy");
    write_binary_array(results_directory / (snapshot_name + ".hbf.i32"), flash_order);
    write_binary_array(results_directory / (snapshot_name + ".hbm.i32"), hbm_order);
    std::ofstream metadata_file(results_directory / (snapshot_name + ".json"));
    metadata_file << "{\"next_row\":" << next_request << ",\"id_cursor\":" << block_cursor
                  << ",\"writes\":" << blocks_written_so_far
                  << ",\"hbf_used\":" << flash_order.size()
                  << ",\"hbm_used\":" << hbm_order.size()
                  << ",\"history\":\"previous-reference stamps reconstructed from the "
                     "immutable raw stream through this cursor\"}\n";
    metadata_file.close();
    require_condition(bool(metadata_file), "snapshot metadata");
  }

  // Save the empty controller state at the evaluation boundary.
  void record_cold_start() {
    write_binary_array(results_directory / "warm_history.bin", std::vector<OfferHistory>{});
    std::ofstream metadata_file(results_directory / "warm_controller.json");
    metadata_file << "{\"rn\":0,\"rp\":0,\"decisions\":0,\"W\":0}\n";
    metadata_file.close();
    require_condition(bool(metadata_file), "controller metadata");
  }

  // Write one decision and update its diagnostic counts.
  void record_decision(const DecisionRecord &record) {
    decision_log.write(reinterpret_cast<const char *>(&record), sizeof(record));
    reason_counts[size_t(record.decision_reason)]++;
    total_offered_write_cost += record.write_cost_blocks;
    if (record.decision_reason == int64_t(AdmissionReason::BelowThreshold))
      cutoff_rejected_write_cost += record.write_cost_blocks;
  }

  // Save one request's hit, write, and cumulative-spending record.
  void record_request(size_t request_index, int64_t arrival_time_us, int64_t block_position_count,
                      size_t first_missing_position, int64_t flash_hits, int64_t hbm_hits,
                      int64_t blocks_written_for_request, bool write_requested,
                      bool write_admitted, int64_t blocks_written_so_far) {
    int64_t record[10] = {int64_t(request_index),
                          arrival_time_us,
                          block_position_count,
                          int64_t(first_missing_position),
                          flash_hits,
                          hbm_hits,
                          blocks_written_for_request,
                          int64_t(write_requested),
                          int64_t(write_admitted),
                          blocks_written_so_far};
    request_log.write(reinterpret_cast<const char *>(record), sizeof(record));
    require_condition(bool(request_log), "row output");
    admitted_mask[request_index] = write_admitted;
    requested_mask[request_index] = write_requested;
  }

  // Optionally save per-request cache contents for debugging.
  void record_debug(size_t request_index, size_t first_missing_position, int64_t flash_hits,
                    int64_t hbm_hits, int64_t blocks_written_for_request,
                    int64_t blocks_written_so_far, bool write_requested, bool write_admitted,
                    const Cache &flash, const Cache &hbm) {
    if (!debug_log.is_open())
      return;
    debug_log << "{\"row\":" << request_index << ",\"first\":" << first_missing_position
              << ",\"hbf_hits\":" << flash_hits << ",\"hbm_hits\":" << hbm_hits
              << ",\"writes\":" << blocks_written_for_request
              << ",\"W\":" << blocks_written_so_far << ",\"requested\":" << write_requested
              << ",\"admitted\":" << write_admitted;
    for (auto cache_contents : {std::make_pair("hbf", flash.blocks_most_recent_first()),
                                std::make_pair("hbm", hbm.blocks_most_recent_first())}) {
      debug_log << ",\"" << cache_contents.first << "\":[";
      for (size_t entry_index = 0; entry_index < cache_contents.second.size(); entry_index++)
        debug_log << (entry_index ? "," : "") << cache_contents.second[entry_index];
      debug_log << "]";
    }
    debug_log << "}\n";
  }

  // Close logs, check accounting, and save final metrics.
  void write_replay_results(const SimulationSettings &settings, const Metrics &evaluation,
                            const std::vector<OfferHistory> &offer_history_in_arrival_order,
                            int64_t blocks_written_so_far, int64_t controller_blocks_written,
                            int64_t offers_seen, size_t next_request, int64_t block_cursor) {
    write_binary_array(results_directory / "effective_mask.u8", admitted_mask);
    write_binary_array(results_directory / "requested_mask.u8", requested_mask);
    write_binary_array(results_directory / "final_history.bin", offer_history_in_arrival_order);
    decision_log.close();
    request_log.close();
    require_condition(bool(decision_log) && bool(request_log), "close replay logs");
    if (debug_log.is_open()) {
      debug_log.close();
      require_condition(bool(debug_log), "close debug");
    }
    std::ofstream diagnostics_file(results_directory / "diagnostic.json");
    diagnostics_file << std::setprecision(17) << "{\"reason_counts\":[";
    for (size_t entry_index = 0; entry_index < reason_counts.size(); entry_index++)
      diagnostics_file << (entry_index ? "," : "") << reason_counts[entry_index];
    diagnostics_file << "],\"offered_write_cost\":" << total_offered_write_cost
                     << ",\"threshold_rejected_write_cost\":" << cutoff_rejected_write_cost
                     << ",\"wallet_W\":" << controller_blocks_written
                     << ",\"wallet_observations\":" << offers_seen << ",\"remaining_budget\":"
                     << settings.write_budget_blocks - blocks_written_so_far << "}\n";
    diagnostics_file.close();
    require_condition(bool(diagnostics_file), "close diagnostic");
    require_condition(blocks_written_so_far == evaluation.blocks_written &&
                          blocks_written_so_far <= settings.write_budget_blocks,
                      "evaluation write identity");
    require_condition(evaluation.hbf_hits + evaluation.hbm_hits <= evaluation.eligible_reads &&
                          evaluation.eligible_reads <= evaluation.block_positions,
                      "credit conservation");
    Metrics warm;
    std::ofstream metrics_file(results_directory / "metrics.json");
    metrics_file << "{\"schema\":\"hbf_replacement_replay_v1\",\"status\":\"PASS\","
                    "\"N\":"
                 << settings.block_id_count
                 << ",\"hbf_capacity\":" << settings.flash_capacity_blocks
                 << ",\"hbm_capacity\":" << settings.hbm_capacity_blocks
                 << ",\"budget\":" << settings.write_budget_blocks
                 << ",\"split_us\":" << settings.evaluation_start_us
                 << ",\"end_us\":" << settings.evaluation_end_us
                 << ",\"W_pre\":0,\"W_test\":" << evaluation.blocks_written
                 << ",\"writes\":" << blocks_written_so_far
                 << ",\"remaining_after_warm\":" << settings.write_budget_blocks
                 << ",\"next_row\":" << next_request << ",\"id_cursor\":" << block_cursor
                 << ",\"warm\":";
    warm.write_metrics_json(metrics_file);
    metrics_file << ",\"evaluation\":";
    evaluation.write_metrics_json(metrics_file);
    metrics_file << "}\n";
    metrics_file.close();
    require_condition(bool(metrics_file), "close metrics");
    std::cout << "{\"status\":\"PASS\",\"evaluation_hits\":"
              << evaluation.hbf_hits + evaluation.hbm_hits
              << ",\"W_pre\":0,\"writes\":" << blocks_written_so_far << "}\n";
  }
};
