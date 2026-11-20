// Read inputs and save outputs for the HBF simulator
// SimulationSettings reads the command line and TraceInputs loads the request arrays
// ReplayOutput saves the write decisions and the hit and write counts

#pragma once

#include "cache.hpp"
#include <array>
#include <cmath>
#include <iomanip>

// Replay paths and limits; capacities are blocks and times are microseconds.
struct SimulationSettings {
  std::string stream;
  int64_t block_id_limit, flash_capacity, hbm_capacity, budget, start, end;
  fs::path output;
  uint64_t seed;
  std::string zero_before_path, zero_after_path, eligible_path, prediction_path, prefix_path;
  bool debug = false;
};

// Read the recorded policy arguments supplied by run_simulation.py.
static SimulationSettings read_settings(int argc, char **argv) {
  require_condition(argc >= 4 && std::string(argv[argc - 3]) == "--shadow" &&
                        std::string(argv[argc - 2]) == "shadow-linear",
                    "expected --shadow shadow-linear PREFIX");
  std::string prefix_path = argv[argc - 1];
  argc -= 3;
  for (const auto &flag : {"--half-behind", "--cost-bands", "--reuse-bonus"}) {
    require_condition(argc > 1 && std::string(argv[argc - 1]) == flag,
                      "expected recorded policy flag " + std::string(flag));
    --argc;
  }
  require_condition(argc == 16 || argc == 17, "arguments: use experiments/run_simulation.py");
  bool debug = argc == 17;
  require_condition(!debug || std::string(argv[16]) == "--debug", "debug flag");
  require_condition(std::string(argv[10]) == "v1", "format version");
  require_condition(std::string(argv[12]) == "--live-rancho", "live flag");

  SimulationSettings settings;
  settings.stream = argv[1];
  settings.block_id_limit = std::stoll(argv[2]);
  settings.flash_capacity = std::stoll(argv[3]);
  settings.hbm_capacity = std::stoll(argv[4]);
  settings.zero_before_path = argv[5];
  settings.zero_after_path = argv[6];
  settings.budget = std::stoll(argv[7]);
  settings.start = std::stoll(argv[8]);
  settings.end = std::stoll(argv[9]);
  settings.output = argv[11];
  settings.seed = std::stoull(argv[13]);
  settings.eligible_path = argv[14];
  settings.prediction_path = argv[15];
  settings.prefix_path = prefix_path;
  settings.debug = debug;
  require_condition(settings.block_id_limit > 0 && settings.block_id_limit <= INT32_MAX &&
                        settings.flash_capacity > 0 && settings.flash_capacity <= INT32_MAX &&
                        settings.hbm_capacity >= 0 && settings.hbm_capacity <= INT32_MAX,
                    "geometry");
  require_condition(settings.budget >= 0 && settings.start > 0 && settings.end > settings.start,
                    "time or budget");
  return settings;
}

// Aligned request metadata, predictions, and HBM-only prefix lengths.
struct TraceInputs {
  std::vector<int32_t> block_counts, sessions, hbm_prefix;
  std::vector<int64_t> times;
  std::vector<uint8_t> eligible;
  std::vector<double> predicted_reuse;

  // Load input arrays and check lengths, ordering, and values.
  explicit TraceInputs(const SimulationSettings &settings) {
    block_counts = read_binary_array<int32_t>(settings.stream + "_nb.i32");
    sessions = read_binary_array<int32_t>(settings.stream + "_sess.i32");
    times = read_binary_array<int64_t>(settings.stream + "_ts.i64");
    hbm_prefix = read_binary_array<int32_t>(settings.prefix_path);
    eligible = read_binary_array<uint8_t>(settings.eligible_path);
    predicted_reuse = read_binary_array<double>(settings.prediction_path);
    auto before = read_binary_array<uint8_t>(settings.zero_before_path);
    auto after = read_binary_array<uint8_t>(settings.zero_after_path);
    size_t count = block_counts.size();
    require_condition(count <= INT32_MAX && sessions.size() == count && times.size() == count &&
                          before.size() == count && after.size() == count &&
                          hbm_prefix.size() == count && eligible.size() == count &&
                          predicted_reuse.size() == count,
                      "metadata lengths");
    require_condition(std::is_sorted(times.begin(), times.end()), "timestamp order");
    int64_t total_blocks = 0;
    for (size_t row = 0; row < count; row++) {
      require_condition(block_counts[row] >= 0 && sessions[row] >= 0 && before[row] == 0 &&
                            after[row] == 0 && eligible[row] <= 1 &&
                            std::isfinite(predicted_reuse[row]) && predicted_reuse[row] >= 0,
                        "invalid input: this recipe requires the two zero placeholder "
                        "arrays");
      total_blocks += block_counts[row];
      require_condition(total_blocks >= 0, "count overflow");
    }
    require_condition(fs::file_size(settings.stream + "_ids.i32") == uint64_t(total_blocks) * 4,
                      "full block stream length");
  }
};

// Aggregate request, hit, write, and admission counters.
struct Metrics {
  int64_t requests = 0, positions = 0, fetched = 0, hbf_hits = 0, hbm_hits = 0, eligible = 0,
          writes = 0;
  int64_t requested = 0, admitted = 0, budget_rejected = 0, no_offer = 0;

  // Write the counters in the existing results format.
  void json(std::ostream &f) const {
    f << "{\"requests\":" << requests << ",\"positions\":" << positions
      << ",\"fetched_prefix_positions\":" << fetched << ",\"hbf_hits\":" << hbf_hits
      << ",\"hbm_hits\":" << hbm_hits << ",\"credited_hits\":" << hbf_hits + hbm_hits
      << ",\"eligible_first_distinct\":" << eligible << ",\"writes\":" << writes
      << ",\"requested_admissions\":" << requested << ",\"admitted\":" << admitted
      << ",\"budget_rejections\":" << budget_rejected << ",\"no_offer_requests\":" << no_offer
      << "}";
  }
};

// Original binary layout for a recorded suffix offer.
struct OfferHistory {
  int64_t row = -1, distinct = 0, cost = 0, time = 0;
};

// Original binary layout for a scored admission decision.
struct DecisionRecord {
  int64_t row, time, distinct, cost, spent;
  double predicted_reuse, score, threshold, tie_probability;
  int64_t has_threshold, reason, admitted;
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
  fs::path folder;
  std::ofstream rows, decisions, debug;
  std::vector<uint8_t> admitted_mask, requested_mask;
  std::array<int64_t, 8> reason_counts{};
  int64_t offered_cost = 0, rejected_cost = 0;

  // Create a new results folder and open the logs.
  ReplayOutput(const SimulationSettings &settings, size_t requests)
      : folder(settings.output), admitted_mask(requests, 0), requested_mask(requests, 0) {
    require_condition(!fs::exists(folder), "output already exists");
    require_condition(fs::create_directory(folder), "create output");
    rows.open(folder / "rows.i64", std::ios::binary);
    decisions.open(folder / "decisions.bin", std::ios::binary);
    require_condition(bool(rows) && bool(decisions), "open replay logs");
    if (settings.debug) {
      debug.open(folder / "debug.jsonl");
      require_condition(bool(debug), "debug open");
    }
  }

  // Save both cache orders and the current trace position.
  void snapshot(const std::string &name, const Cache &flash, const Cache &hbm,
                size_t next_request, int64_t block_cursor, int64_t spent) {
    auto flash_order = flash.order(), hbm_order = hbm.order();
    for (int32_t block : hbm_order)
      require_condition(!flash.has(block), "duplicate retained tier copy");
    write_binary_array(folder / (name + ".hbf.i32"), flash_order);
    write_binary_array(folder / (name + ".hbm.i32"), hbm_order);
    std::ofstream meta(folder / (name + ".json"));
    meta << "{\"next_row\":" << next_request << ",\"id_cursor\":" << block_cursor
         << ",\"writes\":" << spent << ",\"hbf_used\":" << flash_order.size()
         << ",\"hbm_used\":" << hbm_order.size()
         << ",\"history\":\"previous-reference stamps reconstructed from the "
            "immutable raw stream through this cursor\"}\n";
    meta.close();
    require_condition(bool(meta), "snapshot metadata");
  }

  // Save the empty controller state at the evaluation boundary.
  void record_cold_start() {
    write_binary_array(folder / "warm_history.bin", std::vector<OfferHistory>{});
    std::ofstream meta(folder / "warm_controller.json");
    meta << "{\"rn\":0,\"rp\":0,\"decisions\":0,\"W\":0}\n";
    meta.close();
    require_condition(bool(meta), "controller metadata");
  }

  // Write one decision and update its diagnostic counts.
  void record_decision(const DecisionRecord &record) {
    decisions.write(reinterpret_cast<const char *>(&record), sizeof(record));
    reason_counts[size_t(record.reason)]++;
    offered_cost += record.cost;
    if (record.reason == int64_t(AdmissionReason::BelowThreshold))
      rejected_cost += record.cost;
  }

  // Save one request's hit, write, and cumulative-spending record.
  void record_request(size_t request, int64_t time, int64_t positions, size_t first,
                      int64_t flash_hits, int64_t hbm_hits, int64_t charged, bool requested,
                      bool admitted, int64_t spent) {
    int64_t record[10] = {int64_t(request),  time,     positions, int64_t(first),
                          flash_hits,        hbm_hits, charged,   int64_t(requested),
                          int64_t(admitted), spent};
    rows.write(reinterpret_cast<const char *>(record), sizeof(record));
    require_condition(bool(rows), "row output");
    admitted_mask[request] = admitted;
    requested_mask[request] = requested;
  }

  // Optionally save per-request cache contents for debugging.
  void record_debug(size_t request, size_t first, int64_t flash_hits, int64_t hbm_hits,
                    int64_t charged, int64_t spent, bool requested, bool admitted,
                    const Cache &flash, const Cache &hbm) {
    if (!debug.is_open())
      return;
    debug << "{\"row\":" << request << ",\"first\":" << first << ",\"hbf_hits\":" << flash_hits
          << ",\"hbm_hits\":" << hbm_hits << ",\"writes\":" << charged << ",\"W\":" << spent
          << ",\"requested\":" << requested << ",\"admitted\":" << admitted;
    for (auto item :
         {std::make_pair("hbf", flash.order()), std::make_pair("hbm", hbm.order())}) {
      debug << ",\"" << item.first << "\":[";
      for (size_t j = 0; j < item.second.size(); j++)
        debug << (j ? "," : "") << item.second[j];
      debug << "]";
    }
    debug << "}\n";
  }

  // Close logs, check accounting, and save final metrics.
  void finish(const SimulationSettings &settings, const Metrics &evaluation,
              const std::vector<OfferHistory> &history, int64_t spent, int64_t controller_spent,
              int64_t observations, size_t next_request, int64_t cursor) {
    write_binary_array(folder / "effective_mask.u8", admitted_mask);
    write_binary_array(folder / "requested_mask.u8", requested_mask);
    write_binary_array(folder / "final_history.bin", history);
    decisions.close();
    rows.close();
    require_condition(bool(decisions) && bool(rows), "close replay logs");
    if (debug.is_open()) {
      debug.close();
      require_condition(bool(debug), "close debug");
    }
    std::ofstream diag(folder / "diagnostic.json");
    diag << std::setprecision(17) << "{\"reason_counts\":[";
    for (size_t j = 0; j < reason_counts.size(); j++)
      diag << (j ? "," : "") << reason_counts[j];
    diag << "],\"offered_write_cost\":" << offered_cost
         << ",\"threshold_rejected_write_cost\":" << rejected_cost
         << ",\"wallet_W\":" << controller_spent << ",\"wallet_observations\":" << observations
         << ",\"remaining_budget\":" << settings.budget - spent << "}\n";
    diag.close();
    require_condition(bool(diag), "close diagnostic");
    require_condition(spent == evaluation.writes && spent <= settings.budget,
                      "evaluation write identity");
    require_condition(evaluation.hbf_hits + evaluation.hbm_hits <= evaluation.eligible &&
                          evaluation.eligible <= evaluation.positions,
                      "credit conservation");
    Metrics warm;
    std::ofstream result(folder / "metrics.json");
    result << "{\"schema\":\"hbf_replacement_replay_v1\",\"status\":\"PASS\","
              "\"N\":"
           << settings.block_id_limit << ",\"hbf_capacity\":" << settings.flash_capacity
           << ",\"hbm_capacity\":" << settings.hbm_capacity << ",\"budget\":" << settings.budget
           << ",\"split_us\":" << settings.start << ",\"end_us\":" << settings.end
           << ",\"W_pre\":0,\"W_test\":" << evaluation.writes << ",\"writes\":" << spent
           << ",\"remaining_after_warm\":" << settings.budget
           << ",\"next_row\":" << next_request << ",\"id_cursor\":" << cursor << ",\"warm\":";
    warm.json(result);
    result << ",\"evaluation\":";
    evaluation.json(result);
    result << "}\n";
    result.close();
    require_condition(bool(result), "close metrics");
    std::cout << "{\"status\":\"PASS\",\"evaluation_hits\":"
              << evaluation.hbf_hits + evaluation.hbm_hits
              << ",\"W_pre\":0,\"writes\":" << spent << "}\n";
  }
};
