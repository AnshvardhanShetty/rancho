// Count later reads for every suffix offered by the reference replay
// Use the saved prefix to find the offer including offers the policy declined
// Write one count per hour so each fit uses only reads it could have seen
// Audit mode checks every count using the per-block hourly totals

#define main index_program_main
#include "reuse_index.cpp"
#undef main
#include <array>

// The decision-record layout written by record_write_offers.cpp.
struct Decision {
  int64_t row, time, distinct, cost, W;
  double pred, score, threshold, tie;
  int64_t have, reason, admit;
};

static_assert(sizeof(Decision) == 96);

// Add the offer's later eligible reads to their respective hours.
static void count_later_reads(int32_t *event_begin, int32_t *event_end,
                              int64_t offer_row, int offer_hour,
                              const std::vector<int64_t> &arrival_times_us,
                              const std::array<int64_t, 169> &first_row_of_hour,
                              uint64_t *reads_by_hour, int64_t request_count) {
  // Start after the offer so the current request cannot count as reuse.
  auto next_event =
      std::upper_bound(event_begin, event_end, int32_t(offer_row));
  while (next_event < event_end) {
    int hour = arrival_times_us[*next_event] / H;
    require_condition(hour >= offer_hour && hour < 168, "future hour");
    auto stop = std::lower_bound(next_event, event_end,
                                 int32_t(first_row_of_hour[hour + 1]));
    // Add this block’s later reads to the hour when they happened.
    reads_by_hour[hour * request_count + offer_row] += stop - next_event;
    next_event = stop;
  }
}

// Independently count the same reads using the per-block hourly totals.
static void audit_later_reads(int block, int32_t *event_begin,
                              int32_t *event_end, int64_t offer_row,
                              int offer_hour,
                              const std::array<int64_t, 169> &first_row_of_hour,
                              Map<uint64_t> &hour_offsets,
                              Map<Hour> &hour_totals,
                              std::array<uint64_t, 168> &expected) {

  for (uint64_t j = hour_offsets[block]; j < hour_offsets[block + 1]; j++) {
    auto hour_total = hour_totals[j];
    if (int(hour_total.h) > offer_hour)
      expected[hour_total.h] += hour_total.count;
    else if (int(hour_total.h) == offer_hour) {
      auto events_at_offer_hour = std::lower_bound(
          event_begin, event_end, int32_t(first_row_of_hour[offer_hour]));
      auto first_later_event =
          std::upper_bound(events_at_offer_hour, event_end, int32_t(offer_row));
      require_condition(uint64_t(first_later_event - events_at_offer_hour) <=
                            hour_total.count,
                        "within-hour subtraction");
      expected[offer_hour] +=
          hour_total.count - (first_later_event - events_at_offer_hour);
    }
  }
}

int main(int argc, char **argv) {
  try {
    require_condition(
        argc == 9, "mode prefix N index calib_native old_native outdir start");
    std::string mode = argv[1], stream_prefix = argv[2],
                reuse_index_directory = argv[4], calibration_replay = argv[5],
                evaluation_replay = argv[6], output_directory = argv[7];
    int64_t block_id_count = std::stoll(argv[3]),
            training_start_us = std::stoll(argv[8]);
    bool build = mode == "build";
    require_condition(build || mode == "audit", "mode");
    std::vector<int64_t> arrival_times_us;
    {
      std::ifstream arrival_times_file(stream_prefix + "_ts.i64",
                                       std::ios::binary);
      int64_t arrival_us;
      while (read_binary_record(arrival_times_file, arrival_us)) {
        if (arrival_us >= 168 * H)
          break;
        arrival_times_us.push_back(arrival_us);
      }
    }
    int64_t request_count = arrival_times_us.size();
    // Locate hour boundaries and open the checked per-block read index.
    std::array<int64_t, 169> first_row_of_hour;
    for (int h = 0; h <= 168; h++)
      first_row_of_hour[h] = std::lower_bound(arrival_times_us.begin(),
                                              arrival_times_us.end(), h * H) -
                             arrival_times_us.begin();
    Map<uint64_t> event_offsets(reuse_index_directory + "/offsets.u64",
                                block_id_count + 1, false),
        hour_offsets(reuse_index_directory + "/hour_offsets.u64",
                     block_id_count + 1, false);
    Map<int32_t> events(reuse_index_directory + "/events.i32",
                        event_offsets[block_id_count], false);
    Map<Hour> hour_totals(reuse_index_directory + "/hour_entries.bin",
                          hour_offsets[block_id_count], false);
    Map<uint64_t> reads_by_hour(output_directory + "/Y.u64",
                                168 * request_count, build);
    Map<int32_t> counts(output_directory + "/c.i32", request_count, build);
    // Open the earlier replay logs, including each request's suffix start.
    std::ifstream calibration_rows(calibration_replay + "/rows.i64",
                                   std::ios::binary),
        evaluation_rows(evaluation_replay + "/rows.i64", std::ios::binary),
        calibration_decisions(calibration_replay + "/decisions.bin",
                              std::ios::binary),
        evaluation_decisions(evaluation_replay + "/decisions.bin",
                             std::ios::binary),
        block_count_file(stream_prefix + "_nb.i32", std::ios::binary),
        ids(stream_prefix + "_ids.i32", std::ios::binary);
    require_condition(calibration_rows && evaluation_rows &&
                          calibration_decisions && evaluation_decisions &&
                          block_count_file && ids,
                      "input open");
    std::vector<int32_t> row;
    int64_t offers = 0, positive = 0, checked = 0;
    uint64_t total_label_reads = 0;
    // For each offer, recover its distinct suffix blocks and count later reads.
    for (int64_t offer_row = 0; offer_row < request_count; offer_row++) {
      int32_t request_block_count;
      require_condition(
          read_binary_record(block_count_file, request_block_count) &&
              request_block_count >= 0,
          "count");
      row.resize(request_block_count);
      ids.read((char *)row.data(), int64_t(request_block_count) * 4);
      require_condition(ids.gcount() == int64_t(request_block_count) * 4,
                        "IDs");
      std::array<int64_t, 10> calibration_row, evaluation_row;
      require_condition(read_binary_record(evaluation_rows, evaluation_row) &&
                            evaluation_row[0] == offer_row &&
                            evaluation_row[1] == arrival_times_us[offer_row] &&
                            evaluation_row[2] == request_block_count,
                        "eval row identity");
      if (arrival_times_us[offer_row] < 72 * H)
        require_condition(
            read_binary_record(calibration_rows, calibration_row) &&
                calibration_row[0] == offer_row &&
                calibration_row[1] == arrival_times_us[offer_row] &&
                calibration_row[2] == request_block_count,
            "cal row identity");
      if (arrival_times_us[offer_row] < training_start_us) {
        require_condition(counts[offer_row] == 0, "pre-training count");
        for (int h = 0; h < 168; h++)
          require_condition(reads_by_hour[h * request_count + offer_row] == 0,
                            "pre-training labels");
        continue;
      }
      auto replay_row = arrival_times_us[offer_row] < 72 * H ? calibration_row
                                                             : evaluation_row;
      Decision decision;
      require_condition(read_binary_record(arrival_times_us[offer_row] < 72 * H
                                               ? calibration_decisions
                                               : evaluation_decisions,
                                           decision) &&
                            decision.row == offer_row &&
                            decision.time == arrival_times_us[offer_row],
                        "offer identity");
      require_condition(replay_row[3] >= 0 &&
                            replay_row[3] <= request_block_count,
                        "suffix start");
      std::vector<int32_t> suffix(row.begin() + replay_row[3], row.end());
      std::sort(suffix.begin(), suffix.end());
      suffix.erase(std::unique(suffix.begin(), suffix.end()), suffix.end());
      // Repeated block IDs in one offer contribute only once to its label
      // denominator.
      require_condition(int64_t(suffix.size()) == decision.distinct,
                        "suffix count");
      require_condition(decision.cost >= 0 &&
                            (decision.distinct == 0 || decision.cost > 0),
                        "cost contract");
      if (build)
        counts[offer_row] = decision.distinct;
      else
        require_condition(counts[offer_row] == decision.distinct,
                          "stored count");
      offers++;
      positive += decision.distinct > 0;
      std::array<uint64_t, 168> expected{};
      int offer_hour = arrival_times_us[offer_row] / H;
      for (int block : suffix) {
        require_condition(block >= 0 && block < block_id_count, "suffixID");
        auto first = events.p + event_offsets[block],
             last = events.p + event_offsets[block + 1];
        if (build) {
          count_later_reads(first, last, offer_row, offer_hour,
                            arrival_times_us, first_row_of_hour,
                            reads_by_hour.p, request_count);
        } else {
          audit_later_reads(block, first, last, offer_row, offer_hour,
                            first_row_of_hour, hour_offsets, hour_totals,
                            expected);
        }
      }
      for (int h = 0; h < 168; h++) {
        if (!build)
          require_condition(reads_by_hour[h * request_count + offer_row] ==
                                expected[h],
                            "label bucket mismatch");
        if (h < offer_hour)
          require_condition(reads_by_hour[h * request_count + offer_row] == 0,
                            "label before offer");
        total_label_reads += reads_by_hour[h * request_count + offer_row];
        checked++;
      }
    }
    // Require complete log consumption and report how many entries were
    // checked.
    char extra;
    require_condition(!calibration_rows.read(&extra, 1) &&
                          !evaluation_rows.read(&extra, 1) &&
                          !calibration_decisions.read(&extra, 1) &&
                          !evaluation_decisions.read(&extra, 1),
                      "leftover native");
    std::cout << "{\"status\":\""
              << (build ? "BUILT_SHARED_SUFFIX_LABELS"
                        : "PASS_FULL_INDEPENDENT_SUFFIX_LABEL_AUDIT")
              << "\",\"rows\":" << request_count << ",\"offers\":" << offers
              << ",\"positive_suffixes\":" << positive
              << ",\"bucket_entries\":" << checked
              << ",\"label_credit_mass\":" << total_label_reads << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
