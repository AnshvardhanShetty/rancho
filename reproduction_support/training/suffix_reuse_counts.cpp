/* Shared original-policy suffix labels, one dataset per geometry/seed.
 * Every offer (including rejected ones) uses the physically audited prefix.
 * Main generation groups indexed future event rows via hourly row boundaries.
 * Full independent audit uses separately certified per-block hour histograms,
 * plus within-hour subtraction, and compares EVERY bucket/offer entry.
 * Current row is excluded. Fitting later masks all buckets at/after cutoff.
 */
#define main index_program_main
#include "reuse_index.cpp"
#undef main
#include <array>
// The decision-record layout written by record_write_offers.cpp.
struct Decision {
  I row, time, distinct, cost, W;
  double pred, score, threshold, tie;
  I have, reason, admit;
};
static_assert(sizeof(Decision) == 96);
// Add the offer's later eligible reads to their respective hours.
static void count_later_reads(int32_t *first, int32_t *last, I offer_row, int current,
                              const std::vector<I> &times,
                              const std::array<I, 169> &first_row_of_hour,
                              uint64_t *reads_by_hour, I request_count) {
  // Start after the offer so the current request cannot count as reuse.
  auto at = std::upper_bound(first, last, int32_t(offer_row));
  while (at < last) {
    int hour = times[*at] / H;
    require_condition(hour >= current && hour < 168, "future hour");
    auto stop = std::lower_bound(at, last, int32_t(first_row_of_hour[hour + 1]));
    // Add this block’s later reads to the hour when they happened.
    reads_by_hour[hour * request_count + offer_row] += stop - at;
    at = stop;
  }
}

// Independently count the same reads using the per-block hourly totals.
static void audit_later_reads(int block, int32_t *first, int32_t *last, I offer_row,
                              int current, const std::array<I, 169> &first_row_of_hour,
                              Map<uint64_t> &hour_offsets, Map<Hour> &hour_totals,
                              std::array<uint64_t, 168> &expected) {

  for (uint64_t j = hour_offsets[block]; j < hour_offsets[block + 1]; j++) {
    auto hour_total = hour_totals[j];
    if (int(hour_total.h) > current)
      expected[hour_total.h] += hour_total.count;
    else if (int(hour_total.h) == current) {
      auto lo = std::lower_bound(first, last, int32_t(first_row_of_hour[current]));
      auto past = std::upper_bound(lo, last, int32_t(offer_row));
      require_condition(uint64_t(past - lo) <= hour_total.count,
                        "within-hour subtraction");
      expected[current] += hour_total.count - (past - lo);
    }
  }
}

int main(int argc, char **argv) {
  try {
    require_condition(argc == 9,
                      "mode prefix N index calib_native old_native outdir start");
    std::string mode = argv[1], pre = argv[2], idx = argv[4], cal = argv[5],
                old = argv[6], out = argv[7];
    I N = std::stoll(argv[3]), start = std::stoll(argv[8]);
    bool build = mode == "build";
    require_condition(build || mode == "audit", "mode");
    std::vector<I> times;
    {
      std::ifstream f(pre + "_ts.i64", std::ios::binary);
      I t;
      while (read_record(f, t)) {
        if (t >= 168 * H)
          break;
        times.push_back(t);
      }
    }
    I request_count = times.size();
    // Locate hour boundaries and open the checked per-block read index.
    std::array<I, 169> first_row_of_hour;
    for (int h = 0; h <= 168; h++)
      first_row_of_hour[h] =
          std::lower_bound(times.begin(), times.end(), h * H) - times.begin();
    Map<uint64_t> event_offsets(idx + "/offsets.u64", N + 1, false),
        hour_offsets(idx + "/hour_offsets.u64", N + 1, false);
    Map<int32_t> events(idx + "/events.i32", event_offsets[N], false);
    Map<Hour> hour_totals(idx + "/hour_entries.bin", hour_offsets[N], false);
    Map<uint64_t> reads_by_hour(out + "/Y.u64", 168 * request_count, build);
    Map<int32_t> counts(out + "/c.i32", request_count, build);
    // Open the earlier replay logs, including each request's suffix start.
    std::ifstream ar(cal + "/rows.i64", std::ios::binary),
        br(old + "/rows.i64", std::ios::binary),
        ad(cal + "/decisions.bin", std::ios::binary),
        bd(old + "/decisions.bin", std::ios::binary),
        ns(pre + "_nb.i32", std::ios::binary), ids(pre + "_ids.i32", std::ios::binary);
    require_condition(ar && br && ad && bd && ns && ids, "input open");
    std::vector<int32_t> row;
    I offers = 0, positive = 0, checked = 0;
    uint64_t mass = 0;
    // For each offer, recover its distinct suffix blocks and count later reads.
    for (I offer_row = 0; offer_row < request_count; offer_row++) {
      int32_t n;
      require_condition(read_record(ns, n) && n >= 0, "count");
      row.resize(n);
      ids.read((char *)row.data(), I(n) * 4);
      require_condition(ids.gcount() == I(n) * 4, "IDs");
      std::array<I, 10> a, b;
      require_condition(read_record(br, b) && b[0] == offer_row &&
                            b[1] == times[offer_row] && b[2] == n,
                        "eval row identity");
      if (times[offer_row] < 72 * H)
        require_condition(read_record(ar, a) && a[0] == offer_row &&
                              a[1] == times[offer_row] && a[2] == n,
                          "cal row identity");
      if (times[offer_row] < start) {
        require_condition(counts[offer_row] == 0, "pre-training count");
        for (int h = 0; h < 168; h++)
          require_condition(reads_by_hour[h * request_count + offer_row] == 0,
                            "pre-training labels");
        continue;
      }
      auto rec = times[offer_row] < 72 * H ? a : b;
      Decision d;
      require_condition(read_record(times[offer_row] < 72 * H ? ad : bd, d) &&
                            d.row == offer_row && d.time == times[offer_row],
                        "offer identity");
      require_condition(rec[3] >= 0 && rec[3] <= n, "suffix start");
      std::vector<int32_t> suffix(row.begin() + rec[3], row.end());
      std::sort(suffix.begin(), suffix.end());
      suffix.erase(std::unique(suffix.begin(), suffix.end()), suffix.end());
      // Repeated block IDs in one offer contribute only once to its label denominator.
      require_condition(I(suffix.size()) == d.distinct, "suffix count");
      require_condition(d.cost >= 0 && (d.distinct == 0 || d.cost > 0),
                        "cost contract");
      if (build)
        counts[offer_row] = d.distinct;
      else
        require_condition(counts[offer_row] == d.distinct, "stored count");
      offers++;
      positive += d.distinct > 0;
      std::array<uint64_t, 168> expected{};
      int current = times[offer_row] / H;
      for (int block : suffix) {
        require_condition(block >= 0 && block < N, "suffixID");
        auto first = events.p + event_offsets[block],
             last = events.p + event_offsets[block + 1];
        if (build) {
          count_later_reads(first, last, offer_row, current, times, first_row_of_hour,
                            reads_by_hour.p, request_count);
        } else {
          audit_later_reads(block, first, last, offer_row, current, first_row_of_hour,
                            hour_offsets, hour_totals, expected);
        }
      }
      for (int h = 0; h < 168; h++) {
        if (!build)
          require_condition(reads_by_hour[h * request_count + offer_row] == expected[h],
                            "label bucket mismatch");
        if (h < current)
          require_condition(reads_by_hour[h * request_count + offer_row] == 0,
                            "label before offer");
        mass += reads_by_hour[h * request_count + offer_row];
        checked++;
      }
    }
    // Require complete log consumption and report how many entries were checked.
    char extra;
    require_condition(!ar.read(&extra, 1) && !br.read(&extra, 1) &&
                          !ad.read(&extra, 1) && !bd.read(&extra, 1),
                      "leftover native");
    std::cout << "{\"status\":\""
              << (build ? "BUILT_SHARED_SUFFIX_LABELS"
                        : "PASS_FULL_INDEPENDENT_SUFFIX_LABEL_AUDIT")
              << "\",\"rows\":" << request_count << ",\"offers\":" << offers
              << ",\"positive_suffixes\":" << positive
              << ",\"bucket_entries\":" << checked << ",\"label_credit_mass\":" << mass
              << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
