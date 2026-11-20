// Use the first predictor to replay the cache and see which suffix each request
// offers Save the readable prefix and the write decision including declined
// offers suffix_reuse_counts uses these files to build the final reuse labels
// rancho.cpp in the HBF simulator folder is the evaluation replay

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static void require_condition(bool b, const std::string &s) {
  if (!b)
    throw std::runtime_error(s);
}

template <class T>
static std::vector<T> read_binary_array(const std::string &p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  require_condition(bool(f), "cannot open " + p);
  auto bytes = f.tellg();
  require_condition(bytes >= 0 && bytes % sizeof(T) == 0,
                    "unaligned input " + p);
  std::vector<T> a(size_t(bytes) / sizeof(T));
  f.seekg(0);
  if (bytes)
    require_condition(bool(f.read(reinterpret_cast<char *>(a.data()), bytes)),
                      "short input " + p);
  return a;
}

template <class T>
static void write_binary_array(const fs::path &p, const std::vector<T> &a) {
  std::ofstream f(p, std::ios::binary);
  require_condition(bool(f), "cannot create " + p.string());
  f.write(reinterpret_cast<const char *>(a.data()),
          std::streamsize(a.size() * sizeof(T)));
  f.close();
  require_condition(bool(f), "output failed " + p.string());
}

// CACHE STORAGE: IDs and links ordered from most to least recently used.
struct Node {
  int32_t block = -1, prev = -1, next = -1;
};

struct Change {
  int32_t *p;
  int32_t value;
};

struct Cache {
  int32_t cap, used = 0, allocated = 0, free_head = -1, head = -1, tail = -1;
  std::vector<int32_t> slots;
  std::vector<Node> nodes;
  std::vector<Change> journal;
  bool logging = false;

  Cache(int32_t c, int32_t n) : cap(c), slots(c ? n : 0, -1), nodes(c) {}

  bool contains_block(int32_t b) const { return cap && slots[b] >= 0; }

  void set_and_record_for_undo(int32_t &x, int32_t v) {
    if (logging)
      journal.push_back({&x, x});
    x = v;
  }

  void detach_from_use_order(int32_t s) {
    int32_t p = nodes[s].prev, n = nodes[s].next;
    if (p >= 0)
      set_and_record_for_undo(nodes[p].next, n);
    else
      set_and_record_for_undo(head, n);
    if (n >= 0)
      set_and_record_for_undo(nodes[n].prev, p);
    else
      set_and_record_for_undo(tail, p);
  }

  void move_to_most_recent(int32_t s) {
    set_and_record_for_undo(nodes[s].prev, -1);
    set_and_record_for_undo(nodes[s].next, head);
    if (head >= 0)
      set_and_record_for_undo(nodes[head].prev, s);
    else
      set_and_record_for_undo(tail, s);
    set_and_record_for_undo(head, s);
  }

  void mark_block_used(int32_t b) {
    require_condition(contains_block(b), "touch absent block");
    int32_t s = slots[b];
    if (s != head) {
      detach_from_use_order(s);
      move_to_most_recent(s);
    }
  }

  void remove_block(int32_t b) {
    if (!contains_block(b))
      return;
    int32_t s = slots[b];
    detach_from_use_order(s);
    set_and_record_for_undo(slots[b], -1);
    set_and_record_for_undo(nodes[s].block, -1);
    set_and_record_for_undo(nodes[s].next, free_head);
    set_and_record_for_undo(free_head, s);
    set_and_record_for_undo(used, used - 1);
  }

  bool store_block(int32_t b) {
    if (!cap)
      return false;
    if (contains_block(b)) {
      mark_block_used(b);
      return false;
    }
    if (used == cap)
      remove_block(nodes[tail].block);
    int32_t s;
    if (free_head >= 0) {
      s = free_head;
      set_and_record_for_undo(free_head, nodes[s].next);
    } else {
      s = allocated;
      require_condition(allocated < cap, "slot overflow");
      set_and_record_for_undo(allocated, allocated + 1);
    }
    set_and_record_for_undo(nodes[s].block, b);
    set_and_record_for_undo(slots[b], s);
    set_and_record_for_undo(used, used + 1);
    move_to_most_recent(s);
    return true;
  }

  int64_t measure_suffix_write_cost(const std::vector<int32_t> &row,
                                    size_t first) {
    require_condition(!logging && journal.empty(), "nested transaction");
    logging = true;
    int64_t w = 0;
    for (size_t j = first; j < row.size(); j++)
      w += store_block(row[j]);
    logging = false;
    for (auto it = journal.rbegin(); it != journal.rend(); ++it)
      *it->p = it->value;
    journal.clear();
    return w;
  }

  std::vector<int32_t> blocks_most_recent_first() const {
    std::vector<int32_t> a;
    int32_t prev = -1;
    for (int32_t s = head; s >= 0; s = nodes[s].next) {
      require_condition(s < allocated && nodes[s].prev == prev,
                        "broken LRU links");
      int32_t b = nodes[s].block;
      require_condition(b >= 0 && contains_block(b) && slots[b] == s,
                        "broken LRU index");
      a.push_back(b);
      prev = s;
      require_condition(a.size() <= size_t(used), "LRU cycle");
    }
    require_condition(a.size() == size_t(used) && prev == tail && used <= cap,
                      "broken LRU size");
    return a;
  }
};

// REPLAY COUNTERS: requests, eligible hits, writes, and admission outcomes.
struct Metrics {
  int64_t requests = 0, positions = 0, fetched = 0, hbf_hits = 0, hbm_hits = 0,
          eligible = 0, writes = 0;
  int64_t requested = 0, admitted = 0, budget_rejected = 0, no_offer = 0;

  void json(std::ostream &f) const {
    f << "{\"requests\":" << requests << ",\"positions\":" << positions
      << ",\"fetched_prefix_positions\":" << fetched
      << ",\"hbf_hits\":" << hbf_hits << ",\"hbm_hits\":" << hbm_hits
      << ",\"credited_hits\":" << hbf_hits + hbm_hits
      << ",\"eligible_first_distinct\":" << eligible << ",\"writes\":" << writes
      << ",\"requested_admissions\":" << requested
      << ",\"admitted\":" << admitted
      << ",\"budget_rejections\":" << budget_rejected
      << ",\"no_offer_requests\":" << no_offer << "}";
  }
};

static uint64_t mix_hash(uint64_t v) {
  v ^= v >> 33;
  v *= 0xff51afd7ed558ccdULL;
  v ^= v >> 33;
  v *= 0xc4ceb9fe1a85ec53ULL;
  v ^= v >> 33;
  return v;
}

static double random_value_for_request(uint64_t row, uint64_t seed) {
  return double(mix_hash(row * 0x9E3779B97F4A7C15ULL ^ seed) >> 11) /
         9007199254740992.0;
}

// WRITE CONTROLLER: earlier offers and the remaining write allowance.
struct Observation {
  double score = 0;
  int64_t cost = 0, time = 0;
};

struct Wallet {
  int64_t budget, lo, hi, release, update, ring_size;
  double rho, release_fraction, threshold = -INFINITY, tie = 1;
  int64_t W = 0, cap, rn = 0, rp = 0, decisions = 0;
  bool released = false, have_threshold = false;
  int64_t positive = 0, free_admitted = 0, rej_budget = 0, rej_reserve = 0,
          rej_threshold = 0, zero_offer = 0;
  std::vector<Observation> ring;
  // Post-warm controller choices do not alter the stored ring or cache physics.
  // These fields are deliberately not checkpointed: tuned runs restart from
  // the pinned legacy warm checkpoint and cannot write a tuned checkpoint.
  double demand_multiplier = 1;
  int64_t recent_history = 0;
  bool count_quantile = false;
  // Named equal-score policy only; original warm/checkpoint bytes omit these.
  // Changed-policy checkpoint export is prohibited, so restart remains causal.
  int tie_order =
      0; // 0 legacy hash-only, 1 lower exact cost, 2 higher exact cost
  int64_t secondary_cost = 0;
  bool secondary_active = false;

  bool accepts_tied(double score, int64_t cost, int64_t row,
                    uint64_t seed) const {
    if (score > threshold)
      return true;
    if (score < threshold)
      return false;
    if (secondary_active && cost != secondary_cost)
      return tie_order == 1 ? cost < secondary_cost : cost > secondary_cost;
    return random_value_for_request(row, seed ^ 0x5bf03635ULL) < tie;
  }

  void update_tie_probabilities(int64_t now) {
    if (rn == 0)
      return;

    struct Weighted {
      double score;
      int64_t cost;
    };

    std::vector<Weighted> values;
    int64_t take = recent_history > 0 ? std::min(rn, recent_history) : rn;
    double sum = 0;
    int64_t tlo = INT64_MAX, thi = INT64_MIN;
    for (int64_t j = 0; j < take; j++) {
      int64_t i =
          recent_history > 0 ? (rp + ring_size - take + j) % ring_size : j;
      const auto &v = ring[i];
      tlo = std::min(tlo, v.time);
      thi = std::max(thi, v.time);
      if (v.cost > 0 && std::isfinite(v.score)) {
        values.push_back({v.score, v.cost});
        sum += v.cost;
      }
    }
    if (values.size() < 100)
      return;
    double span = std::max(double(thi - tlo) / 1e6, 1.0),
           rate = double(values.size()) / span, mean = sum / values.size();
    double end = double(released ? hi : release),
           horizon = std::max((end - double(now)) / 1e6, 1.0),
           remain = std::max(double(cap - W), 0.0);
    double p = std::max(
        0.0,
        std::min(1.0, demand_multiplier *
                          (remain / std::max(rate * horizon * mean, 1e-9))));
    secondary_active = false;
    secondary_cost = 0;
    if (p <= 0) {
      threshold = INFINITY;
      tie = 0;
    } else if (p >= 1) {
      threshold = -INFINITY;
      tie = 1;
    } else {
      std::sort(values.begin(), values.end(),
                [&](const Weighted &a, const Weighted &b) {
                  return a.score != b.score
                             ? a.score > b.score
                             : (tie_order == 1 ? a.cost < b.cost
                                               : a.cost > b.cost);
                });
      double wanted = p * sum, above = 0, equal = 0;
      for (size_t i = 0; i < values.size();) {
        size_t j = i;
        equal = 0;
        while (j < values.size() && values[j].score == values[i].score &&
               values[j].cost == values[i].cost) {
          equal += values[j].cost;
          j++;
        }
        if (above + equal >= wanted) {
          threshold = values[i].score;
          secondary_cost = values[i].cost;
          break;
        }
        above += equal;
        i = j;
      }
      tie = std::max(0.0,
                     std::min(1.0, equal > 0 ? (wanted - above) / equal : 0));
      secondary_active = true;
    }
    have_threshold = true;
  }

  Wallet(int64_t b, double r, double f, int64_t u, int64_t n, int64_t l,
         int64_t h)
      : budget(b), lo(l), hi(h), release(l + int64_t(f * double(h - l))),
        update(u), ring_size(n), rho(r), release_fraction(f),
        cap(int64_t(floor((1 - r) * double(b)))), ring(n) {}

  void release_reserved_budget_if_due(int64_t now) {
    if (!released && now >= release) {
      cap = budget;
      released = true;
      have_threshold = false;
    }
  }

  void record_priced_offer(double score, int64_t cost, int64_t now) {
    ring[rp] = {score, cost, now};
    rp = (rp + 1) % ring_size;
    if (rn < ring_size)
      rn++;
    decisions++;
    if (decisions % update != 0 || rn == 0)
      return;
    recalculate_admission_cutoff(now);
  }

  void recalculate_admission_cutoff(int64_t now) {
    if (tie_order) {
      update_tie_probabilities(now);
      return;
    }
    if (demand_multiplier != 1 || recent_history > 0 || count_quantile) {
      recalculate_configured_cutoffs(now);
      return;
    }
    if (rn == 0)
      return;

    struct Weighted {
      double score;
      int64_t cost;
    };

    std::vector<Weighted> weighted;
    weighted.reserve(rn);
    double sum = 0;
    int64_t tlo = ring[0].time, thi = tlo;
    for (int64_t i = 0; i < rn; i++) {
      tlo = std::min(tlo, ring[i].time);
      thi = std::max(thi, ring[i].time);
      if (ring[i].cost > 0 && std::isfinite(ring[i].score)) {
        weighted.push_back({ring[i].score, ring[i].cost});
        sum += ring[i].cost;
      }
    }
    if (weighted.size() < 100)
      return;
    double span = std::max(double(thi - tlo) / 1e6, 1.0),
           rate = double(weighted.size()) / span, mean = sum / weighted.size();
    double end = double(released ? hi : release),
           horizon = std::max((end - double(now)) / 1e6, 1.0),
           remain = std::max(double(cap - W), 0.0);
    double p = std::max(
        0.0, std::min(1.0, remain / std::max(rate * horizon * mean, 1e-9)));
    if (p <= 0) {
      threshold = INFINITY;
      tie = 0;
    } else if (p >= 1) {
      threshold = -INFINITY;
      tie = 1;
    } else {
      std::sort(weighted.begin(), weighted.end(),
                [](const Weighted &a, const Weighted &b) {
                  return a.score > b.score;
                });
      double wanted = p * sum, above = 0, equal = 0;
      threshold = weighted.back().score;
      for (size_t i = 0; i < weighted.size();) {
        size_t j = i;
        equal = 0;
        while (j < weighted.size() && weighted[j].score == weighted[i].score) {
          equal += weighted[j].cost;
          j++;
        }
        if (above + equal >= wanted) {
          threshold = weighted[i].score;
          break;
        }
        above += equal;
        i = j;
      }
      tie = equal > 0 ? (wanted - above) / equal : 0;
      tie = std::max(0.0, std::min(1.0, tie));
    }
    have_threshold = true;
  }

  void recalculate_configured_cutoffs(int64_t now) {
    if (rn == 0)
      return;

    struct Weighted {
      double score;
      int64_t cost;
    };

    std::vector<Weighted> values;
    int64_t take = recent_history > 0 ? std::min(rn, recent_history) : rn;
    double sum = 0;
    int64_t tlo = INT64_MAX, thi = INT64_MIN;
    // Select newest observations by ring position; include zero-cost rows in
    // the elapsed-time interval, exactly as in the full-window controller.
    for (int64_t j = 0; j < take; j++) {
      int64_t i = (rp + ring_size - take + j) % ring_size;
      const auto &v = ring[i];
      tlo = std::min(tlo, v.time);
      thi = std::max(thi, v.time);
      if (v.cost > 0 && std::isfinite(v.score)) {
        values.push_back({v.score, v.cost});
        sum += v.cost;
      }
    }
    if (values.size() < 100)
      return;
    double span = std::max(double(thi - tlo) / 1e6, 1.0),
           rate = double(values.size()) / span, mean = sum / values.size();
    double end = double(released ? hi : release),
           horizon = std::max((end - double(now)) / 1e6, 1.0),
           remain = std::max(double(cap - W), 0.0);
    double p = std::max(
        0.0,
        std::min(1.0, demand_multiplier *
                          (remain / std::max(rate * horizon * mean, 1e-9))));
    if (p <= 0) {
      threshold = INFINITY;
      tie = 0;
    } else if (p >= 1) {
      threshold = -INFINITY;
      tie = 1;
    } else {
      std::sort(values.begin(), values.end(),
                [](const Weighted &a, const Weighted &b) {
                  return a.score > b.score;
                });
      if (count_quantile) {
        // Historical request-count quantile and tie convention; only the
        // demand/window knobs above are new, explicitly named choices.
        int64_t m = values.size(),
                want = std::max<int64_t>(
                    1, std::min<int64_t>(m, std::llround(p * double(m))));
        threshold = values[want - 1].score;
        int64_t above = 0, equal = 0;
        for (const auto &v : values) {
          if (v.score > threshold)
            above++;
          else if (v.score == threshold)
            equal++;
        }
        tie = equal > 0 ? double(want - above) / double(equal) : 1;
      } else {
        double wanted = p * sum, above = 0, equal = 0;
        threshold = values.back().score;
        for (size_t i = 0; i < values.size();) {
          size_t j = i;
          equal = 0;
          while (j < values.size() && values[j].score == values[i].score) {
            equal += values[j].cost;
            j++;
          }
          if (above + equal >= wanted) {
            threshold = values[i].score;
            break;
          }
          above += equal;
          i = j;
        }
        tie = equal > 0 ? (wanted - above) / equal : 0;
      }
      tie = std::max(0.0, std::min(1.0, tie));
    }
    have_threshold = true;
  }
};

// SAVED RECORDS: binary layouts consumed by the reuse-count builder.
struct History {
  int64_t row = -1, distinct = 0, cost = 0, time = 0;
};

struct Decision {
  int64_t row, time, distinct, cost, W;
  double density, score, threshold, tie;
  int64_t have, reason, admit;
};

static_assert(sizeof(Decision) == 96, "decision record layout");

// REPLAY: load inputs, process cache reads/offers in arrival order, and save
// logs.
int main(int argc, char **argv) {
  try {

    bool early_half = argc > 1 && std::string(argv[argc - 1]) == "--early-half";
    if (early_half)
      --argc;
    bool reuse_bonus =
        argc > 1 && std::string(argv[argc - 1]) == "--reuse-bonus";
    if (reuse_bonus)
      --argc;
    bool initial_history = (argc == 18 || argc == 19);
    bool live = argc == 16 || argc == 17 || initial_history;
    bool debug_mode = (argc == 13 && std::string(argv[12]) == "--debug") ||
                      (argc == 17 && std::string(argv[16]) == "--debug") ||
                      (argc == 19 && std::string(argv[18]) == "--debug");
    require_condition(argc == 12 || argc == 13 || argc == 16 || argc == 17 ||
                          argc == 18 || argc == 19,
                      "arguments");
    require_condition((argc != 13 && argc != 17 && argc != 19) || debug_mode,
                      "debug flag");
    require_condition(!live || std::string(argv[12]) == "--live-rancho",
                      "live flag");
    require_condition(!initial_history ||
                          std::string(argv[16]) == "--initial-history",
                      "history flag");
    uint64_t seed = live ? std::stoull(argv[13]) : 0;
    auto policy_eligible =
        live ? read_binary_array<uint8_t>(argv[14]) : std::vector<uint8_t>{};
    auto predicted_reuse =
        live ? read_binary_array<double>(argv[15]) : std::vector<double>{};
    std::string stream = argv[1];
    int64_t block_id_count = std::stoll(argv[2]),
            flash_capacity = std::stoll(argv[3]),
            hbm_capacity = std::stoll(argv[4]);
    int64_t write_budget = std::stoll(argv[7]), split = std::stoll(argv[8]),
            end = std::stoll(argv[9]);
    // argv[10] is a reserved format version, so an old invocation fails closed.
    require_condition(std::string(argv[10]) == "v1", "format version");
    fs::path output_directory = argv[11];
    require_condition(block_id_count > 0 && block_id_count <= INT32_MAX &&
                          flash_capacity >= 0 && flash_capacity <= INT32_MAX &&
                          hbm_capacity >= 0 && hbm_capacity <= INT32_MAX,
                      "geometry");
    require_condition(write_budget >= 0 && split > 0 && end > split,
                      "time or budget");
    auto counts = read_binary_array<int32_t>(stream + "_nb.i32"),
         sessions = read_binary_array<int32_t>(stream + "_sess.i32");
    auto times = read_binary_array<int64_t>(stream + "_ts.i64");
    auto warm_mask = read_binary_array<uint8_t>(argv[5]),
         eval_mask = read_binary_array<uint8_t>(argv[6]);
    size_t request_count = counts.size();
    require_condition(request_count <= INT32_MAX &&
                          sessions.size() == request_count &&
                          times.size() == request_count &&
                          warm_mask.size() == request_count &&
                          eval_mask.size() == request_count,
                      "metadata lengths");

    require_condition(std::is_sorted(times.begin(), times.end()),
                      "timestamp order");
    if (live) {
      require_condition(policy_eligible.size() == request_count &&
                            predicted_reuse.size() == request_count,
                        "policy input length");
      for (size_t request_index = 0; request_index < request_count;
           request_index++)
        require_condition(policy_eligible[request_index] <= 1 &&
                              std::isfinite(predicted_reuse[request_index]) &&
                              predicted_reuse[request_index] >= 0,
                          "policy input value");
    }
    require_condition(!reuse_bonus || (live && !initial_history),
                      "history bonus requires cold live C7");
    std::vector<uint32_t> reuse_count(reuse_bonus ? size_t(block_id_count) : 0,
                                      0);
    std::vector<int32_t> pending_reuse;
    Wallet wallet(write_budget, 0, 1, 1000, 20000, split, end);
    std::vector<History> history(20000);
    std::vector<int32_t> suffix_mark(live ? size_t(block_id_count) : 0, -1);
    std::array<int64_t, 8> reasons{};
    int64_t offered_cost = 0, threshold_cost = 0;
    // Reasons:0 ineligible,1 no suffix,2 free,3 budget,4 reserve,5 threshold,6
    // startup admit,7 threshold admit.

    int64_t total_ids = 0;
    for (size_t request_index = 0; request_index < request_count;
         request_index++) {
      require_condition(
          counts[request_index] >= 0 && sessions[request_index] >= 0 &&
              warm_mask[request_index] <= 1 && eval_mask[request_index] <= 1,
          "invalid metadata");
      total_ids += counts[request_index];
      require_condition(total_ids >= 0, "count overflow");
    }
    require_condition(fs::file_size(stream + "_ids.i32") ==
                          uint64_t(total_ids) * 4,
                      "full block stream length");
    require_condition(!fs::exists(output_directory), "output already exists");
    require_condition(fs::create_directory(output_directory), "create output");
    Cache flash{int32_t(flash_capacity), int32_t(block_id_count)},
        hbm{int32_t(hbm_capacity), int32_t(block_id_count)};
    std::vector<int32_t> stamp(size_t(block_id_count), -1),
        mark(size_t(block_id_count), -1), row;
    std::vector<uint8_t> effective(request_count, 0),
        requested_bits(request_count, 0), sources;
    Metrics warm, eval;
    int64_t blocks_written = 0, blocks_written_before_evaluation = -1,
            block_cursor = 0;
    size_t next = 0;
    bool boundary = false;
    std::ifstream ids(stream + "_ids.i32", std::ios::binary);
    require_condition(bool(ids), "block stream open");
    std::ofstream rows(output_directory / "rows.i64", std::ios::binary);
    require_condition(bool(rows), "rows open");

    std::ofstream decisions;
    if (live) {
      decisions.open(output_directory / "decisions.bin", std::ios::binary);
      require_condition(bool(decisions), "decision log");
    }
    std::ofstream debug;
    if (debug_mode) {
      debug.open(output_directory / "debug.jsonl");
      require_condition(bool(debug), "debug open");
    }
    auto snapshot = [&](const std::string &name, size_t next_row,
                        int64_t id_cursor) {
      auto f = flash.blocks_most_recent_first(),
           h = hbm.blocks_most_recent_first();
      for (int32_t block_id : h)
        require_condition(!flash.contains_block(block_id),
                          "duplicate retained tier copy");
      write_binary_array(output_directory / (name + ".hbf.i32"), f);
      write_binary_array(output_directory / (name + ".hbm.i32"), h);
      std::ofstream meta(output_directory / (name + ".json"));
      meta << "{\"next_row\":" << next_row << ",\"id_cursor\":" << id_cursor
           << ",\"writes\":" << blocks_written << ",\"hbf_used\":" << f.size()
           << ",\"hbm_used\":" << h.size()
           << ",\"history\":\"previous-reference stamps reconstructed from the "
              "immutable raw stream through this cursor\"}\n";
      meta.close();
      require_condition(bool(meta), "snapshot metadata");
    };
    for (size_t request_index = 0; request_index < request_count;
         request_index++) {
      if (times[request_index] >= end) {
        next = request_index;
        break;
      }

      if (!boundary && times[request_index] >= split) {
        snapshot("warm", request_index, block_cursor);
        blocks_written_before_evaluation = blocks_written;
        boundary = true;
        if (live) {
          wallet.W = blocks_written;
          if (initial_history) {
            auto loaded = read_binary_array<History>(argv[17]);
            require_condition(!loaded.empty() &&
                                  loaded.size() <= history.size(),
                              "initial history length");
            int64_t prior = -1, prior_time = -1;
            for (size_t k = 0; k < loaded.size(); k++) {
              auto v = loaded[k];
              require_condition(
                  v.row > prior && v.row < int64_t(request_index) &&
                      v.time >= 0 && v.time < split && v.time >= prior_time,
                  "initial history chronology");
              require_condition(times[v.row] == v.time &&
                                    policy_eligible[v.row] && v.distinct >= 0 &&
                                    v.cost >= 0,
                                "initial history identity or cost");
              require_condition(v.distinct > 0 || v.cost == 0,
                                "empty history suffix has write cost");
              history[k] = v;
              wallet.ring[k] = {0, v.cost, v.time};
              prior = v.row;
              prior_time = v.time;
            }
            wallet.rn = int64_t(loaded.size());
            wallet.rp = wallet.rn % wallet.ring_size;
            wallet.decisions =
                0; // Same measured 1000-observation update phase as cold arm.
          }
          std::ofstream hist(output_directory / "warm_history.bin",
                             std::ios::binary);
          for (int64_t k = 0; k < wallet.rn; k++) {
            const auto &v = history[k];
            require_condition(v.row >= 0 && v.time < split,
                              "warm history chronology");
            wallet.ring[k].score = v.cost > 0
                                       ? predicted_reuse[v.row] *
                                             double(v.distinct) / double(v.cost)
                                       : INFINITY;
            hist.write(reinterpret_cast<const char *>(&v), sizeof(v));
          }
          hist.close();
          require_condition(bool(hist), "warm history write");
          std::ofstream meta(output_directory / "warm_controller.json");
          meta << "{\"rn\":" << wallet.rn << ",\"rp\":" << wallet.rp
               << ",\"decisions\":" << wallet.decisions
               << ",\"W\":" << blocks_written << "}\n";
          wallet.have_threshold = false;
          wallet.threshold = -INFINITY;
          wallet.tie = 1;
          wallet.recalculate_admission_cutoff(split);
        }
      }
      row.resize(size_t(counts[request_index]));
      ids.read(reinterpret_cast<char *>(row.data()),
               std::streamsize(row.size() * 4));
      require_condition(bool(ids), "short row");
      for (int32_t block_id : row)
        require_condition(block_id >= 0 && block_id < block_id_count,
                          "invalid block identity");
      if (live && times[request_index] < split) {
        // History is observed for the metric/predictor, but storage starts
        // empty.
        if (reuse_bonus) {
          for (int32_t block_id : row) {
            if (mark[block_id] == int32_t(request_index))
              continue;
            if (times[request_index] >= 0 &&
                stamp[block_id] == sessions[request_index]) {
              require_condition(reuse_count[block_id] < UINT32_MAX,
                                "reuse overflow");
              ++reuse_count[block_id];
            }
            mark[block_id] = int32_t(request_index);
            stamp[block_id] = sessions[request_index];
          }
        } else
          for (int32_t block_id : row)
            stamp[block_id] = sessions[request_index];
        int64_t record[10] = {int64_t(request_index),
                              times[request_index],
                              int64_t(row.size()),
                              0,
                              0,
                              0,
                              0,
                              0,
                              0,
                              0};
        rows.write(reinterpret_cast<const char *>(record), sizeof(record));
        require_condition(bool(rows), "pre-cut record");
        block_cursor += row.size();
        next = request_index + 1;
        continue;
      }
      sources.assign(row.size(), 0);
      size_t first = 0;
      if (times[request_index] >= 0) {
        for (; first < row.size(); first++) {
          int32_t block_id = row[first];
          if (flash.contains_block(block_id)) {
            sources[first] = 1;
            flash.mark_block_used(block_id);
          } else if (hbm.contains_block(block_id)) {
            sources[first] = 2;
            hbm.mark_block_used(block_id);
          } else
            break;
        }
      }
      int64_t flash_hits = 0, hbm_hits = 0, eligible = 0;
      pending_reuse.clear();
      for (size_t j = 0; j < row.size(); j++) {
        int32_t block_id = row[j];
        if (mark[block_id] == int32_t(request_index))
          continue;
        bool repeat = stamp[block_id] == sessions[request_index];
        eligible += repeat;
        if (reuse_bonus && repeat)
          pending_reuse.push_back(block_id);
        if (repeat && j < first) {
          flash_hits += sources[j] == 1;
          hbm_hits += sources[j] == 2;
        }
        mark[block_id] = int32_t(request_index);
        stamp[block_id] = sessions[request_index];
      }
      bool inwarm = times[request_index] >= 0 && times[request_index] < split,
           requested =
               times[request_index] >= 0 &&
               (inwarm ? warm_mask[request_index] : eval_mask[request_index]);
      bool admit = false, guard = false;
      int64_t actual = 0, cost = 0;
      int64_t distinct = 0, reason = 0, past_reuses = 0;
      double score = INFINITY, used_threshold = -INFINITY, used_tie = 1;
      bool used_have = false;
      bool observe =
          live && times[request_index] >= 0 && policy_eligible[request_index];
      if (observe) {
        require_condition(flash.cap > 0, "live controller requires HBF");
        for (size_t j = first; j < row.size(); j++)
          if (suffix_mark[row[j]] != int32_t(request_index)) {
            suffix_mark[row[j]] = int32_t(request_index);
            distinct++;
            if (reuse_bonus)
              past_reuses += reuse_count[row[j]];
          }
        cost = first < row.size() ? flash.measure_suffix_write_cost(row, first)
                                  : 0;
        if (inwarm) {
          history[wallet.rp] = {int64_t(request_index), distinct, cost,
                                times[request_index]};
          wallet.record_priced_offer(0, cost, times[request_index]);
        } else {
          require_condition(wallet.W == blocks_written,
                            "wallet/cache spending mismatch");
          wallet.release_reserved_budget_if_due(times[request_index]);
          score = cost > 0 ? predicted_reuse[request_index] * double(distinct) /
                                 double(cost)
                           : INFINITY;
          if (reuse_bonus && cost > 0 && distinct > 0)
            score *= 1.0 + double(past_reuses) / double(distinct);
          used_threshold = wallet.threshold;
          used_tie = wallet.tie;
          used_have = wallet.have_threshold;
          if (early_half && wallet.have_threshold &&
              __int128(blocks_written) * __int128(end - split) <
                  __int128(write_budget) *
                      __int128(times[request_index] - split))
            used_threshold /= 2.0;
          requested = false;
          if (distinct == 0)
            reason = 1;
          else if (cost == 0) {
            reason = 2;
            admit = true;
          } else if (cost > write_budget - blocks_written) {
            reason = 3;
            guard = true;
          } else if (cost > wallet.cap - blocks_written)
            reason = 4;
          else if (!wallet.have_threshold) {
            reason = 6;
            admit = true;
          } else if (score > used_threshold ||
                     (score == used_threshold &&
                      random_value_for_request(
                          request_index, seed ^ 0x5bf03635ULL) < wallet.tie)) {
            reason = 7;
            admit = true;
          } else
            reason = 5;
          requested = admit || guard;
          reasons[reason]++;
          offered_cost += cost;
          if (reason == 5)
            threshold_cost += cost;
          // Save the offered suffix even when it is declined so both choices
          // get training labels.
          Decision d{int64_t(request_index),
                     times[request_index],
                     distinct,
                     cost,
                     blocks_written,
                     predicted_reuse[request_index],
                     score,
                     used_threshold,
                     used_tie,
                     int64_t(used_have),
                     reason,
                     int64_t(admit)};
          decisions.write(reinterpret_cast<const char *>(&d), sizeof(d));
          history[wallet.rp] = {int64_t(request_index), distinct, cost,
                                times[request_index]};
          wallet.record_priced_offer(score, cost, times[request_index]);
        }
      } else if (live && !inwarm && times[request_index] >= 0) {
        requested = false;
        reasons[0]++;
      }
      if ((!live || inwarm) && requested && first < row.size() && flash.cap) {
        cost = flash.measure_suffix_write_cost(row, first);
        if (cost > write_budget - blocks_written) {
          require_condition(!inwarm,
                            "saved warm admission cannot be honoured within "
                            "seven-day budget");
          guard = true;
        } else
          admit = true;
      }
      requested_bits[request_index] = requested;
      if (times[request_index] >= 0) {
        for (size_t j = first; j < row.size(); j++) {
          int32_t block_id = row[j];
          if (admit) {
            actual += flash.store_block(block_id);
            hbm.remove_block(block_id);
          }
          if (!flash.contains_block(block_id))
            hbm.store_block(block_id);
        }
      }
      require_condition(actual == (admit ? cost : 0),
                        "dry/commit write mismatch");
      blocks_written += actual;
      require_condition(blocks_written <= write_budget, "hard wallet exceeded");
      if (live)
        wallet.W = blocks_written;
      if (reuse_bonus)
        for (int32_t block_id : pending_reuse) {
          require_condition(reuse_count[block_id] < UINT32_MAX,
                            "reuse overflow");
          ++reuse_count[block_id];
        }
      effective[request_index] = admit;
      Metrics *m =
          times[request_index] < 0 ? nullptr : (inwarm ? &warm : &eval);
      if (m) {
        m->requests++;
        m->positions += row.size();
        m->fetched += first;
        m->hbf_hits += flash_hits;
        m->hbm_hits += hbm_hits;
        m->eligible += eligible;
        m->writes += actual;
        m->requested += requested;
        m->admitted += admit;
        m->budget_rejected += guard;
        m->no_offer += first == row.size();
      }
      int64_t record[10] = {int64_t(request_index),
                            times[request_index],
                            int64_t(row.size()),
                            int64_t(first),
                            flash_hits,
                            hbm_hits,
                            actual,
                            int64_t(requested),
                            int64_t(admit),
                            blocks_written};
      rows.write(reinterpret_cast<char *>(record), sizeof(record));
      require_condition(bool(rows), "row output");
      if (debug.is_open()) {
        debug << "{\"row\":" << request_index << ",\"first\":" << first
              << ",\"hbf_hits\":" << flash_hits << ",\"hbm_hits\":" << hbm_hits
              << ",\"writes\":" << actual << ",\"W\":" << blocks_written
              << ",\"requested\":" << requested << ",\"admitted\":" << admit;
        for (auto item :
             {std::make_pair("hbf", flash.blocks_most_recent_first()),
              std::make_pair("hbm", hbm.blocks_most_recent_first())}) {
          debug << ",\"" << item.first << "\":[";
          for (size_t j = 0; j < item.second.size(); j++)
            debug << (j ? "," : "") << item.second[j];
          debug << "]";
        }
        debug << "}\n";
      }
      block_cursor += row.size();
      next = request_index + 1;
    }
    if (!boundary) {
      snapshot("warm", next, block_cursor);
      blocks_written_before_evaluation = blocks_written;
    }

    snapshot("final", next, block_cursor);
    write_binary_array(output_directory / "effective_mask.u8", effective);
    if (live) {
      std::vector<History> final_history;
      for (int64_t j = 0; j < wallet.rn; j++)
        final_history.push_back(
            history[(wallet.rp + wallet.ring_size - wallet.rn + j) %
                    wallet.ring_size]);
      write_binary_array(output_directory / "final_history.bin", final_history);
      write_binary_array(output_directory / "requested_mask.u8",
                         requested_bits);
      decisions.close();
      require_condition(bool(decisions), "decision log close");
      std::ofstream diag(output_directory / "diagnostic.json");
      diag << std::setprecision(17);
      diag << "{\"reason_counts\":[";
      for (size_t j = 0; j < reasons.size(); j++)
        diag << (j ? "," : "") << reasons[j];
      diag << "],\"offered_write_cost\":" << offered_cost
           << ",\"threshold_rejected_write_cost\":" << threshold_cost
           << ",\"wallet_W\":" << wallet.W
           << ",\"wallet_observations\":" << wallet.decisions
           << ",\"remaining_budget\":" << write_budget - blocks_written
           << "}\n";
    }
    rows.close();
    require_condition(bool(rows), "close rows");
    if (debug.is_open()) {
      debug.close();
      require_condition(bool(debug), "close debug");
    }
    require_condition(blocks_written_before_evaluation == warm.writes &&
                          blocks_written == warm.writes + eval.writes,
                      "warm/eval write identity");
    require_condition(eval.hbf_hits + eval.hbm_hits <= eval.eligible &&
                          eval.eligible <= eval.positions,
                      "credit conservation");
    std::ofstream result(output_directory / "metrics.json");
    result << "{\"schema\":\"hbf_replacement_replay_v1\",\"status\":\"PASS\","
              "\"N\":"
           << block_id_count << ",\"hbf_capacity\":" << flash_capacity
           << ",\"hbm_capacity\":" << hbm_capacity
           << ",\"budget\":" << write_budget << ",\"split_us\":" << split
           << ",\"end_us\":" << end
           << ",\"W_pre\":" << blocks_written_before_evaluation
           << ",\"W_test\":" << eval.writes << ",\"writes\":" << blocks_written
           << ",\"remaining_after_warm\":"
           << write_budget - blocks_written_before_evaluation
           << ",\"next_row\":" << next << ",\"id_cursor\":" << block_cursor
           << ",\"warm\":";
    warm.json(result);
    result << ",\"evaluation\":";
    eval.json(result);
    result << "}\n";
    result.close();
    require_condition(bool(result), "close metrics");
    std::cout << "{\"status\":\"PASS\",\"evaluation_hits\":"
              << eval.hbf_hits + eval.hbm_hits
              << ",\"W_pre\":" << blocks_written_before_evaluation
              << ",\"writes\":" << blocks_written << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
