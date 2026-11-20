// Build SSD training labels from a checked reference write log
// For each sampled eviction count the extra prefix reads that saving it would allow
// Save the gain with the request where it occurred so fitting can use it at the right hour

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <openssl/sha.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>
using namespace std;

static void ck(bool ok, const string &why) {
  if (!ok)
    throw runtime_error(why);
}

template <class T> struct Map {
  int fd = -1;
  size_t n = 0, bytes = 0;
  const T *p = nullptr;

  explicit Map(const string &s) {
    fd = open(s.c_str(), O_RDONLY);
    ck(fd >= 0, "open " + s);
    struct stat st{};
    ck(fstat(fd, &st) == 0, "stat " + s);
    bytes = st.st_size;
    ck(bytes % sizeof(T) == 0, "unaligned " + s);
    n = bytes / sizeof(T);
    if (bytes) {
      void *z = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
      ck(z != MAP_FAILED, "mmap " + s);
      p = (T *)z;
    }
  }

  ~Map() {
    if (bytes)
      munmap((void *)p, bytes);
    if (fd >= 0)
      close(fd);
  }

  const T &operator[](size_t i) const {
    ck(i < n, "mapped index");
    return p[i];
  }
};

struct Hash {
  SHA256_CTX c;

  Hash() { SHA256_Init(&c); }

  void add(int32_t x) { SHA256_Update(&c, &x, 4); }

  string finish() const {
    auto q = c;
    unsigned char b[32];
    SHA256_Final(b, &q);
    ostringstream z;
    for (auto x : b)
      z << hex << setw(2) << setfill('0') << int(x);
    return z.str();
  }
};

struct Eviction {
  int32_t row, block, last, freq, nev, born;
};

static_assert(sizeof(Eviction) == 24);

static uint64_t mix(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

static double random01(int64_t id, uint64_t seed) {
  return double(mix(uint64_t(id) * 0x9e3779b97f4a7c15ULL ^ seed ^ 0x5bf03635ULL) >> 11) /
         9007199254740992.;
}

// Portable order-statistics treap. Allocation is bounded by live history plus
// one temporary observation; deleting old observations recycles their nodes.
class OST {
  using Key = pair<double, int64_t>;

  struct Item {
    Key key;
    uint64_t priority;
    int left = -1, right = -1, size = 1;
  };

  vector<Item> nodes;
  vector<int> free;
  int root = -1;

  int count(int x) const { return x < 0 ? 0 : nodes[x].size; }

  void pull(int x) {
    if (x >= 0)
      nodes[x].size = 1 + count(nodes[x].left) + count(nodes[x].right);
  }

  int rotate_left(int x) {
    int y = nodes[x].right;
    nodes[x].right = nodes[y].left;
    nodes[y].left = x;
    pull(x);
    pull(y);
    return y;
  }

  int rotate_right(int x) {
    int y = nodes[x].left;
    nodes[x].left = nodes[y].right;
    nodes[y].right = x;
    pull(x);
    pull(y);
    return y;
  }

  int put(int x, int y) {
    if (x < 0)
      return y;
    ck(nodes[x].key != nodes[y].key, "duplicate history ID");
    if (nodes[y].key < nodes[x].key) {
      nodes[x].left = put(nodes[x].left, y);
      if (nodes[nodes[x].left].priority > nodes[x].priority)
        x = rotate_right(x);
    } else {
      nodes[x].right = put(nodes[x].right, y);
      if (nodes[nodes[x].right].priority > nodes[x].priority)
        x = rotate_left(x);
    }
    pull(x);
    return x;
  }

  int merge(int l, int r) {
    if (l < 0)
      return r;
    if (r < 0)
      return l;
    if (nodes[l].priority > nodes[r].priority) {
      nodes[l].right = merge(nodes[l].right, r);
      pull(l);
      return l;
    }
    nodes[r].left = merge(l, nodes[r].left);
    pull(r);
    return r;
  }

  int remove(int x, const Key &k) {
    ck(x >= 0, "missing history key");
    if (k == nodes[x].key) {
      int z = merge(nodes[x].left, nodes[x].right);
      free.push_back(x);
      return z;
    }
    if (k < nodes[x].key)
      nodes[x].left = remove(nodes[x].left, k);
    else
      nodes[x].right = remove(nodes[x].right, k);
    pull(x);
    return x;
  }

public:
  void clear() {
    nodes.clear();
    free.clear();
    root = -1;
  }

  void insert(Key k) {
    int x;
    if (free.empty()) {
      x = nodes.size();
      nodes.push_back({});
    } else {
      x = free.back();
      free.pop_back();
    }
    nodes[x] = {k, mix(uint64_t(k.second) ^ 0xd6e8feb86659fd93ULL), -1, -1, 1};
    root = put(root, x);
  }

  void erase(Key k) { root = remove(root, k); }

  size_t order_of_key(Key k) const {
    int x = root;
    size_t n = 0;
    while (x >= 0) {
      if (nodes[x].key < k) {
        n += count(nodes[x].left) + 1;
        x = nodes[x].right;
      } else
        x = nodes[x].left;
    }
    return n;
  }

  const Key *find_by_order(size_t k) const {
    int x = root;
    while (x >= 0) {
      size_t l = count(nodes[x].left);
      if (k == l)
        return &nodes[x].key;
      if (k < l)
        x = nodes[x].left;
      else {
        k -= l + 1;
        x = nodes[x].right;
      }
    }
    throw runtime_error("history order out of bounds");
  }
};

struct Node {
  int32_t b = -1, prev = -1, next = -1;
};

struct Cache {
  vector<int32_t> slot;
  vector<Node> nd;
  int cap, n = 0, head = -1, tail = -1;
  int64_t peak = 0;

  Cache(int nb, int c) : slot(nb, -1), cap(c) { nd.reserve(min(c, 1000000)); }

  bool has(int b) const { return slot[b] >= 0; }

  void touch(int b) {
    int s = slot[b];
    ck(s >= 0, "touch absent");
    if (s == head)
      return;
    auto &z = nd[s];
    if (z.prev >= 0)
      nd[z.prev].next = z.next;
    else
      head = z.next;
    if (z.next >= 0)
      nd[z.next].prev = z.prev;
    else
      tail = z.prev;
    z.prev = -1;
    z.next = head;
    if (head >= 0)
      nd[head].prev = s;
    head = s;
    if (tail < 0)
      tail = s;
  }

  int pop() {
    ck(tail >= 0, "empty eviction");
    int s = tail, b = nd[s].b;
    tail = nd[s].prev;
    if (tail >= 0)
      nd[tail].next = -1;
    else
      head = -1;
    slot[b] = -1;
    n--;
    nd[s].b = -1;
    return s;
  }

  int victim() const {
    ck(tail >= 0, "empty victim");
    return nd[tail].b;
  }

  void add(int b, int recycled = -1) {
    ck(!has(b) && n < cap, "cache insert geometry");
    int s = recycled;
    if (s < 0) {
      s = nd.size();
      nd.push_back({});
    }
    ck(s < int(nd.size()) && nd[s].b < 0, "recycled slot");
    nd[s] = {b, -1, head};
    if (head >= 0)
      nd[head].prev = s;
    head = s;
    if (tail < 0)
      tail = s;
    slot[b] = s;
    n++;
    peak = max(peak, int64_t(n));
  }

  string hash() const {
    Hash h;
    int count = 0;
    for (int s = head; s >= 0; s = nd[s].next) {
      h.add(nd[s].b);
      ck(++count <= n, "cache cycle");
    }
    ck(count == n, "cache length");
    return h.finish();
  }

  string json() const {
    ostringstream s;
    s << '[';
    bool comma = false;
    for (int p = head; p >= 0; p = nd[p].next) {
      if (comma)
        s << ',';
      comma = true;
      s << nd[p].b;
    }
    s << ']';
    return s.str();
  }
};

#include "marginal_labels.hpp"

struct Obs {
  double score;
  int64_t id, time;
};

struct Wallet {
  int64_t budget, end, release, update, limit, W = 0, observations = 0, cap;
  double reserve, demand, threshold = -INFINITY, tie = 1;
  bool have = false, released = false;
  deque<Obs> hist;
  OST order;

  Wallet(int64_t b, int64_t e, int64_t r, int64_t u, int64_t l, double res, double d)
      : budget(b), end(e), release(r), update(u), limit(l),
        cap(int64_t(floor((1 - res) * double(b)))), reserve(res), demand(d) {}

  void phase(int64_t now) {
    if (!released && now >= release) {
      released = true;
      cap = budget;
      have = false;
    }
  }

  void recompute(int64_t now) {
    phase(now);
    if (hist.size() < 100) {
      threshold = -INFINITY;
      tie = 1;
      have = false;
      return;
    }
    double span = max((hist.back().time - hist.front().time) / 1e6, 1.0),
           rate = hist.size() / span,
           horizon = max(((released ? end : release) - now) / 1e6, 1.0);
    double frac =
        clamp(demand * max(double(cap - W), 0.0) / max(rate * horizon, 1e-9), 0.0, 1.0);
    if (frac <= 0) {
      threshold = INFINITY;
      tie = 0;
    } else if (frac >= 1) {
      threshold = -INFINITY;
      tie = 1;
    } else {
      double wanted = frac * hist.size();
      size_t k = min(hist.size() - 1, size_t(floor(hist.size() - wanted)));
      double th = order.find_by_order(k)->first;
      auto lo = order.order_of_key({th, LLONG_MIN}), hi = order.order_of_key({th, LLONG_MAX});
      threshold = th;
      tie = clamp((wanted - (hist.size() - hi)) / double(hi - lo), 0.0, 1.0);
    }
    have = true;
  }

  bool pass(double score, int64_t id, uint64_t seed, int64_t now) {
    phase(now);
    return W < budget && W < cap &&
           (!have || score > threshold || (score == threshold && random01(id, seed) < tie));
  }

  void observe(double score, int64_t id, int64_t now) {
    ck(isfinite(score) && score >= 0, "nonfinite/negative score");
    hist.push_back({score, id, now});
    order.insert({score, id});
    if (hist.size() > size_t(limit)) {
      auto z = hist.front();
      order.erase({z.score, z.id});
      hist.pop_front();
    }
    observations++;
    if (observations % update == 0)
      recompute(now);
  }
};

// The driver emits this restricted JSON shape. Parse numeric arrays exactly;
// do not accept a different ID order or infer scores from missing entries.
static string field(const string &response_json, const string &field_name) {
  string key = '"' + field_name + '"';
  auto field_position = response_json.find(key);
  ck(field_position != string::npos, "missing response " + field_name);
  field_position = response_json.find(':', field_position + key.size());
  ck(field_position != string::npos, "response colon");
  return response_json.substr(field_position + 1);
}

static vector<double> numeric_array(const string &response_json, const string &field_name) {
  auto field_text = field(response_json, field_name);
  auto array_start = field_text.find('[');
  ck(array_start != string::npos, "response array");
  vector<double> values;
  const char *cursor = field_text.c_str() + array_start + 1;
  while (true) {
    while (isspace(*cursor) || *cursor == ',')
      cursor++;
    if (*cursor == ']')
      break;
    char *number_end = nullptr;
    double parsed_value = strtod(cursor, &number_end);
    ck(number_end != cursor && isfinite(parsed_value), "invalid response number");
    values.push_back(parsed_value);
    cursor = number_end;
  }
  return values;
}

static void reprice(Wallet &w, int64_t cut) {
  cout << "{\"event\":\"reprice\",\"cut_us\":" << cut << ",\"ids\":[";
  bool comma = false;
  for (auto &o : w.hist) {
    if (comma)
      cout << ',';
    comma = true;
    cout << o.id;
  }
  cout << "]}" << endl;
  string s;
  ck(bool(getline(cin, s)), "reprice response missing");
  ck(s.find("reprice_scores") != string::npos, "reprice event");
  ck(stoll(field(s, "cut_us")) == cut, "reprice cut mismatch");
  auto ids = numeric_array(s, "ids"), scores = numeric_array(s, "scores");
  ck(ids.size() == w.hist.size() && scores.size() == ids.size(), "reprice length");
  w.order.clear();
  for (size_t j = 0; j < ids.size(); j++) {
    ck(ids[j] == double(w.hist[j].id) && scores[j] >= 0, "reprice IDs/score");
    w.hist[j].score = scores[j];
    w.order.insert({scores[j], w.hist[j].id});
  }
  w.recompute(cut);
}

struct Metrics {
  int64_t requests = 0, prompt_blocks = 0, prompt_tokens = 0, tail_tokens = 0, hbm_prefix = 0,
          prefix = 0, ssd_positions = 0, ssd_payloads = 0, writes = 0, evictions = 0,
          absent_offers = 0, free_retained = 0, eligible_raw = 0, bypass = 0, cap_lost = 0,
          reprograms = 0, ssd_evictions = 0;
};

static string metrics_json(const Metrics &m) {
  ostringstream s;
  s << "{\"requests\":" << m.requests << ",\"prompt_blocks\":" << m.prompt_blocks
    << ",\"prompt_tokens\":" << m.prompt_tokens
    << ",\"unrepresented_tail_tokens\":" << m.tail_tokens
    << ",\"hbm_only_prefix_positions\":" << m.hbm_prefix
    << ",\"combined_prefix_positions\":" << m.prefix
    << ",\"ssd_fetched_prefix_positions\":" << m.ssd_positions
    << ",\"ssd_payload_reads\":" << m.ssd_payloads
    << ",\"incremental_prefix_positions\":" << m.prefix - m.hbm_prefix
    << ",\"physical_writes\":" << m.writes << ",\"hbm_evictions\":" << m.evictions
    << ",\"absent_offers\":" << m.absent_offers << ",\"free_retained\":" << m.free_retained
    << ",\"raw_repeated_hbm_misses_diagnostic\":" << m.eligible_raw
    << ",\"oversized_bypass\":" << m.bypass
    << ",\"last_token_cap_excluded_positions\":" << m.cap_lost
    << ",\"reprograms\":" << m.reprograms << ",\"ssd_evictions\":" << m.ssd_evictions << '}';
  return s.str();
}

int main(int argc, char **argv) {
  try {
    const set<string> allowed = {"--prefix",
                                 "--out-prefix",
                                 "--mode",
                                 "--nb",
                                 "--hbm",
                                 "--ssd",
                                 "--budget",
                                 "--eval-start-us",
                                 "--end-us",
                                 "--stop-us",
                                 "--seed",
                                 "--update-every",
                                 "--history-size",
                                 "--release-us",
                                 "--reserve",
                                 "--demand-multiplier",
                                 "--label-samples",
                                 "--label-credit",
                                 "--label-event-offset",
                                 "--coin-probability",
                                 "--tokens",
                                 "--scores",
                                 "--warm-program-bits",
                                 "--fixed-program-bits",
                                 "--expected-evictions",
                                 "--cuts-i64",
                                 "--evictions-out",
                                 "--events-out",
                                 "--debug-jsonl"};
    map<string, string> options;
    set<string> flags;
    for (int j = 1; j < argc; j++) {
      string option_name = argv[j];
      ck(option_name.rfind("--", 0) == 0, "expected option");
      if (option_name == "--reprice-stdio") {
        ck(!flags.count(option_name), "duplicate flag");
        flags.insert(option_name);
      } else {
        ck(allowed.count(option_name), "unknown option " + option_name);
        ck(j + 1 < argc, "option value");
        ck(!options.count(option_name), "duplicate option");
        options[option_name] = argv[++j];
      }
    }
    auto required_option = [&](string option_name) {
      ck(options.count(option_name), "missing " + option_name);
      return options[option_name];
    };
    auto option_or_default = [&](string option_name, string default_value) {
      return options.count(option_name) ? options[option_name] : default_value;
    };
    string stream_prefix = required_option("--prefix"),
           output_prefix = required_option("--out-prefix"),
           mode = option_or_default("--mode", "quantile");
    int block_id_count = stoi(required_option("--nb")),
        hbm_capacity = stoi(required_option("--hbm"));
    int64_t ssd_capacity = stoll(required_option("--ssd")),
            write_budget = stoll(required_option("--budget")),
            split = stoll(required_option("--eval-start-us")),
            end = stoll(option_or_default("--end-us", "604800000000")),
            stop = stoll(option_or_default("--stop-us", to_string(end)));
    uint64_t seed = stoull(option_or_default("--seed", "17"));
    int64_t update = stoll(option_or_default("--update-every", "1000")),
            history = stoll(option_or_default("--history-size", "20000")),
            release = stoll(option_or_default("--release-us", to_string(end)));
    double reserve = stod(option_or_default("--reserve", "0")),
           demand = stod(option_or_default("--demand-multiplier", "1")),
           coin = stod(option_or_default("--coin-probability", "0.5"));
    ck(block_id_count > 0 && hbm_capacity > 0 && ssd_capacity > 0 && write_budget >= 0 &&
           split >= 0 && end > split && stop <= end && stop >= 0,
       "geometry");
    ck(update > 0 && history >= 100 && release > 0 && release <= end && reserve >= 0 &&
           reserve < 1 && isfinite(demand) && demand > 0 && coin >= 0 && coin <= 1,
       "controller configuration");
    ck(mode == "quantile" || mode == "all" || mode == "coin" || mode == "bits", "policy mode");
    Map<int32_t> block_counts(stream_prefix + "_nb.i32"), block_ids(stream_prefix + "_ids.i32");
    Map<int64_t> arrival_times(stream_prefix + "_ts.i64"), tokens(required_option("--tokens"));
    ck(block_counts.n == arrival_times.n && block_counts.n == tokens.n &&
           block_counts.n <= INT_MAX,
       "request shapes");
    unique_ptr<Map<double>> scores;
    if (options.count("--scores"))
      scores = make_unique<Map<double>>(options["--scores"]);
    ck(mode != "quantile" || scores, "quantile scores required");
    unique_ptr<Map<uint8_t>> warmbits, fixedbits;
    if (options.count("--warm-program-bits"))
      warmbits = make_unique<Map<uint8_t>>(options["--warm-program-bits"]);
    if (options.count("--fixed-program-bits"))
      fixedbits = make_unique<Map<uint8_t>>(options["--fixed-program-bits"]);
    ck(mode != "bits" || fixedbits, "fixed bits required");
    unique_ptr<Map<Eviction>> expected;
    if (options.count("--expected-evictions"))
      expected = make_unique<Map<Eviction>>(options["--expected-evictions"]);
    unique_ptr<Map<int64_t>> cuts;
    if (options.count("--cuts-i64")) {
      cuts = make_unique<Map<int64_t>>(options["--cuts-i64"]);
      ck(cuts->n > 0 && (*cuts)[0] == split, "initial candidate calibration cut required");
      for (size_t j = 0; j < cuts->n; j++)
        ck((*cuts)[j] >= split && (*cuts)[j] < end && (j == 0 || (*cuts)[j] > (*cuts)[j - 1]),
           "invalid fit cuts");
    }
    ck(!flags.count("--reprice-stdio") || cuts, "repricing cuts required");
    ck(!cuts || flags.count("--reprice-stdio"), "cuts without repricing");
    ck(!cuts || mode == "quantile", "only quantile reprices");
    ck(access((output_prefix + ".rows.i64").c_str(), F_OK) != 0 &&
           access((output_prefix + ".programs.bits").c_str(), F_OK) != 0,
       "output already exists");
    for (string option_name : {"--evictions-out", "--events-out", "--debug-jsonl"})
      if (options.count(option_name))
        ck(access(options[option_name].c_str(), F_OK) != 0,
           "output already exists " + option_name);
    Cache hbm_cache(block_id_count, min(block_id_count, hbm_capacity)),
        ssd_cache(block_id_count, int(min<int64_t>(block_id_count, ssd_capacity)));
    vector<int32_t> mark(block_id_count, -1), fetched(block_id_count, -1),
        last(block_id_count, -1), freq(block_id_count), nev(block_id_count),
        born(block_id_count, -1);
    vector<uint8_t> ever(block_id_count), programs;
    Wallet wallet(write_budget, end, release, update, history, reserve, demand);
    ofstream rows(output_prefix + ".rows.i64", ios::binary), evout, eventsout, debug;
    ck(bool(rows), "rows output");
    if (options.count("--evictions-out")) {
      evout.open(options["--evictions-out"], ios::binary);
      ck(bool(evout), "eviction output");
    }
    if (options.count("--events-out")) {
      eventsout.open(options["--events-out"], ios::binary);
      ck(bool(eventsout), "event output");
    }
    if (options.count("--debug-jsonl")) {
      debug.open(options["--debug-jsonl"]);
      ck(bool(debug), "debug output");
    }
    Marginal marginal(block_id_count, required_option("--label-samples"),
                      required_option("--label-credit"), output_prefix,
                      stoll(option_or_default("--label-event-offset", "0")));
    Hash evhash, eventhash;
    Metrics total, eval;
    int64_t eviction_count = 0, writes_before_evaluation = 0;
    size_t block_offset = 0, next_model_cutoff = 0;
    bool snapped = false, initialized = false;
    string warmh, warms;
    int64_t warmE = 0;
    auto emit_event = [&](int32_t b, int32_t request_index) {
      eventhash.add(b);
      eventhash.add(request_index);
      if (eventsout) {
        int32_t z[2] = {b, request_index};
        eventsout.write((char *)z, sizeof z);
      }
    };
    auto bit = [&](const Map<uint8_t> &m, int64_t e) {
      ck(size_t(e / 8) < m.n, "decision bit coverage");
      return bool((m[e / 8] >> (e % 8)) & 1);
    };
    for (int request_index = 0; request_index < int(block_counts.n); request_index++) {
      int64_t now = arrival_times[request_index];
      ck(request_index == 0 || now >= arrival_times[request_index - 1],
         "nonmonotonic timestamps");
      ck(block_counts[request_index] >= 0 &&
             block_offset + size_t(block_counts[request_index]) <= block_ids.n &&
             tokens[request_index] >= 0 &&
             tokens[request_index] / 16 == block_counts[request_index],
         "token/hash mapping");
      if (now < split) {
        block_offset += block_counts[request_index];
        continue;
      }
      if (now >= split && !snapped) {
        ck(hbm_cache.n == 0 && ssd_cache.n == 0 && wallet.W == 0, "cold day4");
        writes_before_evaluation = wallet.W;
        warmE = eviction_count;
        warmh = hbm_cache.hash();
        warms = ssd_cache.hash();
        snapped = true;
      }
      if (now >= stop)
        break;
      if (now >= split && !initialized) {
        wallet.recompute(split);
        initialized = true;
      }
      while (cuts && next_model_cutoff < cuts->n && (*cuts)[next_model_cutoff] <= now) {
        reprice(wallet, (*cuts)[next_model_cutoff]);
        next_model_cutoff++;
      }
      Metrics row;
      row.requests = 1;
      row.prompt_blocks = block_counts[request_index];
      row.prompt_tokens = tokens[request_index];
      row.tail_tokens = tokens[request_index] % 16;
      int64_t cap = tokens[request_index] > 0 ? (tokens[request_index] - 1) / 16 : 0;
      row.cap_lost = block_counts[request_index] - cap;
      ck(cap >= 0 && cap <= block_counts[request_index] && row.cap_lost <= 1, "prefix cap");
      vector<int> ds;
      ds.reserve(block_counts[request_index]);
      for (int j = 0; j < block_counts[request_index]; j++) {
        int b = block_ids[block_offset + j];
        ck(b >= 0 && b < block_id_count, "block ID");
        if (mark[b] != request_index) {
          mark[b] = request_index;
          ds.push_back(b);
        }
      }
      bool bypass = ds.size() > size_t(hbm_capacity);
      row.bypass = bypass;
      vector<int> rowoffers, rowfetches;
      int64_t beforeW = wallet.W;
      if (!bypass) {
        for (int j = 0; j < cap; j++) {
          if (!hbm_cache.has(block_ids[block_offset + j]))
            break;
          row.hbm_prefix++;
        }
        // Entire prefix is selected and staged before any eviction callback.
        for (int j = 0; j < cap; j++) {
          int b = block_ids[block_offset + j];
          if (hbm_cache.has(b)) {
            row.prefix++;
            continue;
          }
          if (!ssd_cache.has(b))
            break;
          ck(last[b] >= 0, "SSD contains cold hash");
          row.prefix++;
          row.ssd_positions++;
          if (fetched[b] != request_index) {
            fetched[b] = request_index;
            ssd_cache.touch(b);
            row.ssd_payloads++;
            rowfetches.push_back(b);
          }
        }
        marginal.measure(hbm_cache, ssd_cache, block_ids, block_offset, cap, row.prefix,
                         request_index);
        int missing = 0;
        for (int b : ds) {
          if (hbm_cache.has(b))
            hbm_cache.touch(b);
          else
            missing++;
        }
        vector<int> free_slots;
        while (hbm_capacity - hbm_cache.n < missing) {
          int b = hbm_cache.victim();
          ck(last[b] >= 0 && born[b] >= 0, "victim history");
          Eviction ev{request_index, b, last[b], freq[b], nev[b], born[b]};
          if (expected) {
            ck(size_t(eviction_count) < expected->n &&
                   !memcmp(&ev, &(*expected)[eviction_count], sizeof ev),
               "HBM eviction identity mismatch at " + to_string(eviction_count));
          }
          if (evout)
            evout.write((char *)&ev, sizeof ev);
          for (int32_t x : {ev.row, ev.block, ev.last, ev.freq, ev.nev, ev.born})
            evhash.add(x);
          emit_event(b, request_index);
          nev[b]++;
          row.evictions++;
          rowoffers.push_back(b);
          int64_t eid = eviction_count++;
          if (programs.size() < size_t((eviction_count + 7) / 8))
            programs.push_back(0);
          marginal.offer(b, eid + stoll(option_or_default("--label-event-offset", "0")));
          bool forced = mode == "bits" || (warmbits && now < split);
          bool desired =
              forced ? bit(warmbits && now < split ? *warmbits : *fixedbits, eid) : false;
          if (ssd_cache.has(b)) {
            ck(!desired, "fixed program on resident copy");
            ssd_cache.touch(b);
            row.free_retained++;
          } else if (now < 0) {
            ck(!desired, "negative-time program");
          } else {
            row.absent_offers++;
            bool admit = false;
            double score = 0;
            if (mode == "quantile") {
              ck(scores && size_t(eid) < scores->n, "score coverage");
              score = (*scores)[eid];
              ck(isfinite(score) && score >= 0, "score value");
            }
            if (forced) {
              admit = desired;
              ck(!admit || wallet.W < write_budget, "forced program exceeds budget");
            } else if (mode == "all")
              admit = wallet.W < write_budget;
            else if (mode == "coin")
              admit = wallet.W < write_budget && random01(eid, seed) < coin;
            else
              admit = wallet.pass(score, eid, seed, now);
            if (admit) {
              ck(wallet.W < write_budget, "write budget");
              wallet.W++;
              row.reprograms += ever[b];
              ever[b] = 1;
              programs[eid / 8] |= uint8_t(1u << (eid % 8));
              int recycle = -1;
              if (ssd_cache.n == ssd_cache.cap) {
                recycle = ssd_cache.pop();
                row.ssd_evictions++;
              }
              ssd_cache.add(b, recycle);
            }
            // Observation follows this decision; the current score never sets its own
            // cutoff.
            if (mode == "quantile")
              wallet.observe(score, eid, now);
          }
          free_slots.push_back(hbm_cache.pop());
        }
        for (int b : ds)
          if (!hbm_cache.has(b)) {
            if (last[b] >= 0) {
              emit_event(~b, request_index);
              row.eligible_raw++;
            }
            int recycle = -1;
            if (!free_slots.empty()) {
              recycle = free_slots.back();
              free_slots.pop_back();
            }
            hbm_cache.add(b, recycle);
            born[b] = request_index;
          }
        ck(free_slots.empty(), "unused HBM slots");
        for (int b : ds) {
          last[b] = request_index;
          ck(freq[b] < INT_MAX, "frequency overflow");
          freq[b]++;
        }
      }
      row.writes = wallet.W - beforeW;
      ck(row.prefix >= row.hbm_prefix && row.ssd_positions <= row.prefix - row.hbm_prefix,
         "prefix metric invariant");
      auto add = [&](Metrics &m) {
        m.requests += row.requests;
        m.prompt_blocks += row.prompt_blocks;
        m.prompt_tokens += row.prompt_tokens;
        m.tail_tokens += row.tail_tokens;
        m.hbm_prefix += row.hbm_prefix;
        m.prefix += row.prefix;
        m.ssd_positions += row.ssd_positions;
        m.ssd_payloads += row.ssd_payloads;
        m.writes += row.writes;
        m.evictions += row.evictions;
        m.absent_offers += row.absent_offers;
        m.free_retained += row.free_retained;
        m.eligible_raw += row.eligible_raw;
        m.bypass += row.bypass;
        m.cap_lost += row.cap_lost;
        m.reprograms += row.reprograms;
        m.ssd_evictions += row.ssd_evictions;
      };
      if (now >= 0)
        add(total);
      if (now >= split) {
        add(eval);
        int64_t z[13] = {request_index,        now,          block_counts[request_index],
                         row.hbm_prefix,       row.prefix,   row.ssd_positions,
                         row.ssd_payloads,     row.writes,   row.evictions,
                         row.bypass,           row.cap_lost, row.eligible_raw,
                         tokens[request_index]};
        rows.write((char *)z, sizeof z);
      }
      if (debug) {
        debug << "{\"row\":" << request_index << ",\"now\":" << now
              << ",\"metrics\":" << metrics_json(row) << ",\"writes_total\":" << wallet.W
              << ",\"hbm\":" << hbm_cache.json() << ",\"ssd\":" << ssd_cache.json()
              << ",\"offers\":[";
        for (size_t j = 0; j < rowoffers.size(); j++) {
          if (j)
            debug << ',';
          debug << rowoffers[j];
        }
        debug << "],\"fetched\":[";
        for (size_t j = 0; j < rowfetches.size(); j++) {
          if (j)
            debug << ',';
          debug << rowfetches[j];
        }
        debug << "]}\n";
      }
      block_offset += block_counts[request_index];
    }
    if (!snapped) {
      writes_before_evaluation = wallet.W;
      warmE = eviction_count;
      warmh = hbm_cache.hash();
      warms = ssd_cache.hash();
    }
    ck(wallet.W <= write_budget && eval.writes == wallet.W - writes_before_evaluation,
       "final wallet identity");
    rows.close();
    ck(bool(rows), "row output flush");
    if (evout) {
      evout.close();
      ck(bool(evout), "eviction flush");
    }
    if (eventsout) {
      eventsout.close();
      ck(bool(eventsout), "events flush");
    }
    ofstream bitsout(output_prefix + ".programs.bits", ios::binary);
    bitsout.write((char *)programs.data(), programs.size());
    bitsout.close();
    ck(bool(bitsout), "program bit output");
    cout << "{\"event\":\"result\",\"status\":\"PASS\",\"schema\":\"ssd_joint_prefix_"
            "atomic_v1\",\"physical_scope\":\"request_atomic_hash_deduplicated_HBM_not_"
            "vLLM_slots\",\"primary_metric\":\"ssd_fetched_prefix_positions\","
            "\"session_filter\":false,\"budget\":"
         << write_budget << ",\"writes\":" << wallet.W
         << ",\"W_pre\":" << writes_before_evaluation
         << ",\"W_test\":" << wallet.W - writes_before_evaluation
         << ",\"processed_evictions\":" << eviction_count << ",\"warm_evictions\":" << warmE
         << ",\"controller_observations\":" << wallet.observations
         << ",\"repricing_snapshots\":" << next_model_cutoff
         << ",\"peak_hbm\":" << hbm_cache.peak << ",\"peak_ssd\":" << ssd_cache.peak
         << ",\"warm_hbm_sha256\":\"" << warmh << "\",\"warm_ssd_sha256\":\"" << warms
         << "\",\"final_hbm_sha256\":\"" << hbm_cache.hash() << "\",\"final_ssd_sha256\":\""
         << ssd_cache.hash() << "\",\"eviction_records_sha256\":\"" << evhash.finish()
         << "\",\"hbm_signed_event_sha256\":\"" << eventhash.finish()
         << "\",\"total\":" << metrics_json(total) << ",\"evaluation\":" << metrics_json(eval)
         << "}" << endl;
    marginal.events.close();
    ck(bool(marginal.events) && marginal.tests > 0, "label verification");
    cerr << "explicit toggles verified " << marginal.tests << endl;
    return 0;
  } catch (const exception &e) {
    cerr << "ssd_prefix_replay: " << e.what() << '\n';
    return 2;
  }
}
