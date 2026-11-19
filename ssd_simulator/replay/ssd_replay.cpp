// Replay requests through GPU-memory and SSD caches; count reads and writes.
//
// Read in this order:
// 1. MappedArray reads the binary input arrays. BlockDigest checks cache histories.
// 2. ScoreHistory keeps recent scores in sorted order for cutoff selection.
// 3. LruCache stores block IDs and removes the least recently used block when full.
// 4. WriteBudgetController chooses which scored writes the allowance can afford.
// 5. ReplayMetrics holds counters; refresh_scores accepts causal model updates.
// 6. main() reads each request, finds its cached beginning, offers GPU evictions
//    to SSD, updates the caches, then saves counters and a write-decision log.
//
// Assumptions: requests execute one at a time, with full input-block identities.
// This models cache behavior, not vLLM scheduling, transfer latency or GPU time.
// The caller supplies reuse predictions and controls what was available at a fit.
// Raw prefix counts have no session filter; the paper's within-session measure
// requires its separate eligibility accounting. See README.md for the distinction.
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
static void require_condition(bool ok, const string &why) {
  if (!ok)
    throw runtime_error(why);
}
// Read a binary array without loading the entire file into RAM.
template <class T> struct MappedArray {
  int fd = -1;
  size_t n = 0, bytes = 0;
  const T *p = nullptr;
  explicit MappedArray(const string &s) {
    fd = open(s.c_str(), O_RDONLY);
    require_condition(fd >= 0, "open " + s);
    struct stat st{};
    require_condition(fstat(fd, &st) == 0, "stat " + s);
    bytes = st.st_size;
    require_condition(bytes % sizeof(T) == 0, "unaligned " + s);
    n = bytes / sizeof(T);
    if (bytes) {
      void *z = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
      require_condition(z != MAP_FAILED, "mmap " + s);
      p = (T *)z;
    }
  }
  ~MappedArray() {
    if (bytes)
      munmap((void *)p, bytes);
    if (fd >= 0)
      close(fd);
  }
  const T &operator[](size_t i) const {
    require_condition(i < n, "mapped index");
    return p[i];
  }
};
// Hash the cache order so an independent implementation can check it.
struct BlockDigest {
  SHA256_CTX c;
  BlockDigest() { SHA256_Init(&c); }
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
// Describe one block leaving GPU memory; the driver uses this as a model example.
struct EvictionRecord {
  int32_t row, block, last, freq, nev, born;
};
static_assert(sizeof(EvictionRecord) == 24);
static uint64_t mix_hash(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}
static double random_value(int64_t id, uint64_t seed) {
  return double(mix_hash(uint64_t(id) * 0x9e3779b97f4a7c15ULL ^ seed ^ 0x5bf03635ULL) >>
                11) /
         9007199254740992.;
}
// Portable order-statistics treap. Allocation is bounded by live history plus
// one temporary observation; deleting old observations recycles their nodes.
// Keep recent scores sorted so cutoff updates can find the desired rank.
class ScoreHistory {
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
    require_condition(nodes[x].key != nodes[y].key, "duplicate history ID");
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
    require_condition(x >= 0, "missing history key");
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
    nodes[x] = {k, mix_hash(uint64_t(k.second) ^ 0xd6e8feb86659fd93ULL), -1, -1, 1};
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
// Link a stored block to its neighbors in most-recently-used order.
struct CacheSlot {
  int32_t b = -1, prev = -1, next = -1;
};
// Track block IDs only. head is newest; tail is the next block to remove.
struct LruCache {
  vector<int32_t> slot;
  vector<CacheSlot> nd;
  int cap, n = 0, head = -1, tail = -1;
  int64_t peak = 0;
  LruCache(int nb, int c) : slot(nb, -1), cap(c) { nd.reserve(min(c, 1000000)); }
  bool has(int b) const { return slot[b] >= 0; }
  void touch(int b) {
    int s = slot[b];
    require_condition(s >= 0, "touch absent");
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
    require_condition(tail >= 0, "empty eviction");
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
    require_condition(tail >= 0, "empty victim");
    return nd[tail].b;
  }
  void add(int b, int recycled = -1) {
    require_condition(!has(b) && n < cap, "cache insert geometry");
    int s = recycled;
    if (s < 0) {
      s = nd.size();
      nd.push_back({});
    }
    require_condition(s < int(nd.size()) && nd[s].b < 0, "recycled slot");
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
    BlockDigest h;
    int count = 0;
    for (int s = head; s >= 0; s = nd[s].next) {
      h.add(nd[s].b);
      require_condition(++count <= n, "cache cycle");
    }
    require_condition(count == n, "cache length");
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
// Store one earlier score, its eviction ID and its arrival time.
struct OfferObservation {
  double score;
  int64_t id, time;
};
// Forecast how many one-block writes remain, then choose an affordable cutoff.
// Observe declined offers too; new observations affect later decisions.
struct WriteBudgetController {
  int64_t budget, end, release, update, limit, written_blocks = 0, observations = 0,
                                               cap;
  double reserve, demand, threshold = -INFINITY, tie = 1;
  bool have = false, released = false;
  deque<OfferObservation> hist;
  ScoreHistory order;
  WriteBudgetController(int64_t b, int64_t e, int64_t r, int64_t u, int64_t l,
                        double res, double d)
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
    double frac = clamp(demand * max(double(cap - written_blocks), 0.0) /
                            max(rate * horizon, 1e-9),
                        0.0, 1.0);
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
      auto lo = order.order_of_key({th, LLONG_MIN}),
           hi = order.order_of_key({th, LLONG_MAX});
      threshold = th;
      tie = clamp((wanted - (hist.size() - hi)) / double(hi - lo), 0.0, 1.0);
    }
    have = true;
  }
  bool pass(double score, int64_t id, uint64_t seed, int64_t now) {
    phase(now);
    return written_blocks < budget && written_blocks < cap &&
           (!have || score > threshold ||
            (score == threshold && random_value(id, seed) < tie));
  }
  void observe(double score, int64_t id, int64_t now) {
    require_condition(isfinite(score) && score >= 0, "nonfinite/negative score");
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
static string field(const string &s, const string &k) {
  string key = '"' + k + '"';
  auto p = s.find(key);
  require_condition(p != string::npos, "missing response " + k);
  p = s.find(':', p + key.size());
  require_condition(p != string::npos, "response colon");
  return s.substr(p + 1);
}
static vector<double> numeric_array(const string &s, const string &k) {
  auto t = field(s, k);
  auto p = t.find('[');
  require_condition(p != string::npos, "response array");
  vector<double> a;
  const char *x = t.c_str() + p + 1;
  while (true) {
    while (isspace(*x) || *x == ',')
      x++;
    if (*x == ']')
      break;
    char *z = nullptr;
    double y = strtod(x, &z);
    require_condition(z != x && isfinite(y), "invalid response number");
    a.push_back(y);
    x = z;
  }
  return a;
}
// Ask the Python driver to rescore retained history using its current model.
// IDs must stay in order, and the driver must use only observations before cut.
static void refresh_scores(WriteBudgetController &w, int64_t cut) {
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
  require_condition(bool(getline(cin, s)), "reprice response missing");
  require_condition(s.find("reprice_scores") != string::npos, "reprice event");
  require_condition(stoll(field(s, "cut_us")) == cut, "reprice cut mismatch");
  auto ids = numeric_array(s, "ids"), scores = numeric_array(s, "scores");
  require_condition(ids.size() == w.hist.size() && scores.size() == ids.size(),
                    "reprice length");
  w.order.clear();
  for (size_t j = 0; j < ids.size(); j++) {
    require_condition(ids[j] == double(w.hist[j].id) && scores[j] >= 0,
                      "reprice IDs/score");
    w.hist[j].score = scores[j];
    w.order.insert({scores[j], w.hist[j].id});
  }
  w.recompute(cut);
}
// Hold counts for one request or for the whole replay.
struct ReplayMetrics {
  int64_t requests = 0, prompt_blocks = 0, prompt_tokens = 0, tail_tokens = 0,
          hbm_prefix = 0, prefix = 0, ssd_positions = 0, ssd_payloads = 0, writes = 0,
          evictions = 0, absent_offers = 0, free_retained = 0, eligible_raw = 0,
          bypass = 0, cap_lost = 0, reprograms = 0, ssd_evictions = 0;
};
static string metrics_json(const ReplayMetrics &m) {
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
    << ",\"absent_offers\":" << m.absent_offers
    << ",\"free_retained\":" << m.free_retained
    << ",\"raw_repeated_hbm_misses_diagnostic\":" << m.eligible_raw
    << ",\"oversized_bypass\":" << m.bypass
    << ",\"last_token_cap_excluded_positions\":" << m.cap_lost
    << ",\"reprograms\":" << m.reprograms << ",\"ssd_evictions\":" << m.ssd_evictions
    << '}';
  return s.str();
}
// Read settings and arrays, process requests in time order, then save the replay.
int main(int argc, char **argv) {
  try {
    const set<string> allowed = {"--prefix",
                                 "--out-prefix",
                                 "--mode",
                                 "--threshold",
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
                                 "--coin-probability",
                                 "--probability-numerator",
                                 "--probability-denominator",
                                 "--tokens",
                                 "--scores",
                                 "--warm-program-bits",
                                 "--fixed-program-bits",
                                 "--expected-evictions",
                                 "--cuts-i64",
                                 "--evictions-out",
                                 "--events-out",
                                 "--debug-jsonl"};
    map<string, string> a;
    set<string> flags;
    for (int j = 1; j < argc; j++) {
      string k = argv[j];
      require_condition(k.rfind("--", 0) == 0, "expected option");
      if (k == "--reprice-stdio") {
        require_condition(!flags.count(k), "duplicate flag");
        flags.insert(k);
      } else {
        require_condition(allowed.count(k), "unknown option " + k);
        require_condition(j + 1 < argc, "option value");
        require_condition(!a.count(k), "duplicate option");
        a[k] = argv[++j];
      }
    }
    auto req = [&](string k) {
      require_condition(a.count(k), "missing " + k);
      return a[k];
    };
    auto val = [&](string k, string d) { return a.count(k) ? a[k] : d; };
    string p = req("--prefix"), out = req("--out-prefix"),
           mode = val("--mode", "quantile");
    int block_id_count = stoi(req("--nb")), hbm_capacity = stoi(req("--hbm"));
    int64_t ssd_capacity = stoll(req("--ssd")), write_budget = stoll(req("--budget")),
            split = stoll(req("--eval-start-us")),
            end = stoll(val("--end-us", "604800000000")),
            stop = stoll(val("--stop-us", to_string(end)));
    uint64_t seed = stoull(val("--seed", "17"));
    int64_t update = stoll(val("--update-every", "1000")),
            history = stoll(val("--history-size", "20000")),
            release = stoll(val("--release-us", to_string(end)));
    double reserve = stod(val("--reserve", "0")),
           demand = stod(val("--demand-multiplier", "1")),
           coin = stod(val("--coin-probability", "0.5"));
    int threshold = stoi(val("--threshold", "0"));
    bool gate_mode = mode == "stride" || mode == "access_min" ||
                     mode == "eviction_min" || mode == "eviction_exact";
    if (gate_mode)
      require_condition(threshold > 0, "threshold required");
    int64_t numerator = stoll(val("--probability-numerator", "-1")),
            denominator = stoll(val("--probability-denominator", "-1"));
    bool exact =
        a.count("--probability-numerator") || a.count("--probability-denominator");
    if (exact)
      require_condition(mode == "coin" && numerator >= 0 && denominator > 0 &&
                            numerator <= denominator,
                        "invalid rational probability");
    auto pick = [&](int64_t eid) {
      uint64_t draw =
          mix_hash(uint64_t(eid) * 0x9e3779b97f4a7c15ULL ^ seed ^ 0x5bf03635ULL) >> 11;
      return exact ? (__int128(draw) * denominator <
                      __int128(numerator) * (uint64_t(1) << 53))
                   : random_value(eid, seed) < coin;
    };
    require_condition(block_id_count > 0 && hbm_capacity > 0 && ssd_capacity > 0 &&
                          write_budget >= 0 && split >= 0 && end > split &&
                          stop <= end && stop >= 0,
                      "geometry");
    require_condition(update > 0 && history >= 100 && release > 0 && release <= end &&
                          reserve >= 0 && reserve < 1 && isfinite(demand) &&
                          demand > 0 && coin >= 0 && coin <= 1,
                      "controller configuration");
    require_condition(mode == "quantile" || mode == "all" || mode == "coin" ||
                          mode == "bits" || gate_mode,
                      "policy mode");
    MappedArray<int32_t> block_counts(p + "_nb.i32"), block_ids(p + "_ids.i32");
    MappedArray<int64_t> arrival_times(p + "_ts.i64"), token_counts(req("--tokens"));
    require_condition(block_counts.n == arrival_times.n &&
                          block_counts.n == token_counts.n && block_counts.n <= INT_MAX,
                      "request shapes");
    unique_ptr<MappedArray<double>> scores;
    if (a.count("--scores"))
      scores = make_unique<MappedArray<double>>(a["--scores"]);
    require_condition(mode != "quantile" || scores, "quantile scores required");
    unique_ptr<MappedArray<uint8_t>> warmbits, fixedbits;
    if (a.count("--warm-program-bits"))
      warmbits = make_unique<MappedArray<uint8_t>>(a["--warm-program-bits"]);
    if (a.count("--fixed-program-bits"))
      fixedbits = make_unique<MappedArray<uint8_t>>(a["--fixed-program-bits"]);
    require_condition(mode != "bits" || fixedbits, "fixed bits required");
    unique_ptr<MappedArray<EvictionRecord>> expected;
    if (a.count("--expected-evictions"))
      expected = make_unique<MappedArray<EvictionRecord>>(a["--expected-evictions"]);
    unique_ptr<MappedArray<int64_t>> cuts;
    if (a.count("--cuts-i64")) {
      cuts = make_unique<MappedArray<int64_t>>(a["--cuts-i64"]);
      require_condition(cuts->n > 0 && (*cuts)[0] == split,
                        "initial candidate calibration cut required");
      for (size_t j = 0; j < cuts->n; j++)
        require_condition((*cuts)[j] >= split && (*cuts)[j] < end &&
                              (j == 0 || (*cuts)[j] > (*cuts)[j - 1]),
                          "invalid fit cuts");
    }
    require_condition(!flags.count("--reprice-stdio") || cuts,
                      "repricing cuts required");
    require_condition(!cuts || flags.count("--reprice-stdio"),
                      "cuts without repricing");
    require_condition(!cuts || mode == "quantile", "only quantile reprices");
    require_condition(access((out + ".rows.i64").c_str(), F_OK) != 0 &&
                          access((out + ".programs.bits").c_str(), F_OK) != 0,
                      "output already exists");
    for (string k : {"--evictions-out", "--events-out", "--debug-jsonl"})
      if (a.count(k))
        require_condition(access(a[k].c_str(), F_OK) != 0,
                          "output already exists " + k);
    LruCache hbm_cache(block_id_count, min(block_id_count, hbm_capacity)),
        ssd_cache(block_id_count, int(min<int64_t>(block_id_count, ssd_capacity)));
    vector<int32_t> mark(block_id_count, -1), fetched(block_id_count, -1),
        last(block_id_count, -1), freq(block_id_count), nev(block_id_count),
        born(block_id_count, -1);
    vector<uint8_t> ever(block_id_count), programs;
    WriteBudgetController wallet(write_budget, end, release, update, history, reserve,
                                 demand);
    ofstream rows(out + ".rows.i64", ios::binary), evout, eventsout, debug;
    require_condition(bool(rows), "rows output");
    if (a.count("--evictions-out")) {
      evout.open(a["--evictions-out"], ios::binary);
      require_condition(bool(evout), "eviction output");
    }
    if (a.count("--events-out")) {
      eventsout.open(a["--events-out"], ios::binary);
      require_condition(bool(eventsout), "event output");
    }
    if (a.count("--debug-jsonl")) {
      debug.open(a["--debug-jsonl"]);
      require_condition(bool(debug), "debug output");
    }
    BlockDigest evhash, eventhash;
    ReplayMetrics total, eval;
    int64_t eviction_count = 0, Wpre = 0, positive_offers = 0;
    size_t block_offset = 0, nextcut = 0;
    bool snapped = false, initialized = false;
    string warmh, warms;
    int64_t warmE = 0;
    auto emit_event = [&](int32_t block, int32_t request_index) {
      eventhash.add(block);
      eventhash.add(request_index);
      if (eventsout) {
        int32_t z[2] = {block, request_index};
        eventsout.write((char *)z, sizeof z);
      }
    };
    auto bit = [&](const MappedArray<uint8_t> &m, int64_t e) {
      require_condition(size_t(e / 8) < m.n, "decision bit coverage");
      return bool((m[e / 8] >> (e % 8)) & 1);
    };
    // Process each request completely before the next request changes either cache.
    for (int request_index = 0; request_index < int(block_counts.n); request_index++) {
      int64_t now = arrival_times[request_index];
      require_condition(request_index == 0 || now >= arrival_times[request_index - 1],
                        "nonmonotonic timestamps");
      require_condition(
          block_counts[request_index] >= 0 &&
              block_offset + size_t(block_counts[request_index]) <= block_ids.n &&
              token_counts[request_index] >= 0 &&
              token_counts[request_index] / 16 == block_counts[request_index],
          "token/hash mapping");
      if (now < split) {
        block_offset += block_counts[request_index];
        continue;
      }
      if (now >= split && !snapped) {
        require_condition(hbm_cache.n == 0 && ssd_cache.n == 0 &&
                              wallet.written_blocks == 0,
                          "cold day4");
        Wpre = wallet.written_blocks;
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
      while (cuts && nextcut < cuts->n && (*cuts)[nextcut] <= now) {
        refresh_scores(wallet, (*cuts)[nextcut]);
        nextcut++;
      }
      ReplayMetrics row;
      row.requests = 1;
      row.prompt_blocks = block_counts[request_index];
      row.prompt_tokens = token_counts[request_index];
      row.tail_tokens = token_counts[request_index] % 16;
      int64_t cap =
          token_counts[request_index] > 0 ? (token_counts[request_index] - 1) / 16 : 0;
      row.cap_lost = block_counts[request_index] - cap;
      require_condition(cap >= 0 && cap <= block_counts[request_index] &&
                            row.cap_lost <= 1,
                        "prefix cap");
      vector<int> ds;
      ds.reserve(block_counts[request_index]);
      for (int j = 0; j < block_counts[request_index]; j++) {
        int block = block_ids[block_offset + j];
        require_condition(block >= 0 && block < block_id_count, "block ID");
        if (mark[block] != request_index) {
          mark[block] = request_index;
          ds.push_back(block);
        }
      }
      bool bypass = ds.size() > size_t(hbm_capacity);
      row.bypass = bypass;
      vector<int> rowoffers, rowfetches;
      int64_t beforeW = wallet.written_blocks;
      if (!bypass) {
        for (int j = 0; j < cap; j++) {
          if (!hbm_cache.has(block_ids[block_offset + j]))
            break;
          row.hbm_prefix++;
        }
        // Entire prefix is selected and staged before any eviction callback.
        for (int j = 0; j < cap; j++) {
          int block = block_ids[block_offset + j];
          if (hbm_cache.has(block)) {
            row.prefix++;
            continue;
          }
          if (!ssd_cache.has(block))
            break;
          require_condition(last[block] >= 0, "SSD contains cold hash");
          row.prefix++;
          row.ssd_positions++;
          if (fetched[block] != request_index) {
            fetched[block] = request_index;
            ssd_cache.touch(block);
            row.ssd_payloads++;
            rowfetches.push_back(block);
          }
        }
        // Touch existing GPU blocks and count the slots needed for missing blocks.
        int missing = 0;
        for (int block : ds) {
          if (hbm_cache.has(block))
            hbm_cache.touch(block);
          else
            missing++;
        }
        vector<int> free_slots;
        // Offer GPU victims to SSD until enough GPU slots are available.
        while (hbm_capacity - hbm_cache.n < missing) {
          int block = hbm_cache.victim();
          require_condition(last[block] >= 0 && born[block] >= 0, "victim history");
          EvictionRecord ev{request_index, block,      last[block],
                            freq[block],   nev[block], born[block]};
          if (expected) {
            require_condition(size_t(eviction_count) < expected->n &&
                                  !memcmp(&ev, &(*expected)[eviction_count], sizeof ev),
                              "HBM eviction identity mismatch at " +
                                  to_string(eviction_count));
          }
          if (evout)
            evout.write((char *)&ev, sizeof ev);
          for (int32_t x : {ev.row, ev.block, ev.last, ev.freq, ev.nev, ev.born})
            evhash.add(x);
          emit_event(block, request_index);
          nev[block]++;
          row.evictions++;
          rowoffers.push_back(block);
          int64_t eid = eviction_count++;
          if (programs.size() < size_t((eviction_count + 7) / 8))
            programs.push_back(0);
          bool forced = mode == "bits" || (warmbits && now < split);
          bool desired =
              forced ? bit(warmbits && now < split ? *warmbits : *fixedbits, eid)
                     : false;
          // An existing SSD copy is kept for free; absent copies need admission.
          if (ssd_cache.has(block)) {
            require_condition(!desired, "fixed program on resident copy");
            ssd_cache.touch(block);
            row.free_retained++;
          } else if (now < 0) {
            require_condition(!desired, "negative-time program");
          } else {
            row.absent_offers++;
            positive_offers++;
            bool admit = false;
            double score = 0;
            if (mode == "quantile") {
              require_condition(scores && size_t(eid) < scores->n, "score coverage");
              score = (*scores)[eid];
              require_condition(isfinite(score) && score >= 0, "score value");
            }
            if (forced) {
              admit = desired;
              require_condition(!admit || wallet.written_blocks < write_budget,
                                "forced program exceeds budget");
            } else if (mode == "all")
              admit = wallet.written_blocks < write_budget;
            else if (mode == "coin")
              admit = wallet.written_blocks < write_budget && pick(eid);
            else if (mode == "stride")
              admit = wallet.written_blocks < write_budget &&
                      positive_offers % threshold == 0;
            else if (mode == "access_min")
              admit = wallet.written_blocks < write_budget && freq[block] >= threshold;
            else if (mode == "eviction_min")
              admit = wallet.written_blocks < write_budget && nev[block] >= threshold;
            else if (mode == "eviction_exact")
              admit = wallet.written_blocks < write_budget && nev[block] == threshold;
            else
              admit = wallet.pass(score, eid, seed, now);
            // Charge one block and store it, replacing the oldest SSD entry if full.
            if (admit) {
              require_condition(wallet.written_blocks < write_budget, "write budget");
              wallet.written_blocks++;
              row.reprograms += ever[block];
              ever[block] = 1;
              programs[eid / 8] |= uint8_t(1u << (eid % 8));
              int recycle = -1;
              if (ssd_cache.n == ssd_cache.cap) {
                recycle = ssd_cache.pop();
                row.ssd_evictions++;
              }
              ssd_cache.add(block, recycle);
            }
            // Observation follows this decision; the current score never sets its own
            // cutoff.
            if (mode == "quantile")
              wallet.observe(score, eid, now);
          }
          free_slots.push_back(hbm_cache.pop());
        }
        // Insert the request’s missing blocks into the freed GPU slots.
        for (int block : ds)
          if (!hbm_cache.has(block)) {
            if (last[block] >= 0) {
              emit_event(~block, request_index);
              row.eligible_raw++;
            }
            int recycle = -1;
            if (!free_slots.empty()) {
              recycle = free_slots.back();
              free_slots.pop_back();
            }
            hbm_cache.add(block, recycle);
            born[block] = request_index;
          }
        require_condition(free_slots.empty(), "unused HBM slots");
        for (int block : ds) {
          last[block] = request_index;
          require_condition(freq[block] < INT_MAX, "frequency overflow");
          freq[block]++;
        }
      }
      // Save the actual reads and writes; these are raw prefix counts.
      row.writes = wallet.written_blocks - beforeW;
      require_condition(row.prefix >= row.hbm_prefix &&
                            row.ssd_positions <= row.prefix - row.hbm_prefix,
                        "prefix metric invariant");
      auto add = [&](ReplayMetrics &m) {
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
        int64_t z[13] = {request_index,
                         now,
                         block_counts[request_index],
                         row.hbm_prefix,
                         row.prefix,
                         row.ssd_positions,
                         row.ssd_payloads,
                         row.writes,
                         row.evictions,
                         row.bypass,
                         row.cap_lost,
                         row.eligible_raw,
                         token_counts[request_index]};
        rows.write((char *)z, sizeof z);
      }
      if (debug) {
        debug << "{\"row\":" << request_index << ",\"now\":" << now
              << ",\"metrics\":" << metrics_json(row)
              << ",\"writes_total\":" << wallet.written_blocks
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
      Wpre = wallet.written_blocks;
      warmE = eviction_count;
      warmh = hbm_cache.hash();
      warms = ssd_cache.hash();
    }
    require_condition(wallet.written_blocks <= write_budget &&
                          eval.writes == wallet.written_blocks - Wpre,
                      "final wallet identity");
    rows.close();
    require_condition(bool(rows), "row output flush");
    if (evout) {
      evout.close();
      require_condition(bool(evout), "eviction flush");
    }
    if (eventsout) {
      eventsout.close();
      require_condition(bool(eventsout), "events flush");
    }
    ofstream bitsout(out + ".programs.bits", ios::binary);
    bitsout.write((char *)programs.data(), programs.size());
    bitsout.close();
    require_condition(bool(bitsout), "program bit output");
    cout << "{\"event\":\"result\",\"status\":\"PASS\",\"schema\":\"ssd_joint_prefix_"
            "atomic_v1\",\"physical_scope\":\"request_atomic_hash_deduplicated_HBM_not_"
            "vLLM_slots\",\"primary_metric\":\"ssd_fetched_prefix_positions\","
            "\"session_filter\":false,\"budget\":"
         << write_budget << ",\"writes\":" << wallet.written_blocks
         << ",\"W_pre\":" << Wpre << ",\"W_test\":" << wallet.written_blocks - Wpre
         << ",\"processed_evictions\":" << eviction_count
         << ",\"warm_evictions\":" << warmE
         << ",\"controller_observations\":" << wallet.observations
         << ",\"repricing_snapshots\":" << nextcut << ",\"peak_hbm\":" << hbm_cache.peak
         << ",\"peak_ssd\":" << ssd_cache.peak << ",\"warm_hbm_sha256\":\"" << warmh
         << "\",\"warm_ssd_sha256\":\"" << warms << "\",\"final_hbm_sha256\":\""
         << hbm_cache.hash() << "\",\"final_ssd_sha256\":\"" << ssd_cache.hash()
         << "\",\"eviction_records_sha256\":\"" << evhash.finish()
         << "\",\"hbm_signed_event_sha256\":\"" << eventhash.finish()
         << "\",\"total\":" << metrics_json(total)
         << ",\"evaluation\":" << metrics_json(eval) << "}" << endl;
    return 0;
  } catch (const exception &e) {
    cerr << "ssd_prefix_replay: " << e.what() << '\n';
    return 2;
  }
}
