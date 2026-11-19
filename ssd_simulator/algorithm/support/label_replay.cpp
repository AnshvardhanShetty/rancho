// Build training labels against a supplied reference replay's write decisions.
// Marginal measures how much of the later readable prefix each sampled victim
// would unlock alone. Label events record the request, sample and credited gain.
//
// Assumes the reference write bits, eviction identities and same-session credit
// index are verified inputs. No future read enters a label before its cutoff.
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
  return double(mix(uint64_t(id) * 0x9e3779b97f4a7c15ULL ^ seed ^ 0x5bf03635ULL) >>
                11) /
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
      auto lo = order.order_of_key({th, LLONG_MIN}),
           hi = order.order_of_key({th, LLONG_MAX});
      threshold = th;
      tie = clamp((wanted - (hist.size() - hi)) / double(hi - lo), 0.0, 1.0);
    }
    have = true;
  }
  bool pass(double score, int64_t id, uint64_t seed, int64_t now) {
    phase(now);
    return W < budget && W < cap &&
           (!have || score > threshold ||
            (score == threshold && random01(id, seed) < tie));
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
static string field(const string &s, const string &k) {
  string key = '"' + k + '"';
  auto p = s.find(key);
  ck(p != string::npos, "missing response " + k);
  p = s.find(':', p + key.size());
  ck(p != string::npos, "response colon");
  return s.substr(p + 1);
}
static vector<double> numeric_array(const string &s, const string &k) {
  auto t = field(s, k);
  auto p = t.find('[');
  ck(p != string::npos, "response array");
  vector<double> a;
  const char *x = t.c_str() + p + 1;
  while (true) {
    while (isspace(*x) || *x == ',')
      x++;
    if (*x == ']')
      break;
    char *z = nullptr;
    double y = strtod(x, &z);
    ck(z != x && isfinite(y), "invalid response number");
    a.push_back(y);
    x = z;
  }
  return a;
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
  int64_t requests = 0, prompt_blocks = 0, prompt_tokens = 0, tail_tokens = 0,
          hbm_prefix = 0, prefix = 0, ssd_positions = 0, ssd_payloads = 0, writes = 0,
          evictions = 0, absent_offers = 0, free_retained = 0, eligible_raw = 0,
          bypass = 0, cap_lost = 0, reprograms = 0, ssd_evictions = 0;
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
    << ",\"absent_offers\":" << m.absent_offers
    << ",\"free_retained\":" << m.free_retained
    << ",\"raw_repeated_hbm_misses_diagnostic\":" << m.eligible_raw
    << ",\"oversized_bypass\":" << m.bypass
    << ",\"last_token_cap_excluded_positions\":" << m.cap_lost
    << ",\"reprograms\":" << m.reprograms << ",\"ssd_evictions\":" << m.ssd_evictions
    << '}';
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
    map<string, string> a;
    set<string> flags;
    for (int j = 1; j < argc; j++) {
      string k = argv[j];
      ck(k.rfind("--", 0) == 0, "expected option");
      if (k == "--reprice-stdio") {
        ck(!flags.count(k), "duplicate flag");
        flags.insert(k);
      } else {
        ck(allowed.count(k), "unknown option " + k);
        ck(j + 1 < argc, "option value");
        ck(!a.count(k), "duplicate option");
        a[k] = argv[++j];
      }
    }
    auto req = [&](string k) {
      ck(a.count(k), "missing " + k);
      return a[k];
    };
    auto val = [&](string k, string d) { return a.count(k) ? a[k] : d; };
    string p = req("--prefix"), out = req("--out-prefix"),
           mode = val("--mode", "quantile");
    int NB = stoi(req("--nb")), CH = stoi(req("--hbm"));
    int64_t CS64 = stoll(req("--ssd")), B = stoll(req("--budget")),
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
    ck(NB > 0 && CH > 0 && CS64 > 0 && B >= 0 && split >= 0 && end > split &&
           stop <= end && stop >= 0,
       "geometry");
    ck(update > 0 && history >= 100 && release > 0 && release <= end && reserve >= 0 &&
           reserve < 1 && isfinite(demand) && demand > 0 && coin >= 0 && coin <= 1,
       "controller configuration");
    ck(mode == "quantile" || mode == "all" || mode == "coin" || mode == "bits",
       "policy mode");
    Map<int32_t> nb(p + "_nb.i32"), ids(p + "_ids.i32");
    Map<int64_t> ts(p + "_ts.i64"), tokens(req("--tokens"));
    ck(nb.n == ts.n && nb.n == tokens.n && nb.n <= INT_MAX, "request shapes");
    unique_ptr<Map<double>> scores;
    if (a.count("--scores"))
      scores = make_unique<Map<double>>(a["--scores"]);
    ck(mode != "quantile" || scores, "quantile scores required");
    unique_ptr<Map<uint8_t>> warmbits, fixedbits;
    if (a.count("--warm-program-bits"))
      warmbits = make_unique<Map<uint8_t>>(a["--warm-program-bits"]);
    if (a.count("--fixed-program-bits"))
      fixedbits = make_unique<Map<uint8_t>>(a["--fixed-program-bits"]);
    ck(mode != "bits" || fixedbits, "fixed bits required");
    unique_ptr<Map<Eviction>> expected;
    if (a.count("--expected-evictions"))
      expected = make_unique<Map<Eviction>>(a["--expected-evictions"]);
    unique_ptr<Map<int64_t>> cuts;
    if (a.count("--cuts-i64")) {
      cuts = make_unique<Map<int64_t>>(a["--cuts-i64"]);
      ck(cuts->n > 0 && (*cuts)[0] == split,
         "initial candidate calibration cut required");
      for (size_t j = 0; j < cuts->n; j++)
        ck((*cuts)[j] >= split && (*cuts)[j] < end &&
               (j == 0 || (*cuts)[j] > (*cuts)[j - 1]),
           "invalid fit cuts");
    }
    ck(!flags.count("--reprice-stdio") || cuts, "repricing cuts required");
    ck(!cuts || flags.count("--reprice-stdio"), "cuts without repricing");
    ck(!cuts || mode == "quantile", "only quantile reprices");
    ck(access((out + ".rows.i64").c_str(), F_OK) != 0 &&
           access((out + ".programs.bits").c_str(), F_OK) != 0,
       "output already exists");
    for (string k : {"--evictions-out", "--events-out", "--debug-jsonl"})
      if (a.count(k))
        ck(access(a[k].c_str(), F_OK) != 0, "output already exists " + k);
    Cache h(NB, min(NB, CH)), s(NB, int(min<int64_t>(NB, CS64)));
    vector<int32_t> mark(NB, -1), fetched(NB, -1), last(NB, -1), freq(NB), nev(NB),
        born(NB, -1);
    vector<uint8_t> ever(NB), programs;
    Wallet wallet(B, end, release, update, history, reserve, demand);
    ofstream rows(out + ".rows.i64", ios::binary), evout, eventsout, debug;
    ck(bool(rows), "rows output");
    if (a.count("--evictions-out")) {
      evout.open(a["--evictions-out"], ios::binary);
      ck(bool(evout), "eviction output");
    }
    if (a.count("--events-out")) {
      eventsout.open(a["--events-out"], ios::binary);
      ck(bool(eventsout), "event output");
    }
    if (a.count("--debug-jsonl")) {
      debug.open(a["--debug-jsonl"]);
      ck(bool(debug), "debug output");
    }
    Marginal marginal(NB, req("--label-samples"), req("--label-credit"), out,
                      stoll(val("--label-event-offset", "0")));
    Hash evhash, eventhash;
    Metrics total, eval;
    int64_t E = 0, Wpre = 0;
    size_t off = 0, nextcut = 0;
    bool snapped = false, initialized = false;
    string warmh, warms;
    int64_t warmE = 0;
    auto emit_event = [&](int32_t b, int32_t r) {
      eventhash.add(b);
      eventhash.add(r);
      if (eventsout) {
        int32_t z[2] = {b, r};
        eventsout.write((char *)z, sizeof z);
      }
    };
    auto bit = [&](const Map<uint8_t> &m, int64_t e) {
      ck(size_t(e / 8) < m.n, "decision bit coverage");
      return bool((m[e / 8] >> (e % 8)) & 1);
    };
    for (int r = 0; r < int(nb.n); r++) {
      int64_t now = ts[r];
      ck(r == 0 || now >= ts[r - 1], "nonmonotonic timestamps");
      ck(nb[r] >= 0 && off + size_t(nb[r]) <= ids.n && tokens[r] >= 0 &&
             tokens[r] / 16 == nb[r],
         "token/hash mapping");
      if (now < split) {
        off += nb[r];
        continue;
      }
      if (now >= split && !snapped) {
        ck(h.n == 0 && s.n == 0 && wallet.W == 0, "cold day4");
        Wpre = wallet.W;
        warmE = E;
        warmh = h.hash();
        warms = s.hash();
        snapped = true;
      }
      if (now >= stop)
        break;
      if (now >= split && !initialized) {
        wallet.recompute(split);
        initialized = true;
      }
      while (cuts && nextcut < cuts->n && (*cuts)[nextcut] <= now) {
        reprice(wallet, (*cuts)[nextcut]);
        nextcut++;
      }
      Metrics row;
      row.requests = 1;
      row.prompt_blocks = nb[r];
      row.prompt_tokens = tokens[r];
      row.tail_tokens = tokens[r] % 16;
      int64_t cap = tokens[r] > 0 ? (tokens[r] - 1) / 16 : 0;
      row.cap_lost = nb[r] - cap;
      ck(cap >= 0 && cap <= nb[r] && row.cap_lost <= 1, "prefix cap");
      vector<int> ds;
      ds.reserve(nb[r]);
      for (int j = 0; j < nb[r]; j++) {
        int b = ids[off + j];
        ck(b >= 0 && b < NB, "block ID");
        if (mark[b] != r) {
          mark[b] = r;
          ds.push_back(b);
        }
      }
      bool bypass = ds.size() > size_t(CH);
      row.bypass = bypass;
      vector<int> rowoffers, rowfetches;
      int64_t beforeW = wallet.W;
      if (!bypass) {
        for (int j = 0; j < cap; j++) {
          if (!h.has(ids[off + j]))
            break;
          row.hbm_prefix++;
        }
        // Entire prefix is selected and staged before any eviction callback.
        for (int j = 0; j < cap; j++) {
          int b = ids[off + j];
          if (h.has(b)) {
            row.prefix++;
            continue;
          }
          if (!s.has(b))
            break;
          ck(last[b] >= 0, "SSD contains cold hash");
          row.prefix++;
          row.ssd_positions++;
          if (fetched[b] != r) {
            fetched[b] = r;
            s.touch(b);
            row.ssd_payloads++;
            rowfetches.push_back(b);
          }
        }
        marginal.measure(h, s, ids, off, cap, row.prefix, r);
        int missing = 0;
        for (int b : ds) {
          if (h.has(b))
            h.touch(b);
          else
            missing++;
        }
        vector<int> free_slots;
        while (CH - h.n < missing) {
          int b = h.victim();
          ck(last[b] >= 0 && born[b] >= 0, "victim history");
          Eviction ev{r, b, last[b], freq[b], nev[b], born[b]};
          if (expected) {
            ck(size_t(E) < expected->n && !memcmp(&ev, &(*expected)[E], sizeof ev),
               "HBM eviction identity mismatch at " + to_string(E));
          }
          if (evout)
            evout.write((char *)&ev, sizeof ev);
          for (int32_t x : {ev.row, ev.block, ev.last, ev.freq, ev.nev, ev.born})
            evhash.add(x);
          emit_event(b, r);
          nev[b]++;
          row.evictions++;
          rowoffers.push_back(b);
          int64_t eid = E++;
          if (programs.size() < size_t((E + 7) / 8))
            programs.push_back(0);
          marginal.offer(b, eid + stoll(val("--label-event-offset", "0")));
          bool forced = mode == "bits" || (warmbits && now < split);
          bool desired =
              forced ? bit(warmbits && now < split ? *warmbits : *fixedbits, eid)
                     : false;
          if (s.has(b)) {
            ck(!desired, "fixed program on resident copy");
            s.touch(b);
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
              ck(!admit || wallet.W < B, "forced program exceeds budget");
            } else if (mode == "all")
              admit = wallet.W < B;
            else if (mode == "coin")
              admit = wallet.W < B && random01(eid, seed) < coin;
            else
              admit = wallet.pass(score, eid, seed, now);
            if (admit) {
              ck(wallet.W < B, "write budget");
              wallet.W++;
              row.reprograms += ever[b];
              ever[b] = 1;
              programs[eid / 8] |= uint8_t(1u << (eid % 8));
              int recycle = -1;
              if (s.n == s.cap) {
                recycle = s.pop();
                row.ssd_evictions++;
              }
              s.add(b, recycle);
            }
            // Observation follows this decision; the current score never sets its own
            // cutoff.
            if (mode == "quantile")
              wallet.observe(score, eid, now);
          }
          free_slots.push_back(h.pop());
        }
        for (int b : ds)
          if (!h.has(b)) {
            if (last[b] >= 0) {
              emit_event(~b, r);
              row.eligible_raw++;
            }
            int recycle = -1;
            if (!free_slots.empty()) {
              recycle = free_slots.back();
              free_slots.pop_back();
            }
            h.add(b, recycle);
            born[b] = r;
          }
        ck(free_slots.empty(), "unused HBM slots");
        for (int b : ds) {
          last[b] = r;
          ck(freq[b] < INT_MAX, "frequency overflow");
          freq[b]++;
        }
      }
      row.writes = wallet.W - beforeW;
      ck(row.prefix >= row.hbm_prefix &&
             row.ssd_positions <= row.prefix - row.hbm_prefix,
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
        int64_t z[13] = {r,
                         now,
                         nb[r],
                         row.hbm_prefix,
                         row.prefix,
                         row.ssd_positions,
                         row.ssd_payloads,
                         row.writes,
                         row.evictions,
                         row.bypass,
                         row.cap_lost,
                         row.eligible_raw,
                         tokens[r]};
        rows.write((char *)z, sizeof z);
      }
      if (debug) {
        debug << "{\"row\":" << r << ",\"now\":" << now
              << ",\"metrics\":" << metrics_json(row)
              << ",\"writes_total\":" << wallet.W << ",\"hbm\":" << h.json()
              << ",\"ssd\":" << s.json() << ",\"offers\":[";
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
      off += nb[r];
    }
    if (!snapped) {
      Wpre = wallet.W;
      warmE = E;
      warmh = h.hash();
      warms = s.hash();
    }
    ck(wallet.W <= B && eval.writes == wallet.W - Wpre, "final wallet identity");
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
    ofstream bitsout(out + ".programs.bits", ios::binary);
    bitsout.write((char *)programs.data(), programs.size());
    bitsout.close();
    ck(bool(bitsout), "program bit output");
    cout << "{\"event\":\"result\",\"status\":\"PASS\",\"schema\":\"ssd_joint_prefix_"
            "atomic_v1\",\"physical_scope\":\"request_atomic_hash_deduplicated_HBM_not_"
            "vLLM_slots\",\"primary_metric\":\"ssd_fetched_prefix_positions\","
            "\"session_filter\":false,\"budget\":"
         << B << ",\"writes\":" << wallet.W << ",\"W_pre\":" << Wpre
         << ",\"W_test\":" << wallet.W - Wpre << ",\"processed_evictions\":" << E
         << ",\"warm_evictions\":" << warmE
         << ",\"controller_observations\":" << wallet.observations
         << ",\"repricing_snapshots\":" << nextcut << ",\"peak_hbm\":" << h.peak
         << ",\"peak_ssd\":" << s.peak << ",\"warm_hbm_sha256\":\"" << warmh
         << "\",\"warm_ssd_sha256\":\"" << warms << "\",\"final_hbm_sha256\":\""
         << h.hash() << "\",\"final_ssd_sha256\":\"" << s.hash()
         << "\",\"eviction_records_sha256\":\"" << evhash.finish()
         << "\",\"hbm_signed_event_sha256\":\"" << eventhash.finish()
         << "\",\"total\":" << metrics_json(total)
         << ",\"evaluation\":" << metrics_json(eval) << "}" << endl;
    marginal.events.close();
    ck(bool(marginal.events) && marginal.tests > 0, "label verification");
    cerr << "explicit toggles verified " << marginal.tests << endl;
    return 0;
  } catch (const exception &e) {
    cerr << "ssd_prefix_replay: " << e.what() << '\n';
    return 2;
  }
}
