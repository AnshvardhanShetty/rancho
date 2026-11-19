// Independently check the SSD replay's cache transitions and write decisions.
//
// Input maps the supplied arrays. LRU uses a separate cache implementation.
// Count records hits, writes and evictions. main() reads the write-decision log,
// reconstructs both caches and produces its own per-request counts and digests.
// A different result or an invalid supplied decision stops the check.
//
// Assumptions: the same request-atomic model as ssd_replay.cpp. This verifies
// cache and write accounting, not predictor quality or serving performance.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <openssl/sha.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
static void demand(bool b, const string &s) {
  if (!b)
    throw runtime_error(s);
}
template <class T> class Input {
  int fd;
  size_t bytes;
  const T *data = nullptr;

public:
  size_t size;
  explicit Input(const string &p) {
    fd = open(p.c_str(), O_RDONLY);
    demand(fd >= 0, "open " + p);
    struct stat st{};
    demand(fstat(fd, &st) == 0, "stat");
    bytes = st.st_size;
    demand(bytes % sizeof(T) == 0, "alignment");
    size = bytes / sizeof(T);
    if (bytes) {
      auto q = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
      demand(q != MAP_FAILED, "map");
      data = (const T *)q;
    }
  }
  ~Input() {
    if (bytes)
      munmap((void *)data, bytes);
    close(fd);
  }
  T operator[](size_t j) const {
    demand(j < size, "input bounds");
    return data[j];
  }
};
class Digest {
  SHA256_CTX h;

public:
  Digest() { SHA256_Init(&h); }
  void value(int32_t n) { SHA256_Update(&h, &n, 4); }
  string str() const {
    auto c = h;
    unsigned char x[32];
    SHA256_Final(x, &c);
    ostringstream o;
    for (auto b : x)
      o << hex << setw(2) << setfill('0') << int(b);
    return o.str();
  }
};
// Independent intrusive LRU. Head is LRU, tail is MRU. Full16GB
// parity against the original hash/list auditor is required by dense_gate.py.
static int audit_domain = 0;
class LRU {
  vector<int32_t> previous, next;
  int head = -1, tail = -1;
  size_t count = 0;

public:
  int capacity;
  int64_t peak = 0;
  explicit LRU(int c)
      : previous(audit_domain, -2), next(audit_domain, -1), capacity(c) {}
  bool contains(int b) const { return previous[b] != -2; }
  size_t size() const { return count; }
  void touch(int b) {
    demand(contains(b), "touch missing");
    if (tail == b)
      return;
    int p = previous[b], n = next[b];
    if (p >= 0)
      next[p] = n;
    else
      head = n;
    if (n >= 0)
      previous[n] = p;
    previous[b] = tail;
    next[b] = -1;
    if (tail >= 0)
      next[tail] = b;
    else
      head = b;
    tail = b;
  }
  void insert(int b) {
    demand(!contains(b) && int(count) < capacity, "insert capacity");
    previous[b] = tail;
    next[b] = -1;
    if (tail >= 0)
      next[tail] = b;
    else
      head = b;
    tail = b;
    count++;
    peak = max(peak, int64_t(count));
  }
  int victim() const {
    demand(head >= 0, "empty victim");
    return head;
  }
  int evict() {
    int b = victim(), n = next[b];
    head = n;
    if (n >= 0)
      previous[n] = -1;
    else
      tail = -1;
    previous[b] = -2;
    next[b] = -1;
    count--;
    return b;
  }
  string digest() const {
    Digest h;
    size_t n = 0;
    for (int b = tail; b >= 0; b = previous[b]) {
      h.value(b);
      demand(++n <= count, "LRU cycle");
    }
    demand(n == count, "LRU length");
    return h.str();
  }
};
struct Count {
  int64_t requests = 0, prompt_blocks = 0, prompt_tokens = 0, tail = 0, hbm = 0,
          prefix = 0, ssd = 0, payloads = 0, writes = 0, evictions = 0, absent = 0,
          free = 0, raw = 0, bypass = 0, capped = 0, reprograms = 0, ssd_evictions = 0;
  void add(const Count &a) {
    requests += a.requests;
    prompt_blocks += a.prompt_blocks;
    prompt_tokens += a.prompt_tokens;
    tail += a.tail;
    hbm += a.hbm;
    prefix += a.prefix;
    ssd += a.ssd;
    payloads += a.payloads;
    writes += a.writes;
    evictions += a.evictions;
    absent += a.absent;
    free += a.free;
    raw += a.raw;
    bypass += a.bypass;
    capped += a.capped;
    reprograms += a.reprograms;
    ssd_evictions += a.ssd_evictions;
  }
  string json() const {
    ostringstream o;
    o << "{\"requests\":" << requests << ",\"prompt_blocks\":" << prompt_blocks
      << ",\"prompt_tokens\":" << prompt_tokens
      << ",\"unrepresented_tail_tokens\":" << tail
      << ",\"hbm_only_prefix_positions\":" << hbm
      << ",\"combined_prefix_positions\":" << prefix
      << ",\"ssd_fetched_prefix_positions\":" << ssd
      << ",\"ssd_payload_reads\":" << payloads
      << ",\"incremental_prefix_positions\":" << prefix - hbm
      << ",\"physical_writes\":" << writes << ",\"hbm_evictions\":" << evictions
      << ",\"absent_offers\":" << absent << ",\"free_retained\":" << free
      << ",\"raw_repeated_hbm_misses_diagnostic\":" << raw
      << ",\"oversized_bypass\":" << bypass
      << ",\"last_token_cap_excluded_positions\":" << capped
      << ",\"reprograms\":" << reprograms << ",\"ssd_evictions\":" << ssd_evictions
      << '}';
    return o.str();
  }
};
int main(int argc, char **argv) {
  try {
    map<string, string> a;
    for (int j = 1; j < argc; j += 2) {
      demand(j + 1 < argc, "option value");
      a[argv[j]] = argv[j + 1];
    }
    auto get = [&](string k) {
      demand(a.count(k), "missing " + k);
      return a[k];
    };
    string p = get("--prefix"), out = get("--out-prefix");
    int NB = stoi(get("--nb")), CH = stoi(get("--hbm"));
    int64_t cs = stoll(get("--ssd")), B = stoll(get("--budget")),
            split = stoll(get("--eval-start-us")), stop = stoll(get("--stop-us"));
    demand(NB > 0 && CH > 0 && cs > 0 && B >= 0 && split >= 0 && stop >= 0, "geometry");
    string policy = a.count("--verify-policy") ? a["--verify-policy"] : "bits";
    uint64_t seed = stoull(a.count("--seed") ? a["--seed"] : "17");
    int64_t numerator =
                stoll(a.count("--probability-numerator") ? a["--probability-numerator"]
                                                         : "0"),
            denominator = stoll(a.count("--probability-denominator")
                                    ? a["--probability-denominator"]
                                    : "1");
    int threshold = stoi(a.count("--threshold") ? a["--threshold"] : "0");
    demand(policy == "bits" || policy == "all" || policy == "coin" ||
               policy == "stride" || policy == "access_min" ||
               policy == "eviction_min" || policy == "eviction_exact",
           "audit policy");
    if (policy != "bits" && policy != "all" && policy != "coin")
      demand(threshold > 0, "audit threshold");
    demand(numerator >= 0 && denominator > 0 && numerator <= denominator,
           "audit probability");
    auto choose = [&](int64_t id) {
      uint64_t x = uint64_t(id) * 0x9e3779b97f4a7c15ULL ^ seed ^ 0x5bf03635ULL;
      x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
      x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
      x ^= x >> 33;
      return __int128(x >> 11) * denominator <
             __int128(numerator) * (uint64_t(1) << 53);
    };
    Input<int32_t> nb(p + "_nb.i32"), ids(p + "_ids.i32");
    Input<int64_t> ts(p + "_ts.i64"), tokens(get("--tokens"));
    Input<uint8_t> bits(get("--program-bits"));
    demand(nb.size == ts.size && nb.size == tokens.size, "request length");
    audit_domain = NB;
    LRU H(min(NB, CH)), S(int(min<int64_t>(NB, cs)));
    vector<int32_t> last(NB, -1), born(NB, -1), frequency(NB), eviction_count(NB);
    vector<uint8_t> written(NB);
    size_t offset = 0;
    int64_t W = 0, E = 0, WP = 0, EP = 0, positive = 0;
    bool snapped = false;
    string warmh, warms;
    Digest event_hash, eviction_hash;
    Count total, evaluation;
    ofstream rows(out + ".rows.i64", ios::binary);
    demand(bool(rows), "row output");
    auto event = [&](int32_t b, int32_t r) {
      event_hash.value(b);
      event_hash.value(r);
    };
    for (size_t row = 0; row < nb.size; row++) {
      auto now = ts[row];
      demand(row == 0 || now >= ts[row - 1], "timestamp order");
      demand(nb[row] >= 0 && offset + size_t(nb[row]) <= ids.size && tokens[row] >= 0 &&
                 tokens[row] / 16 == nb[row],
             "token/hash mapping");
      if (now < split) {
        offset += nb[row];
        continue;
      }
      if (now >= split && !snapped) {
        demand(H.size() == 0 && S.size() == 0 && W == 0, "cold day4");
        WP = W;
        EP = E;
        warmh = H.digest();
        warms = S.digest();
        snapped = true;
      }
      if (now >= stop)
        break;
      Count m;
      m.requests = 1;
      m.prompt_blocks = nb[row];
      m.prompt_tokens = tokens[row];
      m.tail = tokens[row] % 16;
      int64_t max_prefix = tokens[row] > 0 ? (tokens[row] - 1) / 16 : 0;
      m.capped = nb[row] - max_prefix;
      vector<int> distinct;
      unordered_set<int> seen;
      for (int j = 0; j < nb[row]; j++) {
        int b = ids[offset + j];
        demand(b >= 0 && b < NB, "block ID");
        if (seen.insert(b).second)
          distinct.push_back(b);
      }
      m.bypass = distinct.size() > size_t(CH);
      if (!m.bypass) {
        for (int j = 0; j < max_prefix; j++) {
          if (!H.contains(ids[offset + j]))
            break;
          m.hbm++;
        }
        vector<int> restore;
        unordered_set<int> staged;
        // Plan all reads against original physical state. A fetched payload is
        // staged before any later admission can replace its SSD source entry.
        for (int j = 0; j < max_prefix; j++) {
          int b = ids[offset + j];
          if (H.contains(b)) {
            m.prefix++;
            continue;
          }
          if (!S.contains(b))
            break;
          m.prefix++;
          m.ssd++;
          demand(last[b] >= 0, "SSD cold hash");
          if (staged.insert(b).second)
            restore.push_back(b);
        }
        for (int b : restore) {
          S.touch(b);
          m.payloads++;
        }
        vector<int> missing;
        for (int b : distinct) {
          if (H.contains(b))
            H.touch(b);
          else
            missing.push_back(b);
        }
        while (H.size() + missing.size() > size_t(CH)) {
          int b = H.victim();
          demand(last[b] >= 0 && born[b] >= 0, "victim history");
          for (int32_t x : {int32_t(row), int32_t(b), last[b], frequency[b],
                            eviction_count[b], born[b]})
            eviction_hash.value(x);
          eviction_count[b]++;
          event(b, row);
          m.evictions++;
          demand(size_t(E / 8) < bits.size, "missing program bit");
          int64_t eid = E;
          bool program = (bits[E / 8] >> (E % 8)) & 1;
          E++;
          if (S.contains(b)) {
            demand(!program, "program resident");
            S.touch(b);
            m.free++;
          } else if (now < 0)
            demand(!program, "negative-time write");
          else {
            m.absent++;
            positive++;
            bool wanted = false;
            if (policy == "all")
              wanted = true;
            else if (policy == "coin")
              wanted = choose(eid);
            else if (policy == "stride")
              wanted = positive % threshold == 0;
            else if (policy == "access_min")
              wanted = frequency[b] >= threshold;
            else if (policy == "eviction_min")
              wanted = eviction_count[b] >= threshold;
            else if (policy == "eviction_exact")
              wanted = eviction_count[b] == threshold;
            if (policy != "bits")
              demand(program == (W < B && wanted),
                     "admission rule or hard-budget decision mismatch");
            if (program) {
              demand(W < B, "budget violation");
              W++;
              m.writes++;
              m.reprograms += written[b];
              written[b] = 1;
              if (S.size() == size_t(S.capacity)) {
                S.evict();
                m.ssd_evictions++;
              }
              S.insert(b);
            }
          }
          demand(H.evict() == b, "victim changed");
        }
        for (int b : missing) {
          if (last[b] >= 0) {
            event(~b, row);
            m.raw++;
          }
          H.insert(b);
          born[b] = row;
        }
        for (int b : distinct) {
          last[b] = row;
          frequency[b]++;
        }
      }
      demand(m.prefix >= m.hbm && m.ssd <= m.prefix - m.hbm, "prefix identity");
      if (now >= 0)
        total.add(m);
      if (now >= split) {
        evaluation.add(m);
        int64_t z[13] = {int64_t(row), now,        nb[row],    m.hbm,       m.prefix,
                         m.ssd,        m.payloads, m.writes,   m.evictions, m.bypass,
                         m.capped,     m.raw,      tokens[row]};
        rows.write((char *)z, sizeof z);
      }
      offset += nb[row];
    }
    if (!snapped) {
      WP = W;
      EP = E;
      warmh = H.digest();
      warms = S.digest();
    }
    demand(bits.size == size_t((E + 7) / 8), "program file size");
    for (size_t k = E; k < bits.size * 8; k++)
      demand(!((bits[k / 8] >> (k % 8)) & 1), "nonzero program padding");
    demand(W <= B && evaluation.writes == W - WP, "wallet accounting");
    rows.close();
    demand(bool(rows), "rows flush");
    cout << "{\"event\":\"result\",\"status\":\"PASS\",\"schema\":\"ssd_prefix_"
            "independent_fixed_program_audit_v1\",\"budget\":"
         << B << ",\"writes\":" << W << ",\"W_pre\":" << WP << ",\"W_test\":" << W - WP
         << ",\"processed_evictions\":" << E << ",\"warm_evictions\":" << EP
         << ",\"peak_hbm\":" << H.peak << ",\"peak_ssd\":" << S.peak
         << ",\"warm_hbm_sha256\":\"" << warmh << "\",\"warm_ssd_sha256\":\"" << warms
         << "\",\"final_hbm_sha256\":\"" << H.digest() << "\",\"final_ssd_sha256\":\""
         << S.digest() << "\",\"eviction_records_sha256\":\"" << eviction_hash.str()
         << "\",\"hbm_signed_event_sha256\":\"" << event_hash.str()
         << "\",\"total\":" << total.json() << ",\"evaluation\":" << evaluation.json()
         << "}" << endl;
    return 0;
  } catch (const exception &e) {
    cerr << "independent_audit: " << e.what() << endl;
    return 2;
  }
}
