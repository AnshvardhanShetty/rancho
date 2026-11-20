// Check the SSD replay using a separate cache implementation
// Read the input requests and saved write decisions in the same order
// Rebuild both caches and compare the resulting reads and writes

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

  explicit LRU(int c) : previous(audit_domain, -2), next(audit_domain, -1), capacity(c) {}

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
  int64_t requests = 0, prompt_blocks = 0, prompt_tokens = 0, tail = 0, hbm = 0, prefix = 0,
          ssd = 0, payloads = 0, writes = 0, evictions = 0, absent = 0, free = 0, raw = 0,
          bypass = 0, capped = 0, reprograms = 0, ssd_evictions = 0;

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
      << ",\"prompt_tokens\":" << prompt_tokens << ",\"unrepresented_tail_tokens\":" << tail
      << ",\"hbm_only_prefix_positions\":" << hbm << ",\"combined_prefix_positions\":" << prefix
      << ",\"ssd_fetched_prefix_positions\":" << ssd << ",\"ssd_payload_reads\":" << payloads
      << ",\"incremental_prefix_positions\":" << prefix - hbm
      << ",\"physical_writes\":" << writes << ",\"hbm_evictions\":" << evictions
      << ",\"absent_offers\":" << absent << ",\"free_retained\":" << free
      << ",\"raw_repeated_hbm_misses_diagnostic\":" << raw << ",\"oversized_bypass\":" << bypass
      << ",\"last_token_cap_excluded_positions\":" << capped << ",\"reprograms\":" << reprograms
      << ",\"ssd_evictions\":" << ssd_evictions << '}';
    return o.str();
  }
};

int main(int argc, char **argv) {
  try {
    map<string, string> options;
    for (int j = 1; j < argc; j += 2) {
      demand(j + 1 < argc, "option value");
      options[argv[j]] = argv[j + 1];
    }
    auto required_option = [&](string option_name) {
      demand(options.count(option_name), "missing " + option_name);
      return options[option_name];
    };
    string stream_prefix = required_option("--prefix"),
           output_prefix = required_option("--out-prefix");
    int block_id_count = stoi(required_option("--nb")),
        hbm_capacity = stoi(required_option("--hbm"));
    int64_t ssd_capacity = stoll(required_option("--ssd")),
            write_budget = stoll(required_option("--budget")),
            split = stoll(required_option("--eval-start-us")),
            stop = stoll(required_option("--stop-us"));
    demand(block_id_count > 0 && hbm_capacity > 0 && ssd_capacity > 0 && write_budget >= 0 &&
               split >= 0 && stop >= 0,
           "geometry");
    string policy = options.count("--verify-policy") ? options["--verify-policy"] : "bits";
    uint64_t seed = stoull(options.count("--seed") ? options["--seed"] : "17");
    int64_t numerator = stoll(options.count("--probability-numerator")
                                  ? options["--probability-numerator"]
                                  : "0"),
            denominator = stoll(options.count("--probability-denominator")
                                    ? options["--probability-denominator"]
                                    : "1");
    int threshold = stoi(options.count("--threshold") ? options["--threshold"] : "0");
    demand(policy == "bits" || policy == "all" || policy == "coin" || policy == "stride" ||
               policy == "access_min" || policy == "eviction_min" || policy == "eviction_exact",
           "audit policy");
    if (policy != "bits" && policy != "all" && policy != "coin")
      demand(threshold > 0, "audit threshold");
    demand(numerator >= 0 && denominator > 0 && numerator <= denominator, "audit probability");
    auto choose = [&](int64_t id) {
      uint64_t x = uint64_t(id) * 0x9e3779b97f4a7c15ULL ^ seed ^ 0x5bf03635ULL;
      x = (x ^ (x >> 33)) * 0xff51afd7ed558ccdULL;
      x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53ULL;
      x ^= x >> 33;
      return __int128(x >> 11) * denominator < __int128(numerator) * (uint64_t(1) << 53);
    };
    Input<int32_t> block_counts(stream_prefix + "_nb.i32"),
        block_ids(stream_prefix + "_ids.i32");
    Input<int64_t> arrival_times(stream_prefix + "_ts.i64"),
        tokens(required_option("--tokens"));
    Input<uint8_t> bits(required_option("--program-bits"));
    demand(block_counts.size == arrival_times.size && block_counts.size == tokens.size,
           "request length");
    audit_domain = block_id_count;
    LRU hbm_cache(min(block_id_count, hbm_capacity)),
        ssd_cache(int(min<int64_t>(block_id_count, ssd_capacity)));
    vector<int32_t> last(block_id_count, -1), born(block_id_count, -1),
        frequency(block_id_count), eviction_count(block_id_count);
    vector<uint8_t> written(block_id_count);
    size_t offset = 0;
    int64_t blocks_written = 0, eviction_total = 0, writes_before_evaluation = 0,
            evictions_before_evaluation = 0, positive = 0;
    bool snapped = false;
    string warmh, warms;
    Digest event_hash, eviction_hash;
    Count total, evaluation;
    ofstream rows(output_prefix + ".rows.i64", ios::binary);
    demand(bool(rows), "row output");
    auto event = [&](int32_t b, int32_t r) {
      event_hash.value(b);
      event_hash.value(r);
    };
    for (size_t row = 0; row < block_counts.size; row++) {
      auto now = arrival_times[row];
      demand(row == 0 || now >= arrival_times[row - 1], "timestamp order");
      demand(block_counts[row] >= 0 && offset + size_t(block_counts[row]) <= block_ids.size &&
                 tokens[row] >= 0 && tokens[row] / 16 == block_counts[row],
             "token/hash mapping");
      if (now < split) {
        offset += block_counts[row];
        continue;
      }
      if (now >= split && !snapped) {
        demand(hbm_cache.size() == 0 && ssd_cache.size() == 0 && blocks_written == 0,
               "cold day4");
        writes_before_evaluation = blocks_written;
        evictions_before_evaluation = eviction_total;
        warmh = hbm_cache.digest();
        warms = ssd_cache.digest();
        snapped = true;
      }
      if (now >= stop)
        break;
      Count m;
      m.requests = 1;
      m.prompt_blocks = block_counts[row];
      m.prompt_tokens = tokens[row];
      m.tail = tokens[row] % 16;
      int64_t max_prefix = tokens[row] > 0 ? (tokens[row] - 1) / 16 : 0;
      m.capped = block_counts[row] - max_prefix;
      vector<int> distinct;
      unordered_set<int> seen;
      for (int j = 0; j < block_counts[row]; j++) {
        int b = block_ids[offset + j];
        demand(b >= 0 && b < block_id_count, "block ID");
        if (seen.insert(b).second)
          distinct.push_back(b);
      }
      m.bypass = distinct.size() > size_t(hbm_capacity);
      if (!m.bypass) {
        for (int j = 0; j < max_prefix; j++) {
          if (!hbm_cache.contains(block_ids[offset + j]))
            break;
          m.hbm++;
        }
        vector<int> restore;
        unordered_set<int> staged;
        // Plan all reads against original physical state. A fetched payload is
        // staged before any later admission can replace its SSD source entry.
        for (int j = 0; j < max_prefix; j++) {
          int b = block_ids[offset + j];
          if (hbm_cache.contains(b)) {
            m.prefix++;
            continue;
          }
          if (!ssd_cache.contains(b))
            break;
          m.prefix++;
          m.ssd++;
          demand(last[b] >= 0, "SSD cold hash");
          if (staged.insert(b).second)
            restore.push_back(b);
        }
        for (int b : restore) {
          ssd_cache.touch(b);
          m.payloads++;
        }
        vector<int> missing;
        for (int b : distinct) {
          if (hbm_cache.contains(b))
            hbm_cache.touch(b);
          else
            missing.push_back(b);
        }
        while (hbm_cache.size() + missing.size() > size_t(hbm_capacity)) {
          int b = hbm_cache.victim();
          demand(last[b] >= 0 && born[b] >= 0, "victim history");
          for (int32_t x :
               {int32_t(row), int32_t(b), last[b], frequency[b], eviction_count[b], born[b]})
            eviction_hash.value(x);
          eviction_count[b]++;
          event(b, row);
          m.evictions++;
          demand(size_t(eviction_total / 8) < bits.size, "missing program bit");
          int64_t eviction_id = eviction_total;
          bool program = (bits[eviction_total / 8] >> (eviction_total % 8)) & 1;
          eviction_total++;
          if (ssd_cache.contains(b)) {
            demand(!program, "program resident");
            ssd_cache.touch(b);
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
              wanted = choose(eviction_id);
            else if (policy == "stride")
              wanted = positive % threshold == 0;
            else if (policy == "access_min")
              wanted = frequency[b] >= threshold;
            else if (policy == "eviction_min")
              wanted = eviction_count[b] >= threshold;
            else if (policy == "eviction_exact")
              wanted = eviction_count[b] == threshold;
            if (policy != "bits")
              demand(program == (blocks_written < write_budget && wanted),
                     "admission rule or hard-budget decision mismatch");
            if (program) {
              demand(blocks_written < write_budget, "budget violation");
              blocks_written++;
              m.writes++;
              m.reprograms += written[b];
              written[b] = 1;
              if (ssd_cache.size() == size_t(ssd_cache.capacity)) {
                ssd_cache.evict();
                m.ssd_evictions++;
              }
              ssd_cache.insert(b);
            }
          }
          demand(hbm_cache.evict() == b, "victim changed");
        }
        for (int b : missing) {
          if (last[b] >= 0) {
            event(~b, row);
            m.raw++;
          }
          hbm_cache.insert(b);
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
        int64_t z[13] = {int64_t(row), now,        block_counts[row], m.hbm,       m.prefix,
                         m.ssd,        m.payloads, m.writes,          m.evictions, m.bypass,
                         m.capped,     m.raw,      tokens[row]};
        rows.write((char *)z, sizeof z);
      }
      offset += block_counts[row];
    }
    if (!snapped) {
      writes_before_evaluation = blocks_written;
      evictions_before_evaluation = eviction_total;
      warmh = hbm_cache.digest();
      warms = ssd_cache.digest();
    }
    demand(bits.size == size_t((eviction_total + 7) / 8), "program file size");
    for (size_t option_name = eviction_total; option_name < bits.size * 8; option_name++)
      demand(!((bits[option_name / 8] >> (option_name % 8)) & 1), "nonzero program padding");
    demand(blocks_written <= write_budget &&
               evaluation.writes == blocks_written - writes_before_evaluation,
           "wallet accounting");
    rows.close();
    demand(bool(rows), "rows flush");
    cout << "{\"event\":\"result\",\"status\":\"PASS\",\"schema\":\"ssd_prefix_"
            "independent_fixed_program_audit_v1\",\"budget\":"
         << write_budget << ",\"writes\":" << blocks_written
         << ",\"W_pre\":" << writes_before_evaluation
         << ",\"W_test\":" << blocks_written - writes_before_evaluation
         << ",\"processed_evictions\":" << eviction_total
         << ",\"warm_evictions\":" << evictions_before_evaluation
         << ",\"peak_hbm\":" << hbm_cache.peak << ",\"peak_ssd\":" << ssd_cache.peak
         << ",\"warm_hbm_sha256\":\"" << warmh << "\",\"warm_ssd_sha256\":\"" << warms
         << "\",\"final_hbm_sha256\":\"" << hbm_cache.digest() << "\",\"final_ssd_sha256\":\""
         << ssd_cache.digest() << "\",\"eviction_records_sha256\":\"" << eviction_hash.str()
         << "\",\"hbm_signed_event_sha256\":\"" << event_hash.str()
         << "\",\"total\":" << total.json() << ",\"evaluation\":" << evaluation.json() << "}"
         << endl;
    return 0;
  } catch (const exception &e) {
    cerr << "independent_audit: " << e.what() << endl;
    return 2;
  }
}
