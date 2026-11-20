// Record the request and hour of each eligible repeat read
// A block counts once per request if its last reference was in the same session
// Requests before time zero only set the previous session
// Build mode saves the index and audit mode independently checks its events

#include <algorithm>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
using I = int64_t;
constexpr I H = 3600000000LL;

void require_condition(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}

template <class T> bool read_record(std::ifstream &f, T &x) {
  f.read((char *)&x, sizeof x);
  if (!f.gcount())
    return false;
  require_condition(f.gcount() == sizeof x, "partial record");
  return true;
}

// Access a binary array on disk without copying the whole file into memory.
template <class T> struct Map {
  T *p;
  size_t n;
  int fd;
  bool write;

  Map(std::string s, size_t count, bool w) : n(count), write(w) {
    fd = open(s.c_str(), w ? O_RDWR | O_CREAT | O_EXCL : O_RDONLY, 0600);
    require_condition(fd >= 0, "map open");
    if (w)
      require_condition(ftruncate(fd, n * sizeof(T)) == 0, "truncate");
    struct stat st;
    fstat(fd, &st);
    require_condition(size_t(st.st_size) == n * sizeof(T) && n > 0, "map size");
    p = (T *)mmap(0, n * sizeof(T), w ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
    require_condition(p != MAP_FAILED, "mmap");
  }

  ~Map() {
    if (write)
      msync(p, n * sizeof(T), MS_SYNC);
    munmap(p, n * sizeof(T));
    close(fd);
  }

  T &operator[](size_t i) {
    require_condition(i < n, "index bounds");
    return p[i];
  }
};

// One block's eligible-read count for one hour.
struct Hour {
  uint32_t h, count;
};

static_assert(sizeof(Hour) == 8);

// The label builder uses this index to find later reads of each offered block.
int main(int argc, char **argv) {
  try {
    require_condition(argc == 6, "mode prefix N end outdir");
    std::string mode = argv[1], pre = argv[2], out = argv[5];
    I N = std::stoll(argv[3]), end = std::stoll(argv[4]);
    require_condition(N > 0 && N < INT32_MAX && end == 168 * H, "geometry");
    std::vector<I> times;
    {
      std::ifstream f(pre + "_ts.i64", std::ios::binary);
      I t;
      while (read_record(f, t)) {
        if (t >= end)
          break;
        times.push_back(t);
      }
    }
    require_condition(times.size() < INT32_MAX, "row overflow");
    // Build: scan for eligible reads, then save per-block request and hour indexes.
    if (mode == "build") {
      std::vector<uint32_t> counts(N);
      std::vector<int32_t> stamp(N, -1), mark(N, -1), row;
      auto scan = [&](auto callback) {
        std::fill(stamp.begin(), stamp.end(), -1);
        std::fill(mark.begin(), mark.end(), -1);
        std::ifstream ns(pre + "_nb.i32", std::ios::binary),
            ss(pre + "_sess.i32", std::ios::binary), ids(pre + "_ids.i32", std::ios::binary);
        require_condition(ns && ss && ids, "raw open");
        for (size_t r = 0; r < times.size(); r++) {
          int32_t n, s;
          require_condition(read_record(ns, n) && read_record(ss, s) && n >= 0, "metadata");
          row.resize(n);
          ids.read((char *)row.data(), I(n) * 4);
          require_condition(ids.gcount() == I(n) * 4, "raw IDs");
          for (int b : row) {
            require_condition(b >= 0 && b < N, "ID");
            if (mark[b] == int32_t(r))
              continue;
            mark[b] = r;
            bool e = stamp[b] == s;
            stamp[b] = s;
            if (times[r] >= 0 && e)
              callback(b, r);
          }
        }
      };
      scan([&](int b, size_t) {
        require_condition(counts[b] < UINT32_MAX, "credit overflow");
        counts[b]++;
      });
      Map<uint64_t> off(out + "/offsets.u64", N + 1, true);
      off[0] = 0;
      for (I b = 0; b < N; b++)
        off[b + 1] = off[b] + counts[b];
      I total = off[N];
      require_condition(total > 0, "no events");
      Map<int32_t> events(out + "/events.i32", total, true);
      std::fill(counts.begin(), counts.end(), 0);
      scan([&](int b, size_t r) { events[off[b] + counts[b]++] = r; });
      Map<uint64_t> ho(out + "/hour_offsets.u64", N + 1, true);
      ho[0] = 0;
      for (I b = 0; b < N; b++) {
        require_condition(off[b] + counts[b] == off[b + 1], "fill count");
        I last = -1, nh = 0;
        for (uint64_t k = off[b]; k < off[b + 1]; k++) {
          I h = times[events[k]] / H;
          require_condition(h >= 0 && h < 168, "event hour");
          if (h != last) {
            nh++;
            last = h;
          }
        }
        ho[b + 1] = ho[b] + nh;
      }
      Map<Hour> hs(out + "/hour_entries.bin", ho[N], true);
      for (I b = 0; b < N; b++) {
        uint64_t q = ho[b];
        for (uint64_t k = off[b]; k < off[b + 1];) {
          I hour = times[events[k]] / H;
          uint64_t j = k + 1;
          while (j < off[b + 1] && times[events[j]] / H == hour)
            j++;
          require_condition(j - k <= UINT32_MAX, "hour overflow");
          hs[q++] = {uint32_t(hour), uint32_t(j - k)};
          k = j;
        }
        require_condition(q == ho[b + 1], "hour fill");
      }
      std::cout << "{\"status\":\"BUILT_ELIGIBLE_EVENT_INDEX\",\"events\":" << total
                << ",\"hour_entries\":" << ho[N] << ",\"rows\":" << times.size() << "}\n";
    } else {
      // Audit: independently scan requests and check every saved event and hour count.
      require_condition(mode == "audit", "mode");
      Map<uint64_t> off(out + "/offsets.u64", N + 1, false),
          ho(out + "/hour_offsets.u64", N + 1, false);
      Map<int32_t> ev(out + "/events.i32", off[N], false);
      Map<Hour> hs(out + "/hour_entries.bin", ho[N], false);
      std::vector<uint32_t> cursor(N);
      std::vector<int32_t> prev(N, -1), row;
      I checked = 0;
      std::ifstream ns(pre + "_nb.i32", std::ios::binary),
          ss(pre + "_sess.i32", std::ios::binary), ids(pre + "_ids.i32", std::ios::binary);
      require_condition(ns && ss && ids, "audit input");
      for (size_t r = 0; r < times.size(); r++) {
        int32_t n, s;
        require_condition(read_record(ns, n) && read_record(ss, s) && n >= 0, "audit metadata");
        row.resize(n);
        ids.read((char *)row.data(), I(n) * 4);
        require_condition(ids.gcount() == I(n) * 4, "audit raw");
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        for (int b : row) {
          require_condition(b >= 0 && b < N, "auditID");
          if (times[r] >= 0 && prev[b] == s) {
            require_condition(off[b] + cursor[b] < off[b + 1] && ev[off[b] + cursor[b]] == I(r),
                              "event identity mismatch");
            cursor[b]++;
            checked++;
          }
          prev[b] = s;
        }
      }
      for (I b = 0; b < N; b++) {
        require_condition(off[b] + cursor[b] == off[b + 1], "extra event");
        uint64_t at = off[b];
        int last = -1;
        for (uint64_t j = ho[b]; j < ho[b + 1]; j++) {
          auto z = hs[j];
          require_condition(z.h < 168 && I(z.h) > last && z.count > 0, "hour order");
          for (uint32_t k = 0; k < z.count; k++) {
            require_condition(at < off[b + 1] && times[ev[at]] / H == z.h,
                              "hour count mismatch");
            at++;
          }
          last = z.h;
        }
        require_condition(at == off[b + 1], "unaccounted hour");
      }
      require_condition(checked == I(off[N]), "total mismatch");
      std::cout << "{\"status\":\"PASS_INDEPENDENT_ALL_EVENT_AND_HOUR_AUDIT\","
                   "\"events\":"
                << checked << ",\"rows\":" << times.size() << "}\n";
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
