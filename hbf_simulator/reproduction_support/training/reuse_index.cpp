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
constexpr int64_t H = 3600000000LL;

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
    std::string mode = argv[1], stream_prefix = argv[2], output_directory = argv[5];
    int64_t block_id_count = std::stoll(argv[3]), end_us = std::stoll(argv[4]);
    require_condition(block_id_count > 0 && block_id_count < INT32_MAX && end_us == 168 * H,
                      "geometry");
    std::vector<int64_t> arrival_times_us;
    {
      std::ifstream arrival_times_file(stream_prefix + "_ts.i64", std::ios::binary);
      int64_t arrival_us;
      while (read_record(arrival_times_file, arrival_us)) {
        if (arrival_us >= end_us)
          break;
        arrival_times_us.push_back(arrival_us);
      }
    }
    require_condition(arrival_times_us.size() < INT32_MAX, "row overflow");
    // Build: scan for eligible reads, then save per-block request and hour indexes.
    if (mode == "build") {
      std::vector<uint32_t> counts(block_id_count);
      std::vector<int32_t> stamp(block_id_count, -1), mark(block_id_count, -1), row;
      auto scan = [&](auto callback) {
        std::fill(stamp.begin(), stamp.end(), -1);
        std::fill(mark.begin(), mark.end(), -1);
        std::ifstream block_count_file(stream_prefix + "_nb.i32", std::ios::binary),
            session_file(stream_prefix + "_sess.i32", std::ios::binary),
            block_id_file(stream_prefix + "_ids.i32", std::ios::binary);
        require_condition(block_count_file && session_file && block_id_file, "raw open");
        for (size_t request_index = 0; request_index < arrival_times_us.size();
             request_index++) {
          int32_t request_block_count, session_code;
          require_condition(read_record(block_count_file, request_block_count) &&
                                read_record(session_file, session_code) &&
                                request_block_count >= 0,
                            "metadata");
          row.resize(request_block_count);
          block_id_file.read((char *)row.data(), int64_t(request_block_count) * 4);
          require_condition(block_id_file.gcount() == int64_t(request_block_count) * 4,
                            "raw IDs");
          for (int block_id : row) {
            require_condition(block_id >= 0 && block_id < block_id_count, "ID");
            if (mark[block_id] == int32_t(request_index))
              continue;
            mark[block_id] = request_index;
            bool is_session_repeat = stamp[block_id] == session_code;
            stamp[block_id] = session_code;
            if (arrival_times_us[request_index] >= 0 && is_session_repeat)
              callback(block_id, request_index);
          }
        }
      };
      scan([&](int block_id, size_t) {
        require_condition(counts[block_id] < UINT32_MAX, "credit overflow");
        counts[block_id]++;
      });
      Map<uint64_t> event_offsets(output_directory + "/offsets.u64", block_id_count + 1, true);
      event_offsets[0] = 0;
      for (int64_t block_id = 0; block_id < block_id_count; block_id++)
        event_offsets[block_id + 1] = event_offsets[block_id] + counts[block_id];
      int64_t total = event_offsets[block_id_count];
      require_condition(total > 0, "no events");
      Map<int32_t> events(output_directory + "/events.i32", total, true);
      std::fill(counts.begin(), counts.end(), 0);
      scan([&](int block_id, size_t request_index) {
        events[event_offsets[block_id] + counts[block_id]++] = request_index;
      });
      Map<uint64_t> hour_offsets(output_directory + "/hour_offsets.u64", block_id_count + 1,
                                 true);
      hour_offsets[0] = 0;
      for (int64_t block_id = 0; block_id < block_id_count; block_id++) {
        require_condition(event_offsets[block_id] + counts[block_id] ==
                              event_offsets[block_id + 1],
                          "fill count");
        int64_t last = -1, nh = 0;
        for (uint64_t k = event_offsets[block_id]; k < event_offsets[block_id + 1]; k++) {
          int64_t h = arrival_times_us[events[k]] / H;
          require_condition(h >= 0 && h < 168, "event hour");
          if (h != last) {
            nh++;
            last = h;
          }
        }
        hour_offsets[block_id + 1] = hour_offsets[block_id] + nh;
      }
      Map<Hour> hour_totals(output_directory + "/hour_entries.bin",
                            hour_offsets[block_id_count], true);
      for (int64_t block_id = 0; block_id < block_id_count; block_id++) {
        uint64_t q = hour_offsets[block_id];
        for (uint64_t k = event_offsets[block_id]; k < event_offsets[block_id + 1];) {
          int64_t hour = arrival_times_us[events[k]] / H;
          uint64_t j = k + 1;
          while (j < event_offsets[block_id + 1] && arrival_times_us[events[j]] / H == hour)
            j++;
          require_condition(j - k <= UINT32_MAX, "hour overflow");
          hour_totals[q++] = {uint32_t(hour), uint32_t(j - k)};
          k = j;
        }
        require_condition(q == hour_offsets[block_id + 1], "hour fill");
      }
      std::cout << "{\"status\":\"BUILT_ELIGIBLE_EVENT_INDEX\",\"events\":" << total
                << ",\"hour_entries\":" << hour_offsets[block_id_count]
                << ",\"rows\":" << arrival_times_us.size() << "}\n";
    } else {
      // Audit: independently scan requests and check every saved event and hour count.
      require_condition(mode == "audit", "mode");
      Map<uint64_t> event_offsets(output_directory + "/offsets.u64", block_id_count + 1, false),
          hour_offsets(output_directory + "/hour_offsets.u64", block_id_count + 1, false);
      Map<int32_t> events(output_directory + "/events.i32", event_offsets[block_id_count],
                          false);
      Map<Hour> hour_totals(output_directory + "/hour_entries.bin",
                            hour_offsets[block_id_count], false);
      std::vector<uint32_t> cursor(block_id_count);
      std::vector<int32_t> prev(block_id_count, -1), row;
      int64_t checked = 0;
      std::ifstream block_count_file(stream_prefix + "_nb.i32", std::ios::binary),
          session_file(stream_prefix + "_sess.i32", std::ios::binary),
          block_id_file(stream_prefix + "_ids.i32", std::ios::binary);
      require_condition(block_count_file && session_file && block_id_file, "audit input");
      for (size_t request_index = 0; request_index < arrival_times_us.size(); request_index++) {
        int32_t request_block_count, session_code;
        require_condition(read_record(block_count_file, request_block_count) &&
                              read_record(session_file, session_code) &&
                              request_block_count >= 0,
                          "audit metadata");
        row.resize(request_block_count);
        block_id_file.read((char *)row.data(), int64_t(request_block_count) * 4);
        require_condition(block_id_file.gcount() == int64_t(request_block_count) * 4,
                          "audit raw");
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
        for (int block_id : row) {
          require_condition(block_id >= 0 && block_id < block_id_count, "auditID");
          if (arrival_times_us[request_index] >= 0 && prev[block_id] == session_code) {
            require_condition(event_offsets[block_id] + cursor[block_id] <
                                      event_offsets[block_id + 1] &&
                                  events[event_offsets[block_id] + cursor[block_id]] ==
                                      int64_t(request_index),
                              "event identity mismatch");
            cursor[block_id]++;
            checked++;
          }
          prev[block_id] = session_code;
        }
      }
      for (int64_t block_id = 0; block_id < block_id_count; block_id++) {
        require_condition(event_offsets[block_id] + cursor[block_id] ==
                              event_offsets[block_id + 1],
                          "extra event");
        uint64_t at = event_offsets[block_id];
        int last = -1;
        for (uint64_t j = hour_offsets[block_id]; j < hour_offsets[block_id + 1]; j++) {
          auto hour_entry = hour_totals[j];
          require_condition(hour_entry.h < 168 && int64_t(hour_entry.h) > last &&
                                hour_entry.count > 0,
                            "hour order");
          for (uint32_t k = 0; k < hour_entry.count; k++) {
            require_condition(at < event_offsets[block_id + 1] &&
                                  arrival_times_us[events[at]] / H == hour_entry.h,
                              "hour count mismatch");
            at++;
          }
          last = hour_entry.h;
        }
        require_condition(at == event_offsets[block_id + 1], "unaccounted hour");
      }
      require_condition(checked == int64_t(event_offsets[block_id_count]), "total mismatch");
      std::cout << "{\"status\":\"PASS_INDEPENDENT_ALL_EVENT_AND_HOUR_AUDIT\","
                   "\"events\":"
                << checked << ",\"rows\":" << arrival_times_us.size() << "}\n";
    }
    return 0;
  } catch (const std::exception &is_session_repeat) {
    std::cerr << is_session_repeat.what() << '\n';
    return 1;
  }
}
