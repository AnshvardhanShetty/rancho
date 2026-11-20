// Build and check the index of reads that count as reuse in the same session
// credit builds the index from prepared requests
// rescore uses it to count eligible hits in completed replay rows

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
using I = int64_t;
using U = uint32_t;
using Q = uint64_t;
namespace fs = std::filesystem;

void ck(bool x, const std::string &s) {
  if (!x)
    throw std::runtime_error(s);
}

struct File {
  int fd;
  size_t bytes;
  void *p = nullptr;
  struct stat before{};

  explicit File(const fs::path &path) {
    fd = open(path.c_str(), O_RDONLY);
    ck(fd >= 0 && fstat(fd, &before) == 0 && S_ISREG(before.st_mode), "input " + path.string());
    bytes = size_t(before.st_size);
    if (bytes) {
      p = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
      ck(p != MAP_FAILED, "mmap");
    }
  }

  ~File() {
    if (bytes)
      munmap(p, bytes);
    close(fd);
  }

  template <class T> const T *as() const {
    ck(bytes % sizeof(T) == 0, "alignment");
    return static_cast<const T *>(p);
  }

  template <class T> size_t n() const { return bytes / sizeof(T); }

  void stable() const {
    struct stat s{};
    ck(fstat(fd, &s) == 0 && s.st_ino == before.st_ino && s.st_size == before.st_size &&
           s.st_mtime == before.st_mtime,
       "input changed");
  }
};

template <class T> void write(const fs::path &p, const std::vector<T> &v) {
  ck(!fs::exists(p), "existing output");
  std::ofstream f(p, std::ios::binary);
  ck(bool(f), "output");
  f.write(reinterpret_cast<const char *>(v.data()), std::streamsize(v.size() * sizeof(T)));
  f.close();
  ck(bool(f), "flush");
}

void report(const fs::path &out, const std::string &s) {
  ck(!fs::exists(out), "existing report");
  std::ofstream f(out);
  f << s << '\n';
  f.close();
  ck(bool(f), "report flush");
  std::cout << s << '\n';
}

void fresh(const fs::path &p) {
  ck(!fs::exists(p) && fs::create_directories(p), "fresh output directory");
}

bool bit(const Q *p, Q x) { return (p[x / 64] >> (x % 64)) & 1; }

void set(std::vector<Q> &v, Q x) { v[size_t(x / 64)] |= Q(1) << (x % 64); }

void savebits(const fs::path &out, const std::string &name, const std::vector<Q> &v) {
  std::vector<Q> rank(v.size() + 1, 0);
  for (size_t i = 0; i < v.size(); i++)
    rank[i + 1] = rank[i] + Q(__builtin_popcountll(v[i]));
  write(out / (name + ".bits"), v);
  write(out / (name + ".rank"), rank);
}

struct Rank {
  File bits, rank;
  const Q *b, *a;
  Q positions;

  Rank(const fs::path &p, const std::string &name, Q n)
      : bits(p / (name + ".bits")), rank(p / (name + ".rank")), b(bits.as<Q>()),
        a(rank.as<Q>()), positions(n) {
    ck(bits.n<Q>() == (n + 63) / 64 && rank.n<Q>() == bits.n<Q>() + 1 && a[0] == 0,
       "rank extent");
  }

  Q before(Q x) const {
    ck(x <= positions, "rank bound");
    Q w = x / 64, t = x % 64;
    return a[w] + (t ? Q(__builtin_popcountll(b[w] & ((Q(1) << t) - 1))) : 0);
  }

  Q count(Q start, Q n) const {
    ck(start <= positions && n <= positions - start, "rank range");
    return before(start + n) - before(start);
  }
};

void credit(int argc, char **argv) {
  ck(argc == 5, "credit PREFIX ID_COUNT OUT");
  std::string stream_prefix = argv[2];
  I block_id_count = std::stoll(argv[3]);
  fs::path out = argv[4];
  ck(block_id_count > 0 && block_id_count <= INT32_MAX, "ID range");
  File nb(stream_prefix + "_nb.i32"), ids(stream_prefix + "_ids.i32"),
      sessions(stream_prefix + "_sess.i32"), ts(stream_prefix + "_ts.i64");
  auto block_counts = nb.as<int32_t>(), block_ids = ids.as<int32_t>(),
       session_codes = sessions.as<int32_t>();
  auto times = ts.as<I>();
  size_t request_count = nb.n<int32_t>();
  ck(request_count <= INT32_MAX && sessions.n<int32_t>() == request_count &&
         ts.n<I>() == request_count,
     "row shape");
  std::vector<int32_t> last_session_by_block(size_t(block_id_count), -1),
      last_seen_request_by_block(size_t(block_id_count), -1);
  std::vector<Q> bits((ids.n<int32_t>() + 63) / 64, 0), offset(request_count + 1, 0);
  Q total = 0;
  for (size_t request_index = 0; request_index < request_count; request_index++) {
    ck(block_counts[request_index] >= 0 && session_codes[request_index] >= 0 &&
           (request_index == 0 || times[request_index] >= times[request_index - 1]),
       "row metadata");
    offset[request_index + 1] = offset[request_index] + Q(block_counts[request_index]);
    ck(offset[request_index + 1] <= ids.n<int32_t>(), "ID extent");
    for (Q k = offset[request_index]; k < offset[request_index + 1]; k++) {
      int block_id = block_ids[k];
      ck(block_id >= 0 && block_id < block_id_count, "raw block");
      if (last_seen_request_by_block[size_t(block_id)] != int32_t(request_index)) {
        if (last_session_by_block[size_t(block_id)] == session_codes[request_index]) {
          set(bits, k);
          total++;
        }
        last_seen_request_by_block[size_t(block_id)] = int32_t(request_index);
        last_session_by_block[size_t(block_id)] = session_codes[request_index];
      }
    }
  }
  ck(offset.back() == ids.n<int32_t>(), "ID total");
  for (File *f : {&nb, &ids, &sessions, &ts})
    f->stable();
  fresh(out);
  write(out / "offsets.i64", offset);
  savebits(out, "credit", bits);
  report(out / "RESULT.json", "{\"status\":\"PASS_EXACT_RAW_SESSION_CREDIT\",\"rows\":" +
                                  std::to_string(request_count) +
                                  ",\"positions\":" + std::to_string(offset.back()) +
                                  ",\"qualified_positions\":" + std::to_string(total) + "}");
}

struct Calendar {
  File meta, rows, mo, miss, oo, offers;
  const I *m, *rr, *off, *of;
  const U *mi, *oi;
  I R, N;

  explicit Calendar(const fs::path &p)
      : meta(p / "meta.i64"), rows(p / "rows.i64"), mo(p / "miss_offsets.i64"),
        miss(p / "misses.u32"), oo(p / "offer_offsets.i64"), offers(p / "offer_ids.u32") {
    ck(meta.n<I>() == 14, "calendar schema");
    m = meta.as<I>();
    ck(m[0] == 0x5353445052465831LL && m[1] == 1, "calendar magic");
    N = m[2];
    R = m[3];
    ck(N > 0 && N <= INT32_MAX && R >= 0 && R <= INT32_MAX, "calendar geometry");
    ck(rows.n<I>() == size_t(R) * 6 && mo.n<I>() == size_t(R + 1) &&
           oo.n<I>() == size_t(R + 1) && miss.n<U>() == size_t(m[7]) * 2 &&
           offers.n<U>() == size_t(m[8]),
       "calendar shapes");
    rr = rows.as<I>();
    off = mo.as<I>();
    of = oo.as<I>();
    mi = miss.as<U>();
    oi = offers.as<U>();
    ck(off[0] == 0 && off[R] == m[7] && of[0] == 0 && of[R] == m[8], "calendar cursors");
  }

  void stable() const {
    for (const File *f : {&meta, &rows, &mo, &miss, &oo, &offers})
      f->stable();
  }
};

void verify_credit(int argc, char **argv) {
  ck(argc == 5, "verify_credit PREFIX ID_COUNT CREDIT");
  std::string stream_prefix = argv[2];
  I block_id_count = std::stoll(argv[3]);
  fs::path credit_directory = argv[4];
  ck(block_id_count > 0 && block_id_count <= INT32_MAX, "ID range");
  File nb(stream_prefix + "_nb.i32"), ids(stream_prefix + "_ids.i32"),
      sessions(stream_prefix + "_sess.i32");
  auto block_counts = nb.as<int32_t>(), block_ids = ids.as<int32_t>(),
       session_codes = sessions.as<int32_t>();
  size_t request_count = nb.n<int32_t>();
  ck(sessions.n<int32_t>() == request_count && request_count <= INT32_MAX,
     "verification shape");
  Rank rank(credit_directory, "credit", ids.n<int32_t>());
  std::vector<int32_t> previous_request_by_block(size_t(block_id_count), -1);
  Q offset = 0, total = 0;
  for (size_t request_index = 0; request_index < request_count; request_index++) {
    ck(block_counts[request_index] >= 0 &&
           offset + Q(block_counts[request_index]) <= ids.n<int32_t>(),
       "verification row");
    Q count = 0;
    for (int32_t j = 0; j < block_counts[request_index]; j++) {
      int block_id = block_ids[offset + Q(j)];
      ck(block_id >= 0 && block_id < block_id_count, "verification ID");
      int previous = previous_request_by_block[size_t(block_id)];
      bool is_session_repeat = previous >= 0 && previous < int(request_index) &&
                               session_codes[size_t(previous)] == session_codes[request_index];
      ck(is_session_repeat == bit(rank.b, offset + Q(j)),
         "credit predicate differs from independent previous-row implementation");
      previous_request_by_block[size_t(block_id)] = int32_t(request_index);
      count += Q(is_session_repeat);
    }
    ck(rank.count(offset, Q(block_counts[request_index])) == count, "rank row count");
    total += count;
    offset += Q(block_counts[request_index]);
  }
  ck(offset == ids.n<int32_t>() && rank.before(offset) == total, "verification complete");
  for (File *f : {&nb, &ids, &sessions})
    f->stable();
  std::cout << "{\"status\":\"PASS_INDEPENDENT_PREVIOUS_ROW_CREDIT\",\"rows\":" << request_count
            << ",\"positions\":" << offset << ",\"qualified_positions\":" << total << "}\n";
}

void misses(int argc, char **argv) {
  ck(argc == 5, "misses CAL CREDIT OUT");
  Calendar c(argv[2]);
  fs::path cp = argv[3], out = argv[4];
  File offsets(cp / "offsets.i64");
  auto os = offsets.as<Q>();
  ck(offsets.n<Q>() >= size_t(c.R + 1), "offsets");
  Q total = os[offsets.n<Q>() - 1];
  Rank credit(cp, "credit", total);
  std::vector<Q> all((total + 63) / 64, 0), qualified(all.size(), 0);
  Q count = 0, credited = 0;
  for (I r = 0; r < c.R; r++) {
    ck(os[r + 1] - os[r] == Q(c.rr[r * 6 + 1]), "calendar raw row length");
    ck(c.off[r] >= 0 && c.off[r] <= c.off[r + 1] && c.off[r + 1] <= c.m[7], "miss cursors");
    I previous = -1;
    for (I k = c.off[r]; k < c.off[r + 1]; k++) {
      U b = c.mi[2 * k], j = c.mi[2 * k + 1];
      ck(b < U(c.N) && I(j) > previous && I(j) < c.rr[r * 6 + 3] && c.rr[r * 6 + 4] == 0,
         "miss position");
      previous = j;
      Q pos = os[r] + j;
      set(all, pos);
      count++;
      if (bit(credit.b, pos)) {
        set(qualified, pos);
        credited++;
      }
    }
  }
  c.stable();
  offsets.stable();
  fresh(out);
  savebits(out, "all_misses", all);
  savebits(out, "credited_misses", qualified);
  report(out / "RESULT.json",
         "{\"status\":\"PASS_EXACT_CALENDAR_CREDIT\",\"misses\":" + std::to_string(count) +
             ",\"credited_misses\":" + std::to_string(credited) + "}");
}

void rescore(int argc, char **argv) {
  ck(argc == 7, "rescore CREDIT GEOMETRY CAL ROWS OUT");
  fs::path credit_directory = argv[2], geometry_directory = argv[3], out = argv[6];
  Calendar calendar(argv[4]);
  File offsets(credit_directory / "offsets.i64"), rows(argv[5]);
  auto request_offsets = offsets.as<Q>();
  ck(offsets.n<Q>() >= size_t(calendar.R + 1) && rows.n<I>() % 13 == 0, "rescore shape");
  Q total = request_offsets[offsets.n<Q>() - 1];
  Rank credit(credit_directory, "credit", total), miss(geometry_directory, "all_misses", total),
      cmiss(geometry_directory, "credited_misses", total);
  auto saved = rows.as<I>();
  Q count = rows.n<I>() / 13, total_requested_blocks = 0, total_hbm_hits = 0, combined = 0,
    ssd = 0, raw_hbm_prefix_positions = 0, raw_combined_prefix_positions = 0,
    raw_ssd_positions = 0, writes = 0;
  I previous = -1;
  std::vector<U> newrows;
  newrows.reserve(size_t(count) * 3);
  for (Q k = 0; k < count; k++) {
    auto saved_row = saved + k * 13;
    I request_index = saved_row[0];
    ck(request_index >= 0 && request_index < calendar.R && request_index > previous,
       "saved row index");
    previous = request_index;
    auto calendar_row = calendar.rr + request_index * 6;
    ck(saved_row[1] == calendar_row[0] && saved_row[1] >= calendar.m[5] &&
           saved_row[1] < calendar.m[6] && saved_row[2] == calendar_row[1] &&
           saved_row[3] == calendar_row[5] && saved_row[9] == calendar_row[4] &&
           saved_row[10] == calendar_row[1] - calendar_row[3] &&
           saved_row[12] == calendar_row[2],
       "saved row identity");
    ck(saved_row[3] >= 0 && saved_row[3] <= saved_row[4] && saved_row[4] <= calendar_row[3] &&
           saved_row[7] >= 0,
       "saved prefix");
    ck((calendar_row[4] == 0) || saved_row[4] == 0, "bypass prefix");
    Q base = request_offsets[request_index], combined_prefix = Q(saved_row[4]),
      hbm_prefix = Q(saved_row[3]);
    ck(miss.count(base, combined_prefix) == Q(saved_row[5]), "original SSD count mismatch");
    Q eligible_hbm_hits = credit.count(base, hbm_prefix),
      eligible_combined_hits = credit.count(base, combined_prefix),
      eligible_ssd_hits = cmiss.count(base, combined_prefix);
    ck(eligible_hbm_hits <= eligible_combined_hits && eligible_ssd_hits <= Q(saved_row[5]) &&
           eligible_ssd_hits <= eligible_combined_hits - eligible_hbm_hits,
       "weighted counts");
    ck(eligible_combined_hits <= UINT32_MAX, "row credit range");
    newrows.push_back(U(eligible_hbm_hits));
    newrows.push_back(U(eligible_combined_hits));
    newrows.push_back(U(eligible_ssd_hits));
    total_hbm_hits += eligible_hbm_hits;
    combined += eligible_combined_hits;
    ssd += eligible_ssd_hits;
    total_requested_blocks += Q(saved_row[2]);
    raw_hbm_prefix_positions += hbm_prefix;
    raw_combined_prefix_positions += combined_prefix;
    raw_ssd_positions += Q(saved_row[5]);
    writes += Q(saved_row[7]);
  }
  ck(count == Q(calendar.R - calendar.m[12]), "evaluation coverage");
  calendar.stable();
  rows.stable();
  offsets.stable();
  fresh(out);
  write(out / "CREDITED_ROWS.u32", newrows);
  report(
      out / "RESULT.json",
      "{\"status\":\"PASS_EXACT_SAVED_PREFIX_RESCORE\",\"requests\":" + std::to_string(count) +
          ",\"denominator\":" + std::to_string(total_requested_blocks) +
          ",\"credited_hbm\":" + std::to_string(total_hbm_hits) + ",\"credited_combined\":" +
          std::to_string(combined) + ",\"credited_ssd\":" + std::to_string(ssd) +
          ",\"original_hbm\":" + std::to_string(raw_hbm_prefix_positions) +
          ",\"original_combined\":" + std::to_string(raw_combined_prefix_positions) +
          ",\"original_ssd\":" + std::to_string(raw_ssd_positions) +
          ",\"evaluation_writes\":" + std::to_string(writes) + "}");
}

void verify_rescore(int argc, char **argv) {
  ck(argc == 7, "verify_rescore CREDIT GEOMETRY CAL ROWS RESCORE_OUT");
  fs::path credit_directory = argv[2], out = argv[6];
  Calendar calendar(argv[4]);
  File offsets(credit_directory / "offsets.i64"), rows(argv[5]),
      credited(out / "CREDITED_ROWS.u32");
  auto request_offsets = offsets.as<Q>();
  auto saved = rows.as<I>();
  auto target = credited.as<U>();
  Q count = rows.n<I>() / 13;
  Rank credit(credit_directory, "credit", request_offsets[offsets.n<Q>() - 1]);
  ck(rows.n<I>() % 13 == 0 && credited.n<U>() == count * 3, "reference row shapes");
  Q total_hbm_hits = 0, combined = 0, ssd = 0;
  for (Q k = 0; k < count; k++) {
    auto saved_row = saved + k * 13;
    I request_index = saved_row[0];
    ck(request_index >= 0 && request_index < calendar.R, "reference row");
    Q eligible_hbm_hits = 0, eligible_combined_hits = 0, eligible_ssd_hits = 0;
    for (I j = 0; j < saved_row[4]; j++) {
      bool is_session_repeat = bit(credit.b, request_offsets[request_index] + Q(j));
      eligible_hbm_hits += Q(is_session_repeat && j < saved_row[3]);
      eligible_combined_hits += Q(is_session_repeat);
    }
    for (I j = calendar.off[request_index]; j < calendar.off[request_index + 1]; j++) {
      I pos = calendar.mi[2 * j + 1];
      if (pos >= saved_row[4])
        break;
      eligible_ssd_hits += Q(bit(credit.b, request_offsets[request_index] + Q(pos)));
    }
    ck(target[k * 3] == eligible_hbm_hits && target[k * 3 + 1] == eligible_combined_hits &&
           target[k * 3 + 2] == eligible_ssd_hits,
       "literal prefix rescore differs");
    total_hbm_hits += eligible_hbm_hits;
    combined += eligible_combined_hits;
    ssd += eligible_ssd_hits;
  }
  std::cout << "{\"status\":\"PASS_INDEPENDENT_LITERAL_PREFIX_RESCORE\",\"requests\":" << count
            << ",\"credited_hbm\":" << total_hbm_hits << ",\"credited_combined\":" << combined
            << ",\"credited_ssd\":" << ssd << "}\n";
}

Q topsum(std::vector<Q> &v, Q k) {
  k = std::min(k, Q(v.size()));
  if (k && k < v.size())
    std::nth_element(v.begin(), v.begin() + std::ptrdiff_t(k), v.end(), std::greater<Q>());
  Q sum = 0;
  for (Q j = 0; j < k; j++)
    sum += v[j];
  return sum;
}

std::vector<Q> grouped(std::vector<U> &ids) {
  std::sort(ids.begin(), ids.end());
  std::vector<Q> w;
  for (size_t j = 0; j < ids.size();) {
    size_t end = j + 1;
    while (end < ids.size() && ids[end] == ids[j])
      end++;
    w.push_back(Q(end - j));
    j = end;
  }
  ids.clear();
  return w;
}

void upper(int argc, char **argv) {
  ck(argc == 10, "upper CAL CREDIT BOUNDARY WARM_S PERIOD_US SORT_REFERENCE OUT EXTRA_SCHEMA");
  ck(std::string(argv[9]) == "v1", "explicit upper schema");
  Calendar c(argv[2]);
  fs::path cp = argv[3], out = argv[8];
  File offsets(cp / "offsets.i64"), boundary(argv[4]), warmfile(argv[5]);
  auto os = offsets.as<Q>();
  auto bd = boundary.as<I>();
  ck(boundary.n<I>() == 12 && bd[0] == 0x5353445741524d31LL && bd[1] == 1 && bd[6] == c.N &&
         bd[7] == c.m[4] && bd[10] == c.m[5] && bd[11] == c.m[6] && bd[5] >= 0 &&
         bd[9] >= bd[5] && bd[8] > 0,
     "upper fixed geometry");
  ck(offsets.n<Q>() >= size_t(c.R + 1) && bd[2] == c.m[12] && bd[4] == c.of[bd[2]],
     "upper cursor");
  Q capacity = Q(bd[8]), remaining = Q(bd[9] - bd[5]);
  I period = std::stoll(argv[6]);
  bool reference = std::stoi(argv[7]) != 0;
  ck(period > 0, "window period");
  Rank credit(cp, "credit", os[offsets.n<Q>() - 1]);
  std::vector<int32_t> last(size_t(c.N), -1);
  std::vector<uint8_t> warm(size_t(c.N), 0);
  for (size_t j = 0; j < warmfile.n<int32_t>(); j++) {
    int b = warmfile.as<int32_t>()[j];
    ck(b >= 0 && b < c.N && !warm[size_t(b)], "warm membership");
    warm[size_t(b)] = 1;
    last[size_t(b)] = -2;
  }
  ck(warmfile.n<int32_t>() <= capacity, "warm capacity");
  std::vector<U> global_ids, window_ids, touched;
  std::vector<Q> global, window;
  if (!reference) {
    global.resize(size_t(c.N), 0);
    window.resize(size_t(c.N), 0);
  }
  I group = 0;
  Q immortal = 0, freewarm = 0, warmonly = 0, windowfree = 0, windowbound = 0;
  std::vector<Q> cert;
  auto flush = [&]() {
    std::vector<Q> weights;
    if (reference)
      weights = grouped(window_ids);
    else {
      weights.reserve(touched.size());
      for (U b : touched) {
        weights.push_back(window[b]);
        window[b] = 0;
      }
      touched.clear();
    }
    Q top = topsum(weights, capacity);
    cert.push_back(Q(group));
    cert.push_back(windowfree);
    cert.push_back(top);
    cert.push_back(windowfree + top);
    windowbound += windowfree + top;
    windowfree = 0;
    group++;
  };
  for (I r = bd[2]; r < c.R; r++) {
    auto cr = c.rr + r * 6;
    ck(cr[0] >= c.m[5] && cr[0] < c.m[6] && os[r + 1] - os[r] == Q(cr[1]), "upper row");
    I next = (cr[0] - c.m[5]) / period;
    while (group < next)
      flush();
    bool possible = true, warmpossible = true;
    I previous = -1;
    for (I k = c.off[r]; k < c.off[r + 1]; k++) {
      U b = c.mi[2 * k], pos = c.mi[2 * k + 1];
      ck(b < U(c.N) && I(pos) > previous && I(pos) < cr[3], "upper miss");
      previous = pos;
      if (last[b] == -1)
        possible = false;
      if (!warm[b])
        warmpossible = false;
      bool q = bit(credit.b, os[r] + pos);
      if (warmpossible && q)
        warmonly++;
      if (!possible || !q)
        continue;
      immortal++;
      if (warm[b])
        freewarm++;
      if (reference) {
        if (!warm[b])
          global_ids.push_back(b);
      } else
        global[b]++;
      if (last[b] == group)
        windowfree++;
      else {
        ck(last[b] < group, "future offer");
        if (reference)
          window_ids.push_back(b);
        else {
          if (window[b]++ == 0)
            touched.push_back(b);
        }
      }
    }
    for (I k = c.of[r]; k < c.of[r + 1]; k++) {
      U b = c.oi[k];
      ck(b < U(c.N), "upper offer");
      last[b] = int32_t(group);
    }
  }
  I windows = (c.m[6] - c.m[5] + period - 1) / period;
  while (group < windows)
    flush();
  std::vector<Q> values;
  if (reference)
    values = grouped(global_ids);
  else {
    for (size_t b = 0; b < global.size(); b++)
      if (!warm[b] && global[b])
        values.push_back(global[b]);
  }
  Q top = topsum(values, remaining), budgetbound = freewarm + top,
    Ubound = std::min(immortal, std::min(budgetbound, windowbound));
  if (remaining == 0)
    Ubound = std::min(Ubound, warmonly);
  ck(Ubound <= immortal, "upper count");
  c.stable();
  offsets.stable();
  boundary.stable();
  warmfile.stable();
  fresh(out);
  write(out / "WINDOW_CERTIFICATE.u64", cert);
  report(out / "RESULT.json",
         "{\"status\":\"PASS_INTEGER_SESSION_UPPER_ACCOUNTING\",\"sorted_reference\":" +
             (reference ? std::string("true") : std::string("false")) + ",\"period_us\":" +
             std::to_string(period) + ",\"remaining\":" + std::to_string(remaining) +
             ",\"capacity\":" + std::to_string(capacity) + ",\"immortal\":" +
             std::to_string(immortal) + ",\"free_warm\":" + std::to_string(freewarm) +
             ",\"top_remaining\":" + std::to_string(top) + ",\"budget_upper\":" +
             std::to_string(budgetbound) + ",\"window_upper\":" + std::to_string(windowbound) +
             ",\"zero_write_warm_prefix\":" + std::to_string(warmonly) +
             ",\"upper\":" + std::to_string(Ubound) + "}");
}

int main(int argc, char **argv) {
  try {
    ck(argc > 1, "command required");
    std::string mode = argv[1];
    if (mode == "credit")
      credit(argc, argv);
    else if (mode == "verify_credit")
      verify_credit(argc, argv);
    else if (mode == "misses")
      misses(argc, argv);
    else if (mode == "rescore")
      rescore(argc, argv);
    else if (mode == "verify_rescore")
      verify_rescore(argc, argv);
    else if (mode == "upper")
      upper(argc, argv);
    else
      ck(false, "unknown command");
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 2;
  }
}
