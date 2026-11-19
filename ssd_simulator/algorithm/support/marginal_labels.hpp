// Marginal credited prefix length when one sampled victim alone is present versus
// absent on this reference request. A newer offer supersedes its prior owner.
// Writes to other blocks, HBM, and the legal lookup limit follow the reference.
struct Marginal {
  Map<int64_t> samples;
  Map<uint64_t> bits, ranks;
  vector<int32_t> owner, last_accounted;
  ofstream events;
  size_t next;
  uint64_t tests = 0;
  Marginal(int n, string data, string credit, string output, int64_t first)
      : samples(data), bits(credit + "/credit.bits"), ranks(credit + "/credit.rank"),
        owner(n, -1), last_accounted(samples.n, -1),
        events(output + ".gain", ios::binary),
        next(lower_bound(samples.p, samples.p + samples.n, first) - samples.p) {}
  uint64_t credited_before(size_t x) const {
    const size_t word = x / 64, tail = x % 64;
    return ranks[word] +
           (tail ? __builtin_popcountll(bits[word] & ((uint64_t(1) << tail) - 1)) : 0);
  }
  void offer(int b, int64_t ordinal) {
    owner[b] = -1;
    if (next < samples.n && samples[next] == ordinal)
      owner[b] = int(next++);
  }
  void measure(const Cache &h, const Cache &s, const Map<int32_t> &ids, size_t off,
               int cap, int prefix, int r) {
    auto accessible = [&](int b) { return h.has(b) || s.has(b); };
    // Recompute a sample of the contributions by explicitly forcing one
    // identity present and absent. Duplicate positions carry one identity.
    auto record = [&](int b, int start, int end) {
      int i = owner[b];
      if (i < 0 || last_accounted[i] == r)
        return;
      last_accounted[i] = r;
      uint64_t gain = credited_before(off + end) - credited_before(off + start);
      if ((r % 257) == 0) {
        int with = 0, without = 0;
        while (with < cap && (ids[off + with] == b || accessible(ids[off + with])))
          ++with;
        while (without < cap && (ids[off + without] != b || h.has(b)) &&
               accessible(ids[off + without]))
          ++without;
        ck(credited_before(off + with) - credited_before(off + without) == gain,
           "toggle verification");
        ++tests;
      }
      if (gain) {
        ck(gain <= INT32_MAX, "gain overflow");
        int32_t row[3] = {r, i, int32_t(gain)};
        events.write((char *)row, sizeof(row));
      }
    };
    // Removing an SSD-dependent block breaks the reachable prefix at
    // its first occurrence. Everything through the original prefix end counts.
    for (int j = 0; j < prefix; ++j) {
      int b = ids[off + j];
      if (owner[b] >= 0 && !h.has(b))
        record(b, j, prefix);
    }
    if (prefix == cap)
      return;
    int b = ids[off + prefix];
    if (owner[b] < 0)
      return;
    int end = prefix + 1;
    while (end < cap && (ids[off + end] == b || accessible(ids[off + end])))
      ++end;
    record(b, prefix, end);
  }
};
