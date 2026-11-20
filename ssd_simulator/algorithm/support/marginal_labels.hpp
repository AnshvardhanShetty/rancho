// Measure how much extra of the prompt one saved block can unlock
// Keep the reference HBM state and the writes to every other SSD block
// Attribute a block's later gain to its most recent sampled offer

struct Marginal {
  Map<int64_t> samples;
  Map<uint64_t> bits, ranks;
  vector<int32_t> owner, last_accounted;
  ofstream events;
  size_t next;
  uint64_t tests = 0;

  Marginal(int n, string data, string credit, string output, int64_t first)
      : samples(data), bits(credit + "/credit.bits"),
        ranks(credit + "/credit.rank"), owner(n, -1),
        last_accounted(samples.n, -1), events(output + ".gain", ios::binary),
        next(lower_bound(samples.p, samples.p + samples.n, first) - samples.p) {
  }

  uint64_t count_eligible_reads_before(size_t x) const {
    const size_t word = x / 64, tail = x % 64;
    return ranks[word] +
           (tail
                ? __builtin_popcountll(bits[word] & ((uint64_t(1) << tail) - 1))
                : 0);
  }

  void record_eviction_offer(int b, int64_t ordinal) {
    owner[b] = -1;
    if (next < samples.n && samples[next] == ordinal)
      owner[b] = int(next++);
  }

  void count_marginal_read_savings(const Cache &hbm_cache,
                                   const Cache &ssd_cache,
                                   const Map<int32_t> &block_ids,
                                   size_t request_offset, int lookup_limit,
                                   int reference_prefix, int request_index) {
    auto accessible = [&](int block_id) {
      return hbm_cache.contains_block(block_id) ||
             ssd_cache.contains_block(block_id);
    };
    // Recompute a sample of the contributions by explicitly forcing one
    // identity present and absent. Duplicate positions carry one identity.
    auto record = [&](int block_id, int start, int prefix_end) {
      int sample_index = owner[block_id];
      if (sample_index < 0 || last_accounted[sample_index] == request_index)
        return;
      last_accounted[sample_index] = request_index;
      uint64_t gain = count_eligible_reads_before(request_offset + prefix_end) -
                      count_eligible_reads_before(request_offset + start);
      if ((request_index % 257) == 0) {
        int prefix_with_block = 0, prefix_without_block = 0;
        while (prefix_with_block < lookup_limit &&
               (block_ids[request_offset + prefix_with_block] == block_id ||
                accessible(block_ids[request_offset + prefix_with_block])))
          ++prefix_with_block;
        while (prefix_without_block < lookup_limit &&
               (block_ids[request_offset + prefix_without_block] != block_id ||
                hbm_cache.contains_block(block_id)) &&
               accessible(block_ids[request_offset + prefix_without_block]))
          ++prefix_without_block;
        require_condition(
            count_eligible_reads_before(request_offset + prefix_with_block) -
                    count_eligible_reads_before(request_offset +
                                                prefix_without_block) ==
                gain,
            "toggle verification");
        ++tests;
      }
      if (gain) {
        require_condition(gain <= INT32_MAX, "gain overflow");
        int32_t row[3] = {request_index, sample_index, int32_t(gain)};
        events.write((char *)row, sizeof(row));
      }
    };
    // Removing an SSD-dependent block breaks the reachable prefix at
    // its first occurrence. Everything through the original prefix end counts.
    for (int j = 0; j < reference_prefix; ++j) {
      int block_id = block_ids[request_offset + j];
      if (owner[block_id] >= 0 && !hbm_cache.contains_block(block_id))
        record(block_id, j, reference_prefix);
    }
    if (reference_prefix == lookup_limit)
      return;
    int block_id = block_ids[request_offset + reference_prefix];
    if (owner[block_id] < 0)
      return;
    int prefix_end = reference_prefix + 1;
    while (prefix_end < lookup_limit &&
           (block_ids[request_offset + prefix_end] == block_id ||
            accessible(block_ids[request_offset + prefix_end])))
      ++prefix_end;
    record(block_id, reference_prefix, prefix_end);
  }
};
