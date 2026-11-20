// Replay the same requests with just HBM and save the prefix it can read each time
// RANCHO uses this record to count reuse that HBM alone would have missed
// ReferenceCache repeats the calculation with a separate cache to check the result

#include "cache.hpp"
#include <list>
#include <memory>
#include <unordered_map>

// INDEPENDENT CHECK: a second cache using a different implementation.
// order keeps blocks from most to least recently used; location finds each one.
struct ReferenceCache {
  size_t capacity;
  std::list<int> order;
  std::unordered_map<int, std::list<int>::iterator> location;

  explicit ReferenceCache(size_t capacity) : capacity(capacity) {
    location.reserve(capacity + 1);
  }

  // Count the readable prefix while updating the cache for each block.
  // On a miss, insert the block and evict the least recently used if full.
  int process(const std::vector<int> &ids) {
    int readable_prefix = 0;
    for (int j = 0; j < (int)ids.size(); ++j) {
      auto it = location.find(ids[j]);
      if (it != location.end()) {
        if (j == readable_prefix)
          ++readable_prefix;
        if (it->second != order.begin())
          order.splice(order.begin(), order, it->second);
      } else if (capacity) {
        if (order.size() == capacity) {
          location.erase(order.back());
          order.pop_back();
        }
        order.push_front(ids[j]);
        location.emplace(ids[j], order.begin());
      }
    }
    return readable_prefix;
  }
};

// MAIN WALKTHROUGH: replay the requests and save one prefix length per request.
int main(int argc, char **argv) {
  try {
    // 1. Read the file paths, HBM capacity, and evaluation start/end times.
    require_condition(argc == 7, "raw prefix, N, HBM capacity, split, end, output");
    std::string stream_prefix = argv[1];
    int block_id_count = std::stoi(argv[2]), hbm_capacity = std::stoi(argv[3]);
    I evaluation_start = std::stoll(argv[4]), evaluation_end = std::stoll(argv[5]);

    // 2. Load request lengths and arrival times, and create two empty caches.
    auto block_counts = read_binary_array<int>(stream_prefix + "_nb.i32");
    auto arrival_times = read_binary_array<I>(stream_prefix + "_ts.i64");
    require_condition(block_counts.size() == arrival_times.size() &&
                          std::is_sorted(arrival_times.begin(), arrival_times.end()),
                      "time and size");
    Cache hbm(hbm_capacity, block_id_count);
    ReferenceCache reference(hbm_capacity);
    bool evaluation_started = false;
    I measured_requests = 0, total_readable = 0, total_positions = 0;

    // 3. Open the block-ID file. Reserve one output value per request.
    std::ifstream file(stream_prefix + "_ids.i32", std::ios::binary);
    std::vector<int> blocks;
    std::vector<int32_t> prefix_lengths(block_counts.size(), 0);

    // 4. Read each request in arrival order, stopping at the evaluation end.
    for (size_t request = 0;
         request < block_counts.size() && arrival_times[request] < evaluation_end; ++request) {
      int block_count = block_counts[request];
      require_condition(block_count >= 0, "length");
      blocks.resize(block_count);
      file.read(reinterpret_cast<char *>(blocks.data()), sizeof(int) * blocks.size());
      require_condition(file.gcount() == I(sizeof(int) * blocks.size()), "short row");
      if (arrival_times[request] < 0)
        continue;

      // Start the measured period with empty caches, matching the RANCHO replay.
      if (!evaluation_started && arrival_times[request] >= evaluation_start) {
        hbm = Cache(hbm_capacity, block_id_count);
        reference.order.clear();
        reference.location.clear();
        evaluation_started = true;
      }

      // Read from the start until the first miss. Then insert/touch the suffix
      // to represent processing the rest of the request in HBM alone.
      int readable_prefix = 0;
      for (; readable_prefix < block_count && hbm.has(blocks[readable_prefix]);
           ++readable_prefix)
        hbm.touch(blocks[readable_prefix]);
      for (int j = readable_prefix; j < block_count; ++j)
        hbm.put(blocks[j]);

      // Check the same request with the independent cache, then save its length.
      int reference_prefix = reference.process(blocks);
      require_condition(readable_prefix == reference_prefix,
                        "two independent HBM-only prefixes disagree");
      prefix_lengths[request] = readable_prefix;
      if (evaluation_started) {
        ++measured_requests;
        total_readable += readable_prefix;
        total_positions += block_count;
      }
    }

    // 5. Check that both caches ended with identical contents and usage order.
    auto final_order = hbm.order();
    require_condition(std::equal(final_order.begin(), final_order.end(),
                                 reference.order.begin(), reference.order.end()),
                      "final HBM-only order");

    // 6. Save the prefix lengths and print a short summary of the replay.
    write_binary_array(argv[6], prefix_lengths);
    std::cout << "{\"status\":\"PASS_TWO_CHRONOLOGICAL_HBM_PREFIXES\",\"rows\":"
              << prefix_lengths.size() << ",\"measured_requests\":" << measured_requests
              << ",\"positions\":" << total_positions
              << ",\"readable_positions\":" << total_readable
              << ",\"capacity\":" << hbm_capacity << "}\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
