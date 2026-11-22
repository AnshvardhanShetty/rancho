// Replay with HBM alone and record how many leading blocks it can read.
// The simulator uses these prefix lengths to filter its reuse history.

#include "simulation_io.hpp"

int main(int argument_count, char **arguments) {
  try {
    // 1. Read the file paths, HBM capacity, and evaluation start/end times.
    require_condition(argument_count == 7,
                      "trace prefix, block ID count, HBM capacity, start time, end time, output");
    const TraceArrayPaths trace_files(arguments[1]);
    int block_id_count = std::stoi(arguments[2]), hbm_capacity = std::stoi(arguments[3]);
    int64_t evaluation_start_us = std::stoll(arguments[4]),
            evaluation_end_us = std::stoll(arguments[5]);

    // 2. Load request lengths and arrival times, and create an empty HBM cache.
    auto block_counts = read_binary_array<int>(trace_files.block_counts);
    auto arrival_times_us = read_binary_array<int64_t>(trace_files.arrival_times_us);
    require_condition(block_counts.size() == arrival_times_us.size() &&
                          std::is_sorted(arrival_times_us.begin(), arrival_times_us.end()),
                      "time and size");
    Cache hbm(hbm_capacity, block_id_count);
    bool evaluation_started = false;

    // 3. Open the block-ID file. Reserve one output value per request.
    std::ifstream block_id_stream(trace_files.block_ids, std::ios::binary);
    std::vector<int> request_block_ids;
    std::vector<int32_t> hbm_only_prefix_lengths(block_counts.size(), 0);

    // 4. Read each request in arrival order, stopping at the evaluation end.
    for (size_t request_index = 0; request_index < block_counts.size() &&
                                   arrival_times_us[request_index] < evaluation_end_us;
         ++request_index) {
      int block_count = block_counts[request_index];
      require_condition(block_count >= 0, "length");
      request_block_ids.resize(block_count);
      block_id_stream.read(reinterpret_cast<char *>(request_block_ids.data()),
                           sizeof(int) * request_block_ids.size());
      require_condition(block_id_stream.gcount() ==
                            int64_t(sizeof(int) * request_block_ids.size()),
                        "short row");
      if (arrival_times_us[request_index] < 0)
        continue;

      // Start the measured period with empty caches, matching the RANCHO
      // replay.
      if (!evaluation_started && arrival_times_us[request_index] >= evaluation_start_us) {
        hbm = Cache(hbm_capacity, block_id_count);
        evaluation_started = true;
      }

      // Read from the start until the first miss. Then insert/touch the suffix
      // to represent processing the rest of the request in HBM alone.
      int readable_prefix_length = 0;
      for (; readable_prefix_length < block_count &&
             hbm.contains_block(request_block_ids[readable_prefix_length]);
           ++readable_prefix_length)
        hbm.mark_block_used(request_block_ids[readable_prefix_length]);
      for (int block_position = readable_prefix_length; block_position < block_count;
           ++block_position)
        hbm.store_block(request_block_ids[block_position]);

      hbm_only_prefix_lengths[request_index] = readable_prefix_length;
    }

    // 5. Save one readable prefix length per request.
    write_binary_array(arguments[6], hbm_only_prefix_lengths);
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
