// Count later reads for the first reuse predictor
// Record which request introduced each block and count later reads in that session
// Y.i32 stores reads by interval and c.i32 stores distinct block counts by request
// The hourly trainer only uses intervals that have already ended

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
  // 1. Read paths and interval settings, then load lengths, sessions, and times.
  const char *stream_prefix = argv[1];
  int64_t block_id_count = atoll(argv[2]) + 1;
  int64_t interval_us = atoll(argv[3]);
  int64_t interval_count = atoll(argv[4]);
  const char *read_counts_path = argv[5];
  char filename[512];
  FILE *array_file;
  snprintf(filename, sizeof filename, "%s_nb.i32", stream_prefix);
  array_file = fopen(filename, "rb");
  fseek(array_file, 0, SEEK_END);
  int64_t request_count = ftell(array_file) / 4;
  fseek(array_file, 0, SEEK_SET);
  int32_t *block_counts = malloc(request_count * 4);
  if (fread(block_counts, 4, request_count, array_file) != (size_t)request_count)
    return 2;
  fclose(array_file);
  snprintf(filename, sizeof filename, "%s_sess.i32", stream_prefix);
  array_file = fopen(filename, "rb");
  int32_t *session_codes = malloc(request_count * 4);
  if (fread(session_codes, 4, request_count, array_file) != (size_t)request_count)
    return 2;
  fclose(array_file);
  snprintf(filename, sizeof filename, "%s_ts.i64", stream_prefix);
  array_file = fopen(filename, "rb");
  int64_t *arrival_times_us = malloc(request_count * 8);
  if (fread(arrival_times_us, 8, request_count, array_file) != (size_t)request_count)
    return 2;
  fclose(array_file);
  // 2. Track the last session, introducing request, and per-request duplicate marks.
  int32_t *last_session_by_block = malloc(block_id_count * 4),
          *introducing_request_by_block = malloc(block_id_count * 4),
          *last_seen_request_by_block = malloc(block_id_count * 4);
  for (int64_t i = 0; i < block_id_count; i++) {
    last_session_by_block[i] = -1;
    introducing_request_by_block[i] = -1;
    last_seen_request_by_block[i] = -1;
  }
  int32_t *read_counts_by_interval = calloc((size_t)interval_count * request_count, 4);
  if (!read_counts_by_interval) {
    fprintf(stderr, "alloc %lld\n", (long long)((int64_t)interval_count * request_count * 4));
    return 3;
  }
  int32_t *request_blocks = malloc(4 * (1 << 21));
  int64_t request_buffer_capacity = 1 << 21;
  snprintf(filename, sizeof filename, "%s_ids.i32", stream_prefix);
  FILE *block_id_file = fopen(filename, "rb");
  int64_t total_credited_reads = 0;
  int32_t *introduced_block_counts = calloc(request_count, 4);
  // 3. Read requests in order; count each block once and assign reuse to its owner.
  for (int64_t request_index = 0; request_index < request_count; request_index++) {
    int32_t request_block_count = block_counts[request_index];
    if (request_block_count > request_buffer_capacity) {
      request_buffer_capacity = (int64_t)request_block_count * 2;
      request_blocks = realloc(request_blocks, request_buffer_capacity * 4);
    }
    if (request_block_count > 0 && fread(request_blocks, 4, request_block_count,
                                         block_id_file) != (size_t)request_block_count)
      return 4;
    int32_t session_code = session_codes[request_index];
    int64_t interval_index = arrival_times_us[request_index] / interval_us;
    if (interval_index < 0)
      interval_index = 0;
    if (interval_index >= interval_count)
      interval_index = interval_count - 1;
    for (int32_t j = 0; j < request_block_count; j++) {
      int32_t block_id = request_blocks[j];
      if (block_id < 0 || block_id >= block_id_count)
        continue;
      if (last_seen_request_by_block[block_id] == (int32_t)request_index)
        continue;
      last_seen_request_by_block[block_id] = (int32_t)request_index;
      if (last_session_by_block[block_id] != session_code) {
        last_session_by_block[block_id] = session_code;
        introducing_request_by_block[block_id] = (int32_t)request_index;
        introduced_block_counts[request_index]++;
      } else if (introducing_request_by_block[block_id] >= 0) {
        read_counts_by_interval[interval_index * request_count +
                                introducing_request_by_block[block_id]]++;
        total_credited_reads++;
      }
    }
  }
  fclose(block_id_file);
  // 4. Save read counts by interval and each request's introduced-block count.
  array_file = fopen(read_counts_path, "wb");
  fwrite(read_counts_by_interval, 4, (size_t)interval_count * request_count, array_file);
  fclose(array_file);
  printf("{\"requests\":%lld,\"buckets\":%lld,\"credits_bucketed\":%lld}\n",
         (long long)request_count, (long long)interval_count, (long long)total_credited_reads);
  if (argc > 6) {
    array_file = fopen(argv[6], "wb");
    fwrite(introduced_block_counts, 4, request_count, array_file);
    fclose(array_file);
  }
  return 0;
}
