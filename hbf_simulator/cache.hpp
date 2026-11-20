#pragma once

// One Cache keeps the block IDs in one tier and tracks when they were last used
// has checks whether a block is present and touch moves it to the front
// put makes room by removing the least recently used block when the cache is full
// dry_cost tries a write and then undoes it so RANCHO can price the offer

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Stop if an input or cache-state check fails.
static void require_condition(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

// A stored block ID and its neighbouring slots in LRU order.
struct Node {
  int32_t block = -1, prev = -1, next = -1;
};

// A field and its old value, used to undo a trial write.
struct Change {
  int32_t *changed_field;
  int32_t value;
};

// Cache contents and operations.
struct Cache {
  // Capacity and slot counts.
  int32_t cap;
  int32_t used = 0;
  int32_t allocated = 0;

  // Most/least recently used slots, and the first free slot.
  int32_t free_head = -1;
  int32_t head = -1;
  int32_t tail = -1;

  // Map block IDs to slots; nodes hold the IDs and their LRU links.
  std::vector<int32_t> slots;
  std::vector<Node> nodes;

  // Old field values saved while pricing a trial write.
  std::vector<Change> journal;
  bool logging = false;

  // Reserve room for the cache slots and map block IDs to them
  Cache(int32_t capacity_blocks, int32_t block_id_count)
      : cap(capacity_blocks), slots(capacity_blocks ? block_id_count : 0, -1),
        nodes(capacity_blocks) {}

  // Check whether a block is present.
  bool has(int32_t block_id) const { return cap && slots[block_id] >= 0; }

  // Move a present block to the most recently used end.
  void touch(int32_t block_id) {
    require_condition(has(block_id), "touch absent block");
    int32_t slot = slots[block_id];
    if (slot != head) {
      unlink(slot);
      push(slot);
    }
  }

  // Insert or touch; evict the tail if full. Return true only for a new insertion.
  bool put(int32_t block_id) {
    if (!cap)
      return false;
    if (has(block_id)) {
      touch(block_id);
      return false;
    }
    if (used == cap)
      erase(nodes[tail].block);
    int32_t slot;
    if (free_head >= 0) {
      slot = free_head;
      set(free_head, nodes[slot].next);
    } else {
      slot = allocated;
      require_condition(allocated < cap, "slot overflow");
      set(allocated, allocated + 1);
    }
    set(nodes[slot].block, block_id);
    set(slots[block_id], slot);
    set(used, used + 1);
    push(slot);
    return true;
  }

  // Remove a block and make its slot available again.
  void erase(int32_t block_id) {
    if (!has(block_id))
      return;
    int32_t slot = slots[block_id];
    unlink(slot);
    set(slots[block_id], -1);
    set(nodes[slot].block, -1);
    set(nodes[slot].next, free_head);
    set(free_head, slot);
    set(used, used - 1);
  }

  // Trial-write pricing.
  // Trial the suffix, count insertions, then restore the original cache state.
  int64_t dry_cost(const std::vector<int32_t> &request_blocks, size_t suffix_start) {
    require_condition(!logging && journal.empty(), "nested transaction");
    logging = true;
    int64_t blocks_written = 0;
    for (size_t position = suffix_start; position < request_blocks.size(); position++)
      blocks_written += put(request_blocks[position]);
    logging = false;
    for (auto change = journal.rbegin(); change != journal.rend(); ++change)
      *change->changed_field = change->value;
    journal.clear();
    return blocks_written;
  }

  // Saved cache order.
  // Return IDs from most to least recently used, checking the list.
  std::vector<int32_t> order() const {
    std::vector<int32_t> ordered_block_ids;
    int32_t previous_slot = -1;
    for (int32_t slot = head; slot >= 0; slot = nodes[slot].next) {
      require_condition(slot < allocated && nodes[slot].prev == previous_slot,
                        "broken LRU links");
      int32_t block_id = nodes[slot].block;
      require_condition(block_id >= 0 && has(block_id) && slots[block_id] == slot,
                        "broken LRU index");
      ordered_block_ids.push_back(block_id);
      previous_slot = slot;
      require_condition(ordered_block_ids.size() <= size_t(used), "LRU cycle");
    }
    require_condition(ordered_block_ids.size() == size_t(used) && previous_slot == tail &&
                          used <= cap,
                      "broken LRU size");
    return ordered_block_ids;
  }

  // Internal list and undo helpers.
  // Change a field, saving its old value during a trial write.
  void set(int32_t &field, int32_t new_value) {
    if (logging)
      journal.push_back({&field, field});
    field = new_value;
  }

  // Detach a slot from the LRU list.
  void unlink(int32_t slot) {
    int32_t previous_slot = nodes[slot].prev, next_slot = nodes[slot].next;
    if (previous_slot >= 0)
      set(nodes[previous_slot].next, next_slot);
    else
      set(head, next_slot);
    if (next_slot >= 0)
      set(nodes[next_slot].prev, previous_slot);
    else
      set(tail, previous_slot);
  }

  // Attach a slot at the most recently used end.
  void push(int32_t slot) {
    set(nodes[slot].prev, -1);
    set(nodes[slot].next, head);
    if (head >= 0)
      set(nodes[head].prev, slot);
    else
      set(tail, slot);
    set(head, slot);
  }
};

// Binary-file utilities.
// Load an array from a binary file and check its size.
template <class T> static std::vector<T> read_binary_array(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require_condition(bool(file), "cannot open " + path);
  auto bytes = file.tellg();
  require_condition(bytes >= 0 && bytes % sizeof(T) == 0, "unaligned input " + path);
  std::vector<T> values(size_t(bytes) / sizeof(T));
  file.seekg(0);
  if (bytes)
    require_condition(bool(file.read(reinterpret_cast<char *>(values.data()), bytes)),
                      "short input " + path);
  return values;
}

// Save an array in the simulator's binary format.
template <class T>
static void write_binary_array(const fs::path &path, const std::vector<T> &values) {
  std::ofstream file(path, std::ios::binary);
  require_condition(bool(file), "cannot create " + path.string());
  file.write(reinterpret_cast<const char *>(values.data()),
             std::streamsize(values.size() * sizeof(T)));
  file.close();
  require_condition(bool(file), "output failed " + path.string());
}
