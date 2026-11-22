#pragma once

// Each Cache holds one tier's block IDs and their order of use.
//
// contains_block checks for a block. mark_block_used moves it to the front.
// store_block inserts a block and removes the least recently used block if full.
// measure_suffix_write_cost tries the same inserts, counts the writes and undoes them.

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
struct CacheSlot {
  int32_t block_id = -1, previous_slot = -1, next_slot = -1;
};

// A field and its old value, used to undo a trial write.
struct TrialWriteChange {
  int32_t *changed_field;
  int32_t old_value;
};

// Cache contents and operations.
struct Cache {
  // Capacity and slot counts.
  int32_t capacity_in_blocks;
  int32_t stored_block_count = 0;
  int32_t allocated_slot_count = 0;

  // Most/least recently used slots, and the first free slot.
  int32_t first_free_slot = -1;
  int32_t most_recent_slot = -1;
  int32_t least_recent_slot = -1;

  // Map block IDs to slots; nodes hold the IDs and their LRU links.
  std::vector<int32_t> slot_by_block_id;
  std::vector<CacheSlot> slots;

  // Old field values saved while pricing a trial write.
  std::vector<TrialWriteChange> trial_write_changes;
  bool recording_trial_write = false;

  // Reserve room for the cache slots and map block IDs to them
  Cache(int32_t capacity_blocks, int32_t block_id_count)
      : capacity_in_blocks(capacity_blocks),
        slot_by_block_id(capacity_blocks ? block_id_count : 0, -1), slots(capacity_blocks) {}

  // Check whether a block is present.
  bool contains_block(int32_t block_id) const {
    return capacity_in_blocks && slot_by_block_id[block_id] >= 0;
  }

  // Move a present block to the most recently used end.
  void mark_block_used(int32_t block_id) {
    require_condition(contains_block(block_id), "touch absent block");
    int32_t slot = slot_by_block_id[block_id];
    if (slot != most_recent_slot) {
      detach_from_use_order(slot);
      move_to_most_recent(slot);
    }
  }

  // Insert or touch; evict the tail if full. Return true only for a new
  // insertion.
  bool store_block(int32_t block_id) {
    if (!capacity_in_blocks)
      return false;
    if (contains_block(block_id)) {
      mark_block_used(block_id);
      return false;
    }
    if (stored_block_count == capacity_in_blocks)
      remove_block(slots[least_recent_slot].block_id);
    int32_t slot;
    if (first_free_slot >= 0) {
      slot = first_free_slot;
      set_and_record_for_undo(first_free_slot, slots[slot].next_slot);
    } else {
      slot = allocated_slot_count;
      require_condition(allocated_slot_count < capacity_in_blocks, "slot overflow");
      set_and_record_for_undo(allocated_slot_count, allocated_slot_count + 1);
    }
    set_and_record_for_undo(slots[slot].block_id, block_id);
    set_and_record_for_undo(slot_by_block_id[block_id], slot);
    set_and_record_for_undo(stored_block_count, stored_block_count + 1);
    move_to_most_recent(slot);
    return true;
  }

  // Remove a block and make its slot available again.
  void remove_block(int32_t block_id) {
    if (!contains_block(block_id))
      return;
    int32_t slot = slot_by_block_id[block_id];
    detach_from_use_order(slot);
    set_and_record_for_undo(slot_by_block_id[block_id], -1);
    set_and_record_for_undo(slots[slot].block_id, -1);
    set_and_record_for_undo(slots[slot].next_slot, first_free_slot);
    set_and_record_for_undo(first_free_slot, slot);
    set_and_record_for_undo(stored_block_count, stored_block_count - 1);
  }

  // Trial-write pricing.
  // Trial the suffix, count insertions, then restore the original cache state.
  int64_t measure_suffix_write_cost(const std::vector<int32_t> &request_blocks,
                                    size_t suffix_start) {
    require_condition(!recording_trial_write && trial_write_changes.empty(),
                      "nested transaction");
    recording_trial_write = true;
    int64_t blocks_written = 0;
    for (size_t position = suffix_start; position < request_blocks.size(); position++)
      blocks_written += store_block(request_blocks[position]);
    recording_trial_write = false;
    for (auto change = trial_write_changes.rbegin(); change != trial_write_changes.rend();
         ++change)
      *change->changed_field = change->old_value;
    trial_write_changes.clear();
    return blocks_written;
  }

  // Saved cache order.
  // Return IDs from most to least recently used, checking the list.
  std::vector<int32_t> blocks_most_recent_first() const {
    std::vector<int32_t> ordered_block_ids;
    int32_t previous_slot = -1;
    for (int32_t slot = most_recent_slot; slot >= 0; slot = slots[slot].next_slot) {
      require_condition(slot < allocated_slot_count && slots[slot].previous_slot == previous_slot,
                        "broken LRU links");
      int32_t block_id = slots[slot].block_id;
      require_condition(block_id >= 0 && contains_block(block_id) &&
                            slot_by_block_id[block_id] == slot,
                        "broken LRU index");
      ordered_block_ids.push_back(block_id);
      previous_slot = slot;
      require_condition(ordered_block_ids.size() <= size_t(stored_block_count), "LRU cycle");
    }
    require_condition(ordered_block_ids.size() == size_t(stored_block_count) &&
                          previous_slot == least_recent_slot &&
                          stored_block_count <= capacity_in_blocks,
                      "broken LRU size");
    return ordered_block_ids;
  }

  // Internal list and undo helpers.
  // Change a field, saving its old value during a trial write.
  void set_and_record_for_undo(int32_t &field, int32_t new_value) {
    if (recording_trial_write)
      trial_write_changes.push_back({&field, field});
    field = new_value;
  }

  // Detach a slot from the LRU list.
  void detach_from_use_order(int32_t slot) {
    int32_t previous_slot = slots[slot].previous_slot, next_slot = slots[slot].next_slot;
    if (previous_slot >= 0)
      set_and_record_for_undo(slots[previous_slot].next_slot, next_slot);
    else
      set_and_record_for_undo(most_recent_slot, next_slot);
    if (next_slot >= 0)
      set_and_record_for_undo(slots[next_slot].previous_slot, previous_slot);
    else
      set_and_record_for_undo(least_recent_slot, previous_slot);
  }

  // Attach a slot at the most recently used end.
  void move_to_most_recent(int32_t slot) {
    set_and_record_for_undo(slots[slot].previous_slot, -1);
    set_and_record_for_undo(slots[slot].next_slot, most_recent_slot);
    if (most_recent_slot >= 0)
      set_and_record_for_undo(slots[most_recent_slot].previous_slot, slot);
    else
      set_and_record_for_undo(least_recent_slot, slot);
    set_and_record_for_undo(most_recent_slot, slot);
  }
};

// Binary-file utilities.
// Load an array from a binary file and check its size.
template <class Element> static std::vector<Element> read_binary_array(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  require_condition(bool(file), "cannot open " + path);
  auto byte_count = file.tellg();
  require_condition(byte_count >= 0 && byte_count % sizeof(Element) == 0,
                    "unaligned input " + path);
  std::vector<Element> values(size_t(byte_count) / sizeof(Element));
  file.seekg(0);
  if (byte_count)
    require_condition(bool(file.read(reinterpret_cast<char *>(values.data()), byte_count)),
                      "short input " + path);
  return values;
}

// Save an array in the simulator's binary format.
template <class Element>
static void write_binary_array(const fs::path &path, const std::vector<Element> &values) {
  std::ofstream file(path, std::ios::binary);
  require_condition(bool(file), "cannot create " + path.string());
  file.write(reinterpret_cast<const char *>(values.data()),
             std::streamsize(values.size() * sizeof(Element)));
  file.close();
  require_condition(bool(file), "output failed " + path.string());
}
