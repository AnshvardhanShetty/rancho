#pragma once

// One Cache tracks block IDs and least-recently-used order for one tier.
// rancho.cpp creates separate flash/HBM caches and decides what to admit.
// has()/touch(): read; put()/erase(): update; dry_cost(): try and undo a write.
// order() checks/saves contents; lower helpers maintain links and read/write arrays.

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

using I = int64_t;
namespace fs = std::filesystem;

// Stop if an input or cache-state check fails.
static void require_condition(bool b, const std::string &s) {
  if (!b)
    throw std::runtime_error(s);
}

// A stored block ID and its neighbouring slots in LRU order.
struct Node {
  int32_t block = -1, prev = -1, next = -1;
};

// A field and its old value, used to undo a trial write.
struct Change {
  int32_t *p;
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

  // Basic cache operations.
  // c is cache capacity; n is the number of possible block IDs.
  Cache(int32_t c, int32_t n) : cap(c), slots(c ? n : 0, -1), nodes(c) {}

  // Check whether a block is present.
  bool has(int32_t b) const { return cap && slots[b] >= 0; }

  // Move a present block to the most recently used end.
  void touch(int32_t b) {
    require_condition(has(b), "touch absent block");
    int32_t s = slots[b];
    if (s != head) {
      unlink(s);
      push(s);
    }
  }

  // Insert or touch; evict the tail if full. Return true only for a new insertion.
  bool put(int32_t b) {
    if (!cap)
      return false;
    if (has(b)) {
      touch(b);
      return false;
    }
    if (used == cap)
      erase(nodes[tail].block);
    int32_t s;
    if (free_head >= 0) {
      s = free_head;
      set(free_head, nodes[s].next);
    } else {
      s = allocated;
      require_condition(allocated < cap, "slot overflow");
      set(allocated, allocated + 1);
    }
    set(nodes[s].block, b);
    set(slots[b], s);
    set(used, used + 1);
    push(s);
    return true;
  }

  // Remove a block and make its slot available again.
  void erase(int32_t b) {
    if (!has(b))
      return;
    int32_t s = slots[b];
    unlink(s);
    set(slots[b], -1);
    set(nodes[s].block, -1);
    set(nodes[s].next, free_head);
    set(free_head, s);
    set(used, used - 1);
  }

  // Trial-write pricing.
  // Trial the suffix, count insertions, then restore the original cache state.
  I dry_cost(const std::vector<int32_t> &row, size_t first) {
    require_condition(!logging && journal.empty(), "nested transaction");
    logging = true;
    I w = 0;
    for (size_t j = first; j < row.size(); j++)
      w += put(row[j]);
    logging = false;
    for (auto it = journal.rbegin(); it != journal.rend(); ++it)
      *it->p = it->value;
    journal.clear();
    return w;
  }

  // Saved cache order.
  // Return IDs from most to least recently used, checking the list.
  std::vector<int32_t> order() const {
    std::vector<int32_t> a;
    int32_t prev = -1;
    for (int32_t s = head; s >= 0; s = nodes[s].next) {
      require_condition(s < allocated && nodes[s].prev == prev, "broken LRU links");
      int32_t b = nodes[s].block;
      require_condition(b >= 0 && has(b) && slots[b] == s, "broken LRU index");
      a.push_back(b);
      prev = s;
      require_condition(a.size() <= size_t(used), "LRU cycle");
    }
    require_condition(a.size() == size_t(used) && prev == tail && used <= cap,
                      "broken LRU size");
    return a;
  }

  // Internal list and undo helpers.
  // Change a field, saving its old value during a trial write.
  void set(int32_t &x, int32_t v) {
    if (logging)
      journal.push_back({&x, x});
    x = v;
  }

  // Detach a slot from the LRU list.
  void unlink(int32_t s) {
    int32_t p = nodes[s].prev, n = nodes[s].next;
    if (p >= 0)
      set(nodes[p].next, n);
    else
      set(head, n);
    if (n >= 0)
      set(nodes[n].prev, p);
    else
      set(tail, p);
  }

  // Attach a slot at the most recently used end.
  void push(int32_t s) {
    set(nodes[s].prev, -1);
    set(nodes[s].next, head);
    if (head >= 0)
      set(nodes[head].prev, s);
    else
      set(tail, s);
    set(head, s);
  }
};

// Binary-file utilities.
// Load an array from a binary file and check its size.
template <class T> static std::vector<T> read_binary_array(const std::string &p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  require_condition(bool(f), "cannot open " + p);
  auto bytes = f.tellg();
  require_condition(bytes >= 0 && bytes % sizeof(T) == 0, "unaligned input " + p);
  std::vector<T> a(size_t(bytes) / sizeof(T));
  f.seekg(0);
  if (bytes)
    require_condition(bool(f.read(reinterpret_cast<char *>(a.data()), bytes)),
                      "short input " + p);
  return a;
}

// Save an array in the simulator's binary format.
template <class T>
static void write_binary_array(const fs::path &p, const std::vector<T> &a) {
  std::ofstream f(p, std::ios::binary);
  require_condition(bool(f), "cannot create " + p.string());
  f.write(reinterpret_cast<const char *>(a.data()),
          std::streamsize(a.size() * sizeof(T)));
  f.close();
  require_condition(bool(f), "output failed " + p.string());
}
