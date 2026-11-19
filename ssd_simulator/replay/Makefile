CXX ?= c++
PYTHON ?= python3
OPENSSL_CFLAGS ?= $(shell pkg-config --cflags openssl)
OPENSSL_LIBS ?= $(shell pkg-config --libs openssl)
CXXFLAGS ?= -O2 -std=c++17 -Wno-deprecated-declarations
BUILD := ../../build/ssd

.PHONY: all test
all: $(BUILD)/ssd_replay $(BUILD)/check_replay

$(BUILD)/ssd_replay: ssd_replay.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(OPENSSL_CFLAGS) $< $(OPENSSL_LIBS) -o $@

$(BUILD)/check_replay: check_replay.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(OPENSSL_CFLAGS) $< $(OPENSSL_LIBS) -o $@

test: all
	$(PYTHON) test_replay.py --build $(BUILD)

# Full recorded SSD policy, in addition to the smaller baseline replay.
$(BUILD)/recorded_ssd_replay: ../algorithm/replay.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(OPENSSL_CFLAGS) $< $(OPENSSL_LIBS) -o $@

.PHONY: recorded
recorded: $(BUILD)/recorded_ssd_replay $(BUILD)/check_replay

$(BUILD)/label_ssd_replay: ../algorithm/support/label_replay.cpp ../algorithm/support/marginal_labels.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(OPENSSL_CFLAGS) $< $(OPENSSL_LIBS) -o $@

$(BUILD)/session_accounting: ../algorithm/support/session_accounting.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: recorded-support
recorded-support: recorded $(BUILD)/label_ssd_replay $(BUILD)/session_accounting
