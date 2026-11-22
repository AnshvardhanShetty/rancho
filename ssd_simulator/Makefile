.DEFAULT_GOAL := recorded-support
CXX ?= c++
PYTHON ?= python3
OPENSSL_CFLAGS ?= $(shell pkg-config --cflags openssl)
OPENSSL_LIBS ?= $(shell pkg-config --libs openssl)
CXXFLAGS ?= -O2 -std=c++17 -Wno-deprecated-declarations
BUILD := ../build/ssd

$(BUILD)/check_replay: algorithm/support/check_replay.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(OPENSSL_CFLAGS) $< $(OPENSSL_LIBS) -o $@

$(BUILD)/recorded_ssd_replay: algorithm/replay.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(OPENSSL_CFLAGS) $< $(OPENSSL_LIBS) -o $@

.PHONY: recorded
recorded: $(BUILD)/recorded_ssd_replay $(BUILD)/check_replay

$(BUILD)/label_ssd_replay: algorithm/support/label_replay.cpp algorithm/support/marginal_labels.hpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $(OPENSSL_CFLAGS) $< $(OPENSSL_LIBS) -o $@

$(BUILD)/session_accounting: algorithm/support/session_accounting.cpp
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY: recorded-support
recorded-support: recorded $(BUILD)/label_ssd_replay $(BUILD)/session_accounting
