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
