.DEFAULT_GOAL := demo
PYTHON ?= python3
.PHONY: demo demo-serving test-demo test-simulator

build/rancho: hbf_simulator/write_controller.hpp hbf_simulator/simulation_io.hpp
build/%: hbf_simulator/%.cpp hbf_simulator/cache.hpp
	@mkdir -p build
	$(CXX) -O2 -std=c++17 $< -o $@

demo: build/rancho build/hbm_only_readable_prefix
	$(PYTHON) hbf_simulator/examples/demo.py

demo-serving: build/rancho build/hbm_only_readable_prefix
	$(PYTHON) hbf_simulator/examples/serving_replay.py

test-demo: build/rancho build/hbm_only_readable_prefix
	$(PYTHON) -m unittest discover -s hbf_simulator/tests -p 'test_serving_replay.py'

test-simulator: build/rancho build/hbm_only_readable_prefix
	$(PYTHON) hbf_simulator/tests/simulator_regression.py
