"""Check synthetic plans and SSD decisions without starting vLLM or a GPU.

Small stand-ins isolate the real eviction method; they do not validate GPU I/O.
"""

import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
from collections import OrderedDict

from serve import validate
from ssd_policy import Offer, accept_fixture, reject_fixture

HERE = Path(__file__).resolve().parents[1]


def load_backend():
    """Import the connector with stand-ins for GPU-only dependency interfaces."""
    interfaces = {
        "vllm.distributed.kv_transfer.kv_connector.v1.base": {
            "KVConnectorBase_V1": type("Base", (), {}),
            "KVConnectorMetadata": type("Metadata", (), {}),
            "KVConnectorRole": types.SimpleNamespace(SCHEDULER=0, WORKER=1),
        },
        "vllm.v1.core.sched.scheduler": {"Scheduler": type("Scheduler", (), {})},
        "vllm.v1.core.kv_cache_utils": {
            "get_block_hash": lambda value: value[0],
            "get_group_id": lambda value: value[1],
        },
        "ssd_storage": {"Storage": type("Storage", (), {})},
    }
    modules = {}
    for name, attributes in interfaces.items():
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            key = ".".join(parts[:i])
            modules.setdefault(key, types.ModuleType(key))
        vars(modules[name]).update(attributes)
    with patch.dict(sys.modules, modules):
        return importlib.import_module("ssd_backend")


class Inputs(unittest.TestCase):
    def plan(self):
        return json.loads((HERE / "examples/fixture_accept.json").read_text())

    def test_example_is_valid_and_contains_no_storage_path(self):
        plan = self.plan()
        validate(plan)
        self.assertEqual(len(plan["requests"]), 12)
        self.assertNotIn("ssd_directory", plan)

    def test_rejects_requests_that_do_not_fit(self):
        plan = self.plan()
        plan["hbm_pages"] = 1
        with self.assertRaises(ValueError):
            validate(plan)

    def test_rejects_unordered_arrivals(self):
        plan = self.plan()
        plan["requests"]["row4"]["arrival_seconds"] = 0.2
        plan["requests"]["row5"]["arrival_seconds"] = 0.1
        with self.assertRaises(ValueError):
            validate(plan)

    def test_fixture_policies_do_not_need_future_reads(self):
        offer = Offer(b"key", 0, 0, 1.0, 16, 0, 1, 1, 0, 2, 1.0, 0.0, "row0", {})
        self.assertTrue(accept_fixture({})(offer))
        self.assertFalse(reject_fixture({})(offer))


class Eviction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = load_backend()

    def make_connector(self, written=0, budget=1, already_saved=False):
        connector = self.backend.Connector.__new__(self.backend.Connector)
        connector.resident = {1: (b"key", 0.0)}
        connector.new_pages = set()
        connector.now = 2.0
        connector.history = {
            b"key": dict(evictions=0, count=2, last=1.0, rid="row0", features={})
        }
        connector.serving_settings = dict(write_budget_pages=budget)
        connector.evictions = connector.absent = connector.retained = (
            connector.rejected
        ) = 0
        connector.offers = []
        connector.decide = lambda offer: connector.offers.append(offer) or True
        store = types.SimpleNamespace(
            slots=OrderedDict(), capacity=2, writes=written, page_bytes=16
        )
        if already_saved:
            store.slots[b"key"] = 0

        def put(key, page):
            store.slots[key] = page
            store.writes += 1

        store.put = put
        self.backend.store = store
        block = types.SimpleNamespace(block_id=1, ref_cnt=0, block_hash=(b"key", 0))
        return connector, store, block

    def test_accepts_only_within_hard_write_budget(self):
        connector, store, block = self.make_connector()
        connector.evict(block)
        self.assertEqual(store.writes, 1)
        self.assertEqual(connector.offers[0].prior_touch_count, 2)
        self.assertEqual(connector.offers[0].prior_eviction_count, 0)
        connector, store, block = self.make_connector(written=1)
        connector.evict(block)
        self.assertEqual(store.writes, 1)
        self.assertEqual(len(connector.offers), 1)
        self.assertEqual(connector.rejected, 1)

    def test_existing_copy_needs_no_new_write_or_decision(self):
        connector, store, block = self.make_connector(already_saved=True)
        connector.evict(block)
        self.assertEqual(store.writes, 0)
        self.assertEqual(connector.offers, [])
        self.assertEqual(connector.retained, 1)

    def test_active_or_unfinished_pages_cannot_be_written(self):
        connector, store, block = self.make_connector()
        block.ref_cnt = 1
        with self.assertRaises(AssertionError):
            connector.evict(block)
        self.assertEqual(store.writes, 0)
        connector, store, block = self.make_connector()
        connector.new_pages.add(1)
        with self.assertRaises(RuntimeError):
            connector.evict(block)
        self.assertEqual(store.writes, 0)


if __name__ == "__main__":
    unittest.main()
