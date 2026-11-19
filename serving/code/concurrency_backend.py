"""Combine the serving schedulers with per-step request and cache counters."""

from concurrency_observer import Observer
from direct_backend import DirectScheduler
from direct_backend_hbf import HBFScheduler


class BaselineScheduler(Observer, DirectScheduler):
    pass


class TieredScheduler(Observer, HBFScheduler):
    pass
