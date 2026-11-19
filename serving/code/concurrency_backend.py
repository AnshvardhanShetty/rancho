"""Bind the same observational mixin to each unchanged tier scheduler.

Assumptions and limits are documented in concurrency_observer.py. MRO delegates
all allocation/admission operations unchanged to the original scheduler classes.
"""
from concurrency_observer import Observer
from direct_backend import DirectScheduler
from direct_backend_hbf import HBFScheduler
class BaselineScheduler(Observer,DirectScheduler):pass
class TieredScheduler(Observer,HBFScheduler):pass
