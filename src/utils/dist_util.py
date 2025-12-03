"""
Helpers for distributed training.
"""

import torch as th

def setup_dist():
    return

def dev():
    return "cuda" if th.cuda.is_available() else "cpu"

def load_state_dict(path, **kwargs):
    # no MPI
    return th.load(path, **kwargs)