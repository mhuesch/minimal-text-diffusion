"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
# from mpi4py import MPI  # disabled for single-GPU
import torch as th

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 10 #8

SETUP_RETRY_COUNT = 3

def setup_dist():
    return

def dev():
    return "cuda" if th.cuda.is_available() else "cpu"

def get_rank():
    return 0

def get_world_size():
    return 1

def is_master():
    return True

def barrier():
    pass

def all_reduce(tensor):
    return tensor

def load_state_dict(path, **kwargs):
    # no MPI
    return th.load(path, **kwargs)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
