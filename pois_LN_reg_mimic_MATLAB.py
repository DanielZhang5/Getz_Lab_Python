import math
import numpy as np
import matplotlib.pyplot as plt
import torch as tc
from torch.distributions import StudentT

# for the user to decide device number
device_number = 0
if tc.cuda.is_available():
    tc.cuda.set_device(device_number)

tc.set_default_dtype(tc.float64)

# wrapper: ch, chfield and basechange


class X:
    def __init__(self, x):
        self.x = x
        self.reidx = tc.ones_like(x)
        self.ncat = self.reidx.max().int().item()
        self.sums = self.x.sum()  # tc.bincount(self.reidx, self.x)[1:]


class P:
    def __init__(self, mutations: X, t_mh_nr_iter=100, taua=None, taub=None):
        self.t_mh_nr_iter = t_mh_nr_iter
        self.mutau_ihsf = 0
        self.taua = tc.ones(mutations.ncat + 1, 1) if taua is None else taua
        self.taub = tc.full([mutations.ncat, 1], 10.) if taub is None else taub


class Y:
    def __init__(self):
        pass


class MuTauSampler2(X):
    def __init__(self, x, p: P = None):
        X.__init__(self, x)
        self.p = P(self) if p is None else p

