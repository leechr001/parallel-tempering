import numpy as np
import random as r
from Helpers import *


class RW:
    def __init__(self, x_size, lam, c, ceil=True):
        self.lam = lam
        self.c = c
        self.X = r.sample(range(2**c.n), k=x_size)
        self.X_ = indicator(self.X, c.n)
        self.deg = deg(self.X_, c)
        self.f = F(self.X_, c, ceil)
        self.size = x_size
        self.ceil = ceil
        A = c.A
        D_inv = 1/c.n * np.identity(2**c.n)
        self.L = np.matmul(D_inv, A)

    def setState(self, Y, Y_, deg_y, f_y):
        self.X = Y
        self.X_ = Y_
        self.deg = deg_y
        self.f = f_y

    def getState(self):
        return self.X, self.X_, self.deg, self.f

    def proposeStep(self):
        v_num = r.randint(0,self.size - 1)

        v = np.zeros(2**self.c.n)
        v[self.X[v_num]] = 1

        # 2 picking candidate step
        q = v @ self.L
        Y_ = self.X_ - v
        y = r.choices(range(2**self.c.n), weights=q)

        while Y_[y] == 1:
            v_num = r.randint(0,self.size - 1)
            v = np.zeros(2**self.c.n)
            v[self.X[v_num]] = 1
            q = v @ self.L
            Y_ = self.X_ - v
            y = r.choices(range(2**self.c.n), weights=q)

        Y_[y] = 1

        # 3 compute f_y
        f_y = F(Y_, self.c, self.ceil)
        deg_y = deg(Y_, self.c)

        Y = self.X.copy()
        Y[v_num] = y

        return Y, Y_, deg_y, f_y
