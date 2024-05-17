from multiprocessing import Pool
from RandomWalker import RW
from Graphs import CubeGraph
import random as r
import math
import numpy as np


def HillClimb(n, d, x_size, max_t=10000, ceil=True, max_var=0.05):
    chart = []
    cube = CubeGraph(n)

    W = []
    for i in d:
        w = RW(x_size=x_size, lam=i, c=cube, ceil=ceil)
        W.append(w)

    t = 1
    chart.append([rw.f for rw in W])
    while t < max_t:
        # with Pool(processes=coupled) as pool:
        #    W = pool.map(metropolisStep, W)

        for i in range(len(d)):
            metropolisStep(W[i], batch_size=10)

        temp = r.sample(W, k=2)
        parallelTemperStep(temp[0], temp[1])

        chart.append([rw.f for rw in W])
        t += 1
        print("\r" + "    iter: " + str(t), end='')
        for rw in W:
            if rw.f == 0:
                return chart, rw, t

    return chart, W[0], t


def metropolisStep(rw, batch_size=1):
    while rw.f > 0 and batch_size > 0:
        Y, Y_, deg_y, f_y = rw.proposeStep()

        threshold = min(rw.deg / deg_y * math.e ** (-rw.lam * (f_y - rw.f)), 1)

        # 4 decide to step
        if r.random() < threshold:
            rw.setState(Y, Y_, deg_y, f_y)

        batch_size = batch_size - 1

    return rw


def parallelTemperStep(rw1, rw2):
    X, X_, deg_x, f_x = rw1.getState()
    Y, Y_, deg_y, f_y = rw2.getState()

    threshold = min(math.e**((-rw1.lam - rw2.lam) * (f_x-f_y)), 1)
    if r.random() < threshold:
        rw2.setState(X, X_, deg_x, f_x)
        rw1.setState(Y, Y_, deg_y, f_y)
        return rw1, rw2
    else:
        return rw1, rw2
