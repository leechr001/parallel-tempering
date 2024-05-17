import numpy as np
import math

def is_k_Design(A, c, silent=False):
    if not silent:
        print("Checking design... (true if eigenspace integrated)")
    a_size = np.sum(A)
    total = 0
    for b in c.e_basis:
        total += 1
        integrate = True
        for v in c.e_basis[b]:
            t = np.dot(A,v)
            t = t / a_size
            if abs(t - np.mean(v)) > 0.001:
                total -= 1
                integrate = False
                break
        if not silent:
                print('    ', b, integrate)
    if not silent:
        print('Eigenspaces integrated: ', total)
    return total


def indicator(W, n=3):
    i_W = np.zeros(2**n)
    for w in W:
        i_W[w] = 1
    return i_W


def deg(v, cube):
    I = np.diag([1-v_i for v_i in v])
    return np.sum(I @ cube.A @ v)


def bin_array(num, dim):
    return np.array(list(np.binary_repr(num).zfill(dim))).astype(np.int8)


def F(A, c, ceil=True):
    f = 0
    if ceil:
        last = math.ceil(c.n/2) * -2/c.n
    else:
        last = math.floor(c.n/2) * -2/c.n

    # eigen stuff stored as {lambda1 : v1, v2, ...  }
    # loop eval
    for b in c.e_basis:
        # loop associated vectors
        if b != 0 and b != last:
            for v in c.e_basis[b]:
                f += np.dot(A, v)**2

    return f

