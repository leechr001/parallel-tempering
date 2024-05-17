import numpy as np
from Helpers import bin_array


class Node:
    def __init__(self, c, a, h):
        self.c = c
        self.a = a
        self.h = h

    # vertices equal if they have the same coordinates
    def __eq__(self, other):
        return self.h == other.h

    def __hash__(self):
        return self.h

    def __repr__(self):
        return str(self.h)


class Graph:

    def __init__(self, A):
        self.n = np.array(A).shape[0]
        self.A = A
        self.lam, self.psi = np.linalg.eig(A)
        self.target_psi_c = []
        self.target_psi_f = []

        def freq(n, _):
            return np.abs(n+1)

        # sort in frequency order
        self.lam, self.psi = zip(*sorted(zip(self.lam, self.psi), key=freq))


class CubeGraph(Graph):

    # n is dim
    # G is generating group
    # V is vertices
    def __init__(self, n):
        self.n = int(n)
        self.G = list()
        self.V = list()
        self.A = np.zeros((2 ** n, 2 ** n))
        self.E = list()
        self.eval = list()
        self.e_basis = {}

        # get generating group
        for i in range(n):
            g_i = np.zeros((n,), dtype=int)
            g_i[i] = 1
            self.G.append(g_i)

        # create nodes
        for i in range(2 ** n):
            self.V.append(Node(bin_array(i, n), list(), i))

        # add adjacency
        for v in self.V:
            for g in self.G:
                # use that labels are binary of coordinants
                neighbor_index = np.dot((v.c + g) % 2, np.array([2 ** (n - i - 1) for i in range(n)]))

                v.a.append(self.V[neighbor_index])

                # undirected edge so symmetric A
                self.A[v.h, neighbor_index] = 1
                self.A[neighbor_index, v.h] = 1

        # add eigenvectors and eigenvalues
        for v in self.V:
            v_ = []
            b = v.c
            for v1 in self.V:
                v_.append((-1) ** (np.dot(b, v1.c)))
            self.E.append(v_)
            lam = -2 * np.sum(b) / n
            self.eval.append(lam)

            # 1
            # if lam != math.ceil(n/2) * -2/n:
            if lam not in self.e_basis:
                self.e_basis[lam] = [v_]
            else:
                self.e_basis[lam].append(v_)













