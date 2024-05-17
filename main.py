from HillClimbAlg import *
from Graphs import CubeGraph
import matplotlib.pyplot as plt


def run(m=1000, n=7, d=(0.1, 0.2), w=16, floor=True):
    chart, design, t = HillClimb(n, d, w, max_t=m)
    print()
    #print("Design:", design.X)
    #print("Energy:", design.f)
    #print("t: ", t)

    plt.title("Objective vs iterations")
    plt.plot(chart)
    plt.show()


def main():
    test = [(0.05,0.15)]
    for d in test:
        print("d: " + str(d), sep='')
        for _ in range(11):
            run(d=d)
        print()


if __name__ == "__main__":
    main()