# -*- coding: UTF-8 -*-

import copy
import warnings


import matplotlib.pyplot as plt
import numpy as np


class EvolutionViewer:
    def __init__(self):
        pass

    def drawSession(self,monitor,session,featureKey, ylog=False, view=False):
        generation = range(int(session.curTime+1))
        max_feature = [r[featureKey]['max'] for r in session.popRecords]
        avg_feature  = np.array([r[featureKey]['average'] for r in session.popRecords])
        stdev_feature = np.array([r[featureKey]['stdev'] for r in session.popRecords])

        plt.plot(generation, avg_feature, 'b-', label="average")
        plt.plot(generation, avg_feature - stdev_feature, 'g-.', label="-1 sd")
        plt.plot(generation, avg_feature + stdev_feature, 'g-.', label="+1 sd")
        plt.plot(generation, max_feature, 'r-', label="best")

        plt.title("Population's average and best fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="best")
        if ylog:
            plt.gca().set_yscale('symlog')

        filename = 'session'+str(session.taskxh)+"."+featureKey+".svg"
        plt.savefig(filename)
        if view:
            plt.show()

        plt.close()
