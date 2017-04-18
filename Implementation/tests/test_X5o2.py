"""Test for X7 experiment."""

import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import X5o2_Dirty_Equilibrium as X5o2


for i in [0, 1]:
    for j in [0, 1]:
        for k in [0, 1]:
            assert X5o2.run_experiment(['testing', 1, i, j, k]) == 1
