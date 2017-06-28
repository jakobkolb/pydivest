"""testing the phase transition experiment"""

from ..experiments.X5o6_Phase_Transition import run_experiment

for i in [0, 1]:
    for j in [0, 1]:
        for k in [1, 2]:
            run_experiment(['testing', 1, i, j, k])
