"""Test for X7 experiment."""

import sys
import os

from ..experiments.X5o2_Dirty_Equilibrium import run_experiment

for i in [0, 1]:
    for j in [0, 1]:
        for k in [0, 1]:
            assert run_experiment(['testing', 1, i, j, k]) == 1
