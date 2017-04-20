"""Test for X7 experiment."""
import sys
import os

from .. import X6_macro_trajectory as X6

for i in [0, 1]:
    for j in [0, 1]:
        assert X6.run_experiment(['testing', 1, i, j]) == 1
