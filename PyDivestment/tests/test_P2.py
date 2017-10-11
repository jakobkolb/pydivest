"""Test for X7 experiment."""

from ..experiments import P2_macro_trajectory as P2

for mode in [1]:
    for approximate in [1, 2, 3]:
        assert P2.run_experiment(['testing', 1, mode, approximate]) == 1
