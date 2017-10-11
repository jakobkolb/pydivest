"""Test for X7 experiment."""

from ..experiments import P1_compare_interaction as P1

for i in [0, 1]:
    assert P1.run_experiment(['testing', 1, i, 1, 1]) == 1
