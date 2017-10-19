"""Test for X7 experiment."""

from ..experiments import X1_equilibrium_return_rates as X1

for i in [0, 1]:
    for j in [0, 1]:
        assert X1.run_experiment(['testing', 1, i, j]) == 1
