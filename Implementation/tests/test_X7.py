"""Test for X7 experiment."""
import sys
import os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import X7_compare_campaign_in_FFH_to_AV as X7

for i in [0, 1]:
    for j in [0, 1]:
        for k in [0, 1]:
            assert X7.run_experiment(['testing', 1, i, j, k]) == 1
