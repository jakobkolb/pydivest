#!/bin/bash

jids1=$(sbatch S_X5.6_EQUI.job)
jid1=$(echo $jids1 | tr -dc '0-9')
echo $jid1
sleep 1
sbatch --dependency=afterok:$jid1 S_X5.6_TRANS.job
