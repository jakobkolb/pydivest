#!/bin/bash

jids1=$(sbatch S_X7.0_ffh_EQUI.job)
jid1=$(echo $jids1 | tr -dc '0-9')
echo $jid1
sleep 1
sbatch --dependency=afterok:$jid1 S_X7.0_ffh_TRANS.job

sleep 1

jids1=$(sbatch S_X7.0_imi_EQUI.job)
jid1=$(echo $jids1 | tr -dc '0-9')
echo $jid1
sleep 1
sbatch --dependency=afterok:$jid1 S_X7.0_imi_TRANS.job