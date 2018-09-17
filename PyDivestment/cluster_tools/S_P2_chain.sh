#!/usr/bin/env bash

jid1=$(sbatch S_P2_micro.job)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } S_P2_pp_micro.job

jid1=$(sbatch S_P2_aggregate_macro.job)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } S_P2_pp_aggregate_macro.job
