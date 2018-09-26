#!/usr/bin/env bash

jid1=$(sbatch S_P3o2_micro.job)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } S_P3o2_PP_micro.job

jid1=$(sbatch S_P3o2_aggregate_macro.job)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } S_P3o2_PP_aggregate_macro.job

jid1=$(sbatch S_P3o2_representative_macro.job)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } S_P3o2_PP_representative_macro.job
