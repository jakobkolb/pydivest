#!/usr/bin/env bash

jid1=$(sbatch S_P3_micro.job)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } S_P3_PP_micro.job

jid1=$(sbatch S_P3_mean_macro.job)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } S_P3_PP_mean_macro.job

jid1=$(sbatch S_P3_representative_macro.job)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } S_P3_PP_representative_macro.job
