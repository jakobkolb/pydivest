#!/usr/bin/env bash

jid1=$(sbatch S_P1_calc.job)
echo $jid1
sleep 2
sbatch --dependency=afterok:${jid1##* } S_P1_post_processing.job

