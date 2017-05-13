#!/bin/sh

for f in *.job; do
	echo ${f}
	sbatch ${f}
	sleep 1
done
