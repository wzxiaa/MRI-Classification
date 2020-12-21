#!/bin/bash

dataset="c"
let lower=59
let upper=60
let outer=2

qsub -q all.q -pe all.pe 8 -l h_vmem=8G -V -N Outer_"$dataset"_"$outer"_"$lower"_"$upper"<<EOF
python /data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/code/parallel_LOSO.py "$dataset" "$outer" "$lower" "$upper"

