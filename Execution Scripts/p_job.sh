#!/bin/bash

dataset="c"
let lower=2
let upper=60

for outer in 0 1 2 3 4
do
qsub -q all.q -pe all.pe 8 -l h_vmem=5G -V -N Outer_"$dataset"_"$outer"_"$lower"_"$upper"<<EOF
python /data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/code/parallel_LOSO.py "$dataset" "$outer" "$lower" "$upper"
EOF
done