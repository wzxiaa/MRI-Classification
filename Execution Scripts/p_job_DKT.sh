#!/bin/bash

dataset="i"
let lower=51
let upper=53

for outer in 0 1 2 3 4
do
qsub -q all.q -pe all.pe 8 -l h_vmem=5G -V -N DKT_Outer_"$dataset"_"$outer"_"$lower"_"$upper"<<EOF
python /data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/code/parallel_DKT.py "$dataset" "$outer" "$lower" "$upper"
EOF
done
