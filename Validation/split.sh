#!/bin/bash

for class_sep in $(seq 4 13);
do 
    python split.py 5 "$class_sep"
done