# MRI-Classification

Design, Implementation, and Validation of a Multi-stepMachine learning Classification Procedure for MRI-based Prediction.

## 1. Environment and Package Requirements

- python 3.7+
- [NumPy](https://numpy.org/install/)
- [SciPy](https://www.scipy.org/install.html)
- [joblib](https://joblib.readthedocs.io/en/latest/installing.html)
- [pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
- [Matplotlib](https://matplotlib.org/users/installing.html)
- [Scikit-learn](https://scikit-learn.org/stable/install.html)
- [MRMRe](https://github.com/dontgoto/mrmre-tutorial)
- [Similarity fusion network](https://github.com/rmarkello/snfpy)

- [Seaborn](https://seaborn.pydata.org/installing.html)
- [neurocombat_sklearn](https://github.com/Warvito/neurocombat_sklearn)

## 2. Directory Structure

### 2.1 Set Up

The `initialize.sh` file is used to set up all the necessary directories for storing the results from the analysis. Below is the snapshot of it:

```bash
#!/bin/bash

mkdir -p ./base_results/Xs/results/fold{0..4}
mkdir -p ./base_results/Xi/results/fold{0..4}
mkdir -p ./base_results/Xc/results/fold{0..4}

mkdir -p ./612_results/Xs/results/fold{0..4}
mkdir -p ./612_results/Xi/results/fold{0..4}
mkdir -p ./612_results/Xc/results/fold{0..4}

mkdir -p ./DKT_results/Xs/results/fold{0..4}
mkdir -p ./DKT_results/Xi/results/fold{0..4}
mkdir -p ./DKT_results/Xc/results/fold{0..4}

mkdir -p ./612_REPMES_DKT_results/Xs/results/fold{0..4}
mkdir -p ./612_REPMES_DKT_results/Xi/results/fold{0..4}
mkdir -p ./612_REPMES_DKT_results/Xc/results/fold{0..4}
```

After running the `initialize.sh` script, `base_results` , `612_results`, `DKT_results` and `612_REPMES_DKT_results` will be created. 

```bash
mkdir -p ./612_results/Xs/results/fold{0..4} // this indicates there are 5 outer folds and 
																						// change to fold{0..9} if 10 outer folds are 																							// needed
```



This repository consists mainly of implemented code files. The raw data can be found on the BIC server under the following directory:

/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI

```
.
├── 612_REPMES_DKT_results
│   ├── Xc
│   │   └── results
│   │       ├── fold0
│   │       ├── fold1
│   │       ├── fold2
│   │       ├── fold3
│   │       └── fold4
│   ├── Xi
│   │   └── results
│   │       ├── fold0
│   │       ├── fold1
│   │       ├── fold2
│   │       ├── fold3
│   │       └── fold4
│   └── Xs
│       └── results
│           ├── fold0
│           ├── fold1
│           ├── fold2
│           ├── fold3
│           └── fold4
├── 612_results
│   ├── Xc
│   │   └── results
│   │       ├── fold0
│   │       ├── fold1
│   │       ├── fold2
│   │       ├── fold3
│   │       └── fold4
│   ├── Xi
│   │   └── results
│   │       ├── fold0
│   │       ├── fold1
│   │       ├── fold2
│   │       ├── fold3
│   │       └── fold4
│   └── Xs
│       └── results
│           ├── fold0
│           ├── fold1
│           ├── fold2
│           ├── fold3
│           └── fold4
├── DKT_results
│   ├── Xc
│   │   └── results
│   │       ├── fold0
│   │       ├── fold1
│   │       ├── fold2
│   │       ├── fold3
│   │       └── fold4
│   ├── Xi
│   │   └── results
│   │       ├── fold0
│   │       ├── fold1
│   │       ├── fold2
│   │       ├── fold3
│   │       └── fold4
│   └── Xs
│       └── results
│           ├── fold0
│           ├── fold1
│           ├── fold2
│           ├── fold3
│           └── fold4
├── base_results
│   ├── Xc
│   │   └── results
│   │       ├── fold0
│   │       ├── fold1
│   │       ├── fold2
│   │       ├── fold3
│   │       └── fold4
│   ├── Xi
│   │   └── results
│   │       ├── fold0
│   │       ├── fold1
│   │       ├── fold2
│   │       ├── fold3
│   │       └── fold4
│   └── Xs
│       └── results
│           ├── fold0
│           ├── fold1
│           ├── fold2
│           ├── fold3
│           └── fold4
├── code
│   ├── p_job_DKT.sh
│   ├── p_job_LOSO.sh
│   ├── p_job_LOSO_DKT.sh
│   ├── p_job_base.sh
│   ├── parallel.py
│   ├── parallel_DKT.py
│   ├── parallel_LOSO.py
│   ├── parallel_LOSO_DKT.py
│   └── split.py
├── data
│   ├── Site.csv
│   ├── Xc.csv
│   ├── Xi.csv
│   ├── Xs.csv
│   └── Y.csv
├── data612
│   ├── Site.csv
│   ├── Subjects.csv
│   ├── Xc.csv
│   ├── Xi.csv
│   ├── Xs.csv
│   └── Y.csv
├── dataREPMES_DKT
│   ├── Site.csv
│   ├── Subjects.csv
│   ├── Xc.csv
│   ├── Xi.csv
│   ├── Xs.csv
│   └── Y.csv
├── data_DKT
│   ├── Site.csv
│   ├── Xc.csv
│   ├── Xi.csv
│   ├── Xs.csv
│   └── Y.csv
├── initialize.sh
├── split
│   ├── 10outerfolds
│   │   ├── outer_test_index.pickle
│   │   └── outer_train_index.pickle
│   ├── 5outerfolds
│   │   ├── outer_test_index.pickle
│   │   └── outer_train_index.pickle
│   ├── outer_test_index.pickle
│   └── outer_train_index.pickle
└── split_matlab
    ├── inner_test_index.pickle
    ├── inner_train_index.pickle
    ├── outer_test_index.pickle
    ├── outer_train_index.pickle
    ├── outerloop_test
    │   ├── test-0.csv
    │   ├── test-1.csv
    │   ├── test-2.csv
    │   ├── test-3.csv
    │   └── test-4.csv
    ├── outerloop_train
    │   ├── subjListLO_ACO2020-0.txt
    │   ├── subjListLO_ACO2020-1.txt
    │   ├── subjListLO_ACO2020-2.txt
    │   ├── subjListLO_ACO2020-3.txt
    │   └── subjListLO_ACO2020-4.txt
    ├── subjects_scans_index.pickle
    └── subjects_scans_index_DKT.pickle
```

---

## 3. Data Sets

### 3.1 Data Sets

The dataset are placed under different folders:

```
├── data
│   ├── Site.csv
│   ├── Xc.csv
│   ├── Xi.csv
│   ├── Xs.csv
│   └── Y.csv
├── data612
│   ├── Site.csv
│   ├── Subjects.csv
│   ├── Xc.csv
│   ├── Xi.csv
│   ├── Xs.csv
│   └── Y.csv
├── dataREPMES_DKT
│   ├── Site.csv
│   ├── Subjects.csv
│   ├── Xc.csv
│   ├── Xi.csv
│   ├── Xs.csv
│   └── Y.csv
├── data_DKT
│   ├── Site.csv
│   ├── Xc.csv
│   ├── Xi.csv
│   ├── Xs.csv
│   └── Y.csv
```

### 3.2 Data Splitting for 5 or 10 Outerfolds

The data splitting files for the data indexes of the splitting is stored under `split/10outerfolds` and `split/5outerfolds`. The actual splliting data is stored as `pickle` files. 

### 3.3 Data Splitting for Leave One Subject Out Splitting

The data splitting files for the data indexes of the splitting is stored under `split/split_matlab` and `split/split_matlab`. The actual splliting data is stored as `pickle` files. 

---

## 4. Code

All of the codes are placed under `/code` directory:

```
├── code
│   ├── p_job_DKT.sh
│   ├── p_job_LOSO.sh
│   ├── p_job_LOSO_DKT.sh
│   ├── p_job_base.sh
│   ├── parallel.py
│   ├── parallel_DKT.py
│   ├── parallel_LOSO.py
│   ├── parallel_LOSO_DKT.py
│   └── split.py
```



### 4.1 Inner Loop Python Code

- `parallel.py` is used for the base dataset (302). `p_job_base.sh` is used for inner loop execution.

- `-parallel_DKT.py` is used for the DKT dataset (302). `p_job_DKT.sh` is used for inner loop execution.

- `parallel_LOSO.py` is used for the LOSO dataset (612). `p_job_LOSO.sh` is used for inner loop execution.

- `parallel_LOSO_DKT.py` is used for the LOSO DKT dataset (612). `p_job_LOSO_DKT.sh` is used for inner loop execution.

The arguments are being specified when the p_job.sh is being called. However, depending on how many inner folds we are using, the inner_fold_cv variable needs to be specified inside the script’s main function. Together with the n_jobs variable which specified how many subprocesses we are using to run the grid search in parallel.

### 4.2 Executing the Scripts

```bash
#!/bin/bash

dataset="i"
let lower=2
let upper=5

for outer in 0 1 2 3 4
do
qsub -q all.q -pe all.pe 8 -l h_vmem=5G -V -N LOSO_Outer_"$dataset"_"$outer"_"$lower"_"$upper"<<EOF
python /data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/code/parallel_LOSO.py "$dataset" "$outer" "$lower" "$upper"
EOF
done
```

This is the snapshpt of the bash script for executing `parallel_LOSO.py` on BIC clusters. There are four parameters that needs to be specified: 

- `dataset` ( ‘c’, ‘i’, ‘s’ ): indicates which dataset to use lower: indicates the lower bound number of features
- `upper`: indicates the upper bound number of features 
- `outer` fold list : indicates how many outer folds there are.
- In order to reduce total execution time, the gridsearch process is splitted into multiple jobs. Eg. if dataset = ‘c’, lower = 2, upper = 5, with 10 outer folds (0,1,2,3,4) the script will submit 5 jobs to the clusters by calling `parallel_LOSO.py` 10 times. 
- The gridsearch is performed on the dataset Xc with the number of features from 2 to 5.



