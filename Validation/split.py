import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
import os

if __name__ == "__main__":

    print('Argument list: ', str(sys.argv))

    outer_fold_cv = int(sys.argv[1])
    class_sep = int(sys.argv[2])
    # inner_fold_cv = int(sys.argv[2])

    print('Outer fold cv: ', outer_fold_cv)
    print('Class Separation: ', class_sep)
    # print('Inner fold cv: ', inner_fold_cv)

    dataset_root_path = ('./datasets/integrated/{}').format(class_sep)
    # load data
    Xc = np.genfromtxt(
        (dataset_root_path+('/X_{}_integrated.csv').format(class_sep)), delimiter=',')
    # site = np.genfromtxt('./datasets/Site.csv', delimiter=',')
    Y = np.genfromtxt(
        (dataset_root_path+('/y_{}_integrated.csv').format(class_sep)), delimiter=',')
    # Xc = np.concatenate((Xc, np.reshape(site, (-1, 1))), axis=1)
    sublabels = np.genfromtxt(
        (dataset_root_path+('/subtype_labels_{}.csv').format(class_sep)), delimiter=',')

    skf_outer = StratifiedKFold(
        n_splits=outer_fold_cv, shuffle=True, random_state=10)

    outer_train_indexes = []
    outer_test_indexes = []

    for train_outer, test_outer in skf_outer.split(Xc, sublabels):
        # print ("Outer: {}").format(outer_fold_cv)
        print(test_outer)
        outer_train_indexes.append(train_outer)
        outer_test_indexes.append(test_outer)

    ####################################### create directory #######################################

    save_dir = ('./split/{}').format(class_sep)
    try:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    except OSError:
        print("Creation of the directory %s failed" % save_dir)
        quit()
    else:
        print("Successfully created the directory %s " % save_dir)

    ####################################### save splitting file (pickle) #######################################

    pickle_out = open((save_dir+"/outer_train_index.pickle"), "wb")
    pickle.dump(outer_train_indexes, pickle_out)
    pickle_out.close()
    pickle_out_test = open((save_dir+"/outer_test_index.pickle"), "wb")
    pickle.dump(outer_test_indexes, pickle_out_test)
    pickle_out_test.close()

    ####################################### save splitting file (csv) #######################################

    max_train_fold_size = max([len(fold) for fold in outer_train_indexes])
    train_indexes = np.ones((outer_fold_cv, max_train_fold_size))
    for index, data in enumerate(outer_train_indexes):
        for i, d in enumerate(data):
            train_indexes[index][i] = int(d)
    np.savetxt((save_dir+"/outer_train_index.csv"),
               train_indexes, delimiter=",")

    max_test_fold_size = max([len(fold) for fold in outer_test_indexes])
    test_indexes = np.ones((outer_fold_cv, max_test_fold_size))
    for index, data in enumerate(outer_test_indexes):
        for i, d in enumerate(data):
            test_indexes[index][i] = int(d)
    np.savetxt((save_dir+"/outer_test_index.csv"), test_indexes, delimiter=",")
