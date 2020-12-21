import sys
import pandas as pd 
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":

    print('Argument list: ', str(sys.argv))

    outer_fold_cv = int(sys.argv[1])
    inner_fold_cv = int(sys.argv[2])

    print('Outer fold cv: ', outer_fold_cv)
    print('Inner fold cv: ', inner_fold_cv)

	# load data
    Xc = np.genfromtxt('../data/Xc.csv', delimiter=',')
    site = np.genfromtxt('../data/Site.csv', delimiter=',')
    Y  = np.genfromtxt('../data/Y.csv', delimiter=',')
    Xc = np.concatenate((Xc, np.reshape(site, (-1, 1))), axis=1)

    skf_outer = StratifiedKFold(n_splits=outer_fold_cv, shuffle=True, random_state=10)

    outer_train_indexes = []
    outer_test_indexes = []
    inner_train_indexes = []

    for train_outer, test_outer in skf_outer.split(Xc, Xc[:,-1]):
        # print ("Outer: {}").format(outer_fold_cv)
        print(test_outer)
        outer_train_indexes.append(train_outer)
        outer_test_indexes.append(test_outer)
    pickle_out = open("../split/outer_train_index.pickle","wb")
    pickle.dump(outer_train_indexes, pickle_out)
    pickle_out_test = open("../split/outer_test_index.pickle","wb")
    pickle.dump(outer_test_indexes, pickle_out_test)

    pickle_out.close()
    
