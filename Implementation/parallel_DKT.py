import sys
import pandas as pd
import os
import numpy as np
import pickle
import math
from neurocombat_sklearn import CombatModel
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import cross_validate
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
import time
from skfeature.function.information_theoretical_based import MRMR
from sklearn.model_selection import StratifiedKFold
from neurocombat_sklearn import CombatModel
from skfeature.function.information_theoretical_based import MRMR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import ParameterGrid
from pyper import *
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from scipy import stats
from snf import compute


################################################# Combat Step #################################################

class Combatstep():

    def __init__(self, debug_mode=False):
        """Apply fit_transform on the training data

            Arguments
            ---------
                debug_mode: boolean
                    If true, intermediate results will be printed for debugging purposes 
                    at the end of each function call.
        """
        self.model = CombatModel()
        self.debug_mode = debug_mode
        pass

    def fit_transform(self, X, y):
        """Apply fit_transform on the training data

            This function should be called upon the training data set. The combat model 
            is fitted and transformed. The data is adjusted/harmonized from the effects 
            of site data, sex data, age data and the labels. The discrete covariates includes
            the sex data and labels. The continuous covariates includes the normalized age
            data. 

            Arguments
            ---------
                X: numpy.ndarray
                    The training data with site, sex, age data attached.
                y: numpy.ndarray
                    The training data labels.
            Returns
            -------
                X_harmonized
                    The resulting harmonized data, ex cluding the site, sex, age data.
        """
        # extract site, sex, age data
        site = X[:, -1].reshape(-1, 1)
        sex = X[:, -2].reshape(-1, 1)
        age = X[:, -3].reshape(-1, 1)

        # exclude site, sex, age from from the data for harmonization
        X_raw = X[:, :-3]

        # reshape y array for concatenation
        y = y.reshape(-1, 1)

        # discrete_covariates: sex and y
        discrete_covariates = sex

        # continuous_covariates: age
        continuous_covariates = age

        # apply fit_transform on X
        X_harmonized = self.model.fit_transform(X_raw,
                                                site,
                                                discrete_covariates,
                                                continuous_covariates
                                                )

        # attach back the age and sex data to the X_harmonized
        age_sex_data = np.concatenate((age, sex), axis=1)

        X_harmonized = np.concatenate((X_harmonized, age_sex_data), axis=1)

        # print all the computed matrxi if debug mode is enabled
        if self.debug_mode == True:
            print("=" * 50 + " combat step fit_transform" + "=" * 50)
            print("X_init", X)
            print("X_raw", X_raw)
#             print("discrete_covariates", discrete_covariates)
#             print("continuous_covariates", continuous_covariates)
            print("X_harmonized.shape", X_harmonized.shape)
            print("age_sex_data", age_sex_data)
            print("X_harmonized", X_harmonized)

        return X_harmonized

    def transform(self, X, y):
        """Apply transformation on the testing data based on the fitted combat model.

            This function should be called upon the testing data set. The combat model has 
            already been fitted using the training data and the testing data is transformed
            accordingly. The data is adjusted/harmonized from the effects of site data, sex 
            data, age data and the labels. The discrete covariates includes the sex data 
            and labels. The continuous covariates includes the normalized age
            data. 

            Arguments
            ---------
                X: numpy.ndarray
                    The testing data with site, sex, age data attached.
                y: numpy.ndarray
                    The testing data labels.
            Returns
            -------
                X_harmonized
                    The resulting harmonized data, excluding the site, sex, age data.
        """
        # extract site, sex, age data
        site = X[:, -1].reshape(-1, 1)
        sex = X[:, -2].reshape(-1, 1)
        age = X[:, -3].reshape(-1, 1)
        # exclude site, sex, age from from the data for harmonization
        X_raw = X[:, :-3]

        # reshape y array for concatenation
        y = y.reshape(-1, 1)

        # discrete_covariates: sex and y
        discrete_covariates = sex

        # continuous_covariates: age
        continuous_covariates = age

        # apply fit_transform on X
        X_harmonized = self.model.transform(X_raw,
                                            site,
                                            discrete_covariates,
                                            continuous_covariates
                                            )
        # attach back the age and sex data to the X_harmonized
        age_sex_data = np.concatenate((age, sex), axis=1)

        X_harmonized = np.concatenate((X_harmonized, age_sex_data), axis=1)

        # print all the computed matrxi if debug mode is enabled
        if self.debug_mode == True:
            print("=" * 50 + " combat step transform" + "=" * 50)
            print("X_init", X)
            print("X_raw", X_raw)
            print("discrete_covariates", np.reshape(
                discrete_covariates, (1, -1)))
            print("continuous_covariates", np.reshape(
                continuous_covariates, (1, -1)))
            print("X_harmonized.shape", X_harmonized.shape)
            print("X_harmonized", X_harmonized)

        return X_harmonized


################################################# Statistical Contrast #################################################

def t_test(X, y):

    def separate(X, y):
        # Y is the classes (1=stable, 2=progressor)
        P_X, S_X = [], []
        for i in range(X.shape[0]):
            if y[i] == 1:
                S_X.append(X[i])
            elif y[i] == 2:
                P_X.append(X[i])
        return np.asarray(P_X), np.asarray(S_X)

    def t_test(P, S, y):
        t_vals = []
        for i in range(P.shape[1]):
            t_score, p_val = stats.ttest_ind(P[:, i], S[:, i])
            tuple_ = (abs(t_score), abs(p_val))
            t_vals.append((tuple_, i))
        return t_vals

    def select(X, y, t_vals, percentage=.1):
        total_nfeatures = X.shape[1]
        filtered_nfeatures = math.ceil(total_nfeatures*percentage)
        # print("total_nfeatures: ", total_nfeatures)
        # print("filtered_nfeatures", filtered_nfeatures)
        sorted_t_val = sorted(t_vals, key=lambda tup: tup[0][0], reverse=True)
        # print(sorted_t_val)
        selected_indices = [i[1] for i in sorted_t_val[:filtered_nfeatures]]
        sorted_selected_indices = sorted(selected_indices)
        X_reduced = X[:, sorted_selected_indices]
        return X_reduced, sorted_selected_indices

    P_X, S_X = separate(X, y)
    t_vals = t_test(P_X, S_X, y)
    X_reduced, selected_indices = select(X, y, t_vals)
    return X_reduced, selected_indices

################################################# Clustering Step #################################################


class Clustering():

    def __init__(self, debug_mode=False):
        """Apply fit_transform on the training data

            Arguments
            ---------
                debug_mode: boolean
                    If true, intermediate results will be printed for debugging purposes 
                    at the end of each function call.
        """
        self.debug_mode = debug_mode
        pass

    # setter for number of subclusters for P
    def setNumP(self, numP):
        self.numP = numP

    # setter for number of subclusters for S
    def setNumS(self, numS):
        self.numS = numS

    # Step1. Separate out P and S
    def separate(self, X, y):
        """Separate out the progressor and stable instances to be used for clustering later. 

            As the data in is reindexed by the stratifiedKfold. It is important for us
            to save the indexes of the rows before we separate them out and do clustering 
            accordingly. In this function, while the data is splitted into separate lists 
            by class, the original index for each row is saved. After the clustering is done, 
            we will reindex our combined data set according to the saved indexes.

            Arguments
            ---------
                X: numpy.ndarray
                    The combined data
                y: numpy.ndarray
                    The combined data labels.
            Returns
            -------
                P_X: list
                    Progressor data
                P_y: list
                    Progressor data labels
                S_X: list
                    Stable data
                S_y: list
                    Stable data labels     
        """

        # initialize empty lists to store index of P&S instances in the
        # original data
        index_P, index_S = [], []

        # initialize lists to store the separated P and S instances
        P_X, P_y, S_X, S_y = [], [], [], []
        # store P and S instances separately and keep track of the indexes in
        # the original data

        for i in range(0, y.shape[0]):
            if y[i] == 1:
                S_X.append(X[i])
                S_y.append(y[i])
                index_S.append(i)
            else:
                P_X.append(X[i])
                P_y.append(y[i])
                index_P.append(i)

        # store the indexes as a global attribute for the clasds
        self.index_P = index_P
        self.index_S = index_S

        # print all the computed matrxi if debug mode is enabled
        if self.debug_mode == True:
            print("index_P", self.index_P)
            print("index_S", self.index_S)

#         print("P_X separate", P_X)
        return np.asarray(P_X), np.asarray(P_y), np.asarray(S_X), np.asarray(S_y)

    def clusteringP(self, P_X):
        """Apply Spectral Clustering on Progressor data.

            SpectralClustering is performed on P class instances. The number of clusters 
            is specified by the global attribute num_P. At the end of the step, the original
            labels will be discarded and the new labels will be attached to the end of the data
            for convenience. The assign_labels for the SpectralClustering is "discrete" and 
            random_state is set to 0. 

            Arguments
            ---------
                P_X: list
                    The progressor data.
            Returns
            -------
                P_X_clustered: numpy.ndarray
                    The combined clustered progressor data with new labels attached to the end.
        """
        affinity_networks = compute.make_affinity(P_X, metric='euclidean', K=self.numP, mu=0.5)
        
        clusteringP = SpectralClustering(affinity='precomputed',
            n_clusters=self.numP, random_state=0).fit(affinity_networks)
        
        labels = clusteringP.labels_.reshape(-1, 1)
        P_X = np.asarray(P_X)
        P_X_clustered = np.concatenate((P_X, labels), axis=1)

        if self.debug_mode == True:
            print("P labels", np.reshape(labels, (1, -1)))
            print("P_X_clustered", P_X_clustered)

        return P_X_clustered

    def clusteringS(self, S_X):
        """Apply Spectral Clustering on Stable data.

            SpectralClustering is performed on the S class instances. The number of clusters 
            is specified by the global attribute num_S. At the end of the step, the original
            labels will be discarded and the new labels will be attached to the end of the data
            for convenience. The assign_labels for the SpectralClustering is "discrete" and 
            random_state is set to 0. 

            Arguments
            ---------
                S_X: list
                    The progressor data.
            Returns
            -------
                S_X_clustered: numpy.ndarray
                    The combined clustered progressor data with new labels attached to the end.
        """

        affinity_networks = compute.make_affinity(S_X, metric='euclidean', K=self.numS, mu=0.5)
        
        clusteringS = SpectralClustering(affinity='precomputed',
            n_clusters=self.numS, random_state=0).fit(affinity_networks)

        labels = clusteringS.labels_
        labels = labels + self.numP
        S_X = np.asarray(S_X)
        labels = labels.reshape(-1, 1)
        S_X_clustered = np.concatenate((S_X, labels), axis=1)

        if self.debug_mode == True:
            print("S labels", np.reshape(labels, (1, -1)))
            print("S_X_clustered", S_X_clustered)

        return S_X_clustered

    def transform(self, X, y):
        """Apply Spectral Clustering on stable data and progressor data. 
            Step 1. Separate data out into P and S classes. Save the original data index as 
            the data rows will be reindexed for clustering. 
            Step 2. Perform clustering for P class based on num_P. Attach the new labels.
            Step 3. Perform clustering for S class based on num_S. Attach the new labels. 
            Step 4. Combine the newly clustered P class data and S class data. Reindex the 
            data rows according to the saved indexes. 
            Arguments
            ---------
                X: numpy.ndarray
                    The combined data of both P and S instances.
                y: numpy.ndarray
                    The combined data labels.
            Returns
            -------
                X_output: numpy.ndarray
                    The combined data with newly clustered labels attached.
        """

        P_X, P_y, S_X, S_y = self.separate(X, y)

        P_X_clustered = self.clusteringP(P_X)
        S_X_clustered = self.clusteringS(S_X)

        P_X_clustered_with_index = np.concatenate(
            (P_X_clustered, (np.array(self.index_P)).reshape(-1, 1)), axis=1)
        S_X_clustered_with_index = np.concatenate(
            (S_X_clustered, (np.array(self.index_S)).reshape(-1, 1)), axis=1)

        X_combined_with_index = np.concatenate(
            (P_X_clustered_with_index, S_X_clustered_with_index), axis=0)

        X_sorted = X_combined_with_index[
            np.argsort(X_combined_with_index[:, -1])]
        new_labels = X_sorted[:,-2]
        X_output = X_sorted[:, :-2]

        if self.debug_mode == True:
            print("P_X_clustered_with_index", P_X_clustered_with_index)
            print("S_X_clustered_with_index", S_X_clustered_with_index)
            print("X_sorted", X_sorted)
            print("X_output", X_output)
            print("X_output.labels", X_output[:, -1])

        return X_output, new_labels

################################################# Clustering Step #################################################


def SeparateX(X, y=None, debug_mode=False):
    """Separate out the X data from a combined data matrix which labels are attached.

        This is a simple helper function to faciliate the operations. 

        Arguments
        ---------
            X: numpy.ndarray
                The combined data matrix with labels attached to the end.
        Returns
        -------
            X_: numpy.ndarray
                The matrix which excludes the labels.
    """

    X_ = X[:, :-1]

    if debug_mode == True:
        print("X_", X_)

    return X_


def SeparateY(X, y=None, debug_mode=False):
    """Separate out the labels from a combined data matrix which labels are attached.

        This is a simple helper function to faciliate the operations. 

        Arguments
        ---------
            X: numpy.ndarray
                    The combined data matrix with labels attached to the end.
        Return
        -------
            y_: numpy.ndarray
                The matrix of the labels.
        """
    y_ = X[:, -1]

    if debug_mode == True:
        print("y_", y_)

    return y_

################################################# MRMR Step #################################################


def mrmrSelection(dataframe, nFeature):
    """Perform MRMR on the passed in data. 

        This function performs feature selection based on the mRMRe feature selection. 
        It utilizes the mRMRe package originally written in R. A python wrapper for R is
        used here. The eMRMRe returns a list of the selected column indices given the 
        number of features to be selected. 

        Arguments
        ---------
            dataframe: pandas.DataFrame
                The dataframe containing clustered data. 
        Returns
        -------
            df_: pandas.DataFrame
                The dataframe with only selected columns.
            labels: pandas.DataFrame
                The dataframe of only the labels
            colum_indices: list
                The list of indices of the selected columns
    """ 
    base = importr('base')
    mr = importr('mRMRe')
    # print('initial mrmr shape', dataframe.shape)
    pandas2ri.activate()
    # Convert the data into R format
    with localconverter(ro.default_converter + pandas2ri.converter):
        rdataframe = ro.conversion.py2rpy(dataframe)
    mrmrData = mr.mRMR_data(data = (rdataframe))
    solutionCount = 1  # can i change the value for solutionCount???
    selectionEnsemble = mr.mRMR_ensemble("mRMRe.Data",data = mrmrData, target_indices = (dataframe.shape[1]),
          feature_count = nFeature, solution_count = solutionCount)
    colum_indices = column_indices=(mr.solutions(selectionEnsemble)[0]-1)

    df_ = pd.DataFrame()
    for i,v in enumerate(colum_indices):
        df_[i] = dataframe[v[0]]
    # print('mrmr shape', df_.shape)
    labels = dataframe.iloc[:,-1]

    colum_indices = list(np.reshape(colum_indices,(1,-1))[0])
    # print('mrmr ')
    return df_, labels, colum_indices


def computeSampleWeight(y_train_selected, num_P, weight):
    sample_weight = np.ones(len(y_train_selected))
    range_P = np.arange(num_P)
    for i, v in enumerate(y_train_selected):
        if v in range_P:
            sample_weight[i] = weight * sample_weight[i]
    return sample_weight

################################################# Integrated Search Step ##################################################

def integrated_grid_search(
    X, Y, dataset, train_index, test_index, fold,
    param_grid, train_outer_index, weightings
):
    """Performs gridsearch on the 

    Arguments
    ---------
    X_train: numpy.ndarray
        The training data.
    y_train: numpy.ndarray
        The training data labels.
    X_test: numpy.ndarray
        The testing data.
    y_test: numpy.ndarray
        The testing data labels.
    num_P: int
        Maximum number of sub-clusters for Progressors.
    num_S: int
        Maximum number of sub-clusters for Stables.
    feat_min: int
        Lower bound for number of features to be selected.
    feat_max: int
        Upper bound for number of features to be selected.
    classifiers: integer
        A integer representing the classifiers used. 
        0: linear SVM
        1: RandomForest
        2: Logistic Regression
    combat_step: CombatStep
    integrated_clustering: Clustering
    feature_selector: featureSelector

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    # unwrap the parameter grid

    # print(X.shape)
    # print(X_train_i)
#     print(fold)
    columns_fold_results = ['fold', 'num_P', 'num_S', 'num_Features', 'classifier',
                            'weighting', 'columns_selected', 'pred_y', 'act_y', 'test_index']

    results_for_each_fold = pd.DataFrame(columns=columns_fold_results)

    X_train_i = train_index[fold]
    X_test_i = test_index[fold]

    Y_train_i = train_index[fold]
    Y_test_i = test_index[fold]

    X_train = X[X_train_i]
    X_test = X[X_test_i]

    y_train = Y[Y_train_i]
    y_test = Y[Y_test_i]

    num_P = param_grid.get('numP')
    num_S = param_grid.get('numS')
    nFeatures = param_grid.get('n_features')
    n_classifiers = 3

#     print("num_P", num_P)
#     print("num_S", num_S)

    # combat step
    combat_step = Combatstep(debug_mode=False)
    X_train_harmonized = combat_step.fit_transform(X_train, y_train)
    X_test_harmonized = combat_step.transform(X_test, y_test)
#     print("X_train_harmonized", X_train_harmonized)
#     print(numpy.isnan(X_train_harmonized).any())
    columns_ttest = []
    # for Xc and Xi dataset, perform the t_test feature reduction
    if dataset == 'c' or dataset == 'i':
        # remove the age and sex data for the training data before t_test
        train_sex = X_train_harmonized[:,-1].reshape(-1,1)
        train_age = X_train_harmonized[:,-2].reshape(-1,1)
        train_age_sex_data = np.concatenate((train_age, train_sex), axis=1)
        # print("T test train_age_sex_data", train_age_sex_data)

        X_train_harmonized_without_age_sex = X_train_harmonized[:,:-2]
        
        # print("X_train_harmonized_without_age_sex", X_train_harmonized_without_age_sex)
        # print("X_train_harmonized_without_age_sex.shape", X_train_harmonized_without_age_sex.shape)
        
        X_train_ttest_without_age_sex, columns_ttest = t_test(X_train_harmonized_without_age_sex, y_train)
        X_train_ttest_with_age_sex = np.concatenate((X_train_ttest_without_age_sex, train_age_sex_data), axis=1)
        
        # print("X_train_ttest_without_age_sex", X_train_ttest_without_age_sex)
        # print("X_train_ttest_without_age_sex.shape", X_train_ttest_without_age_sex.shape)
        
        
        # print("X_train_ttest_with_age_sex", X_train_ttest_with_age_sex)
        # print("X_train_ttest_with_age_sex.shape", X_train_ttest_with_age_sex.shape)
        
        # remove the age and sex data for testing data
        test_sex = X_test_harmonized[:,-1].reshape(-1, 1)
        test_age = X_test_harmonized[:,-2].reshape(-1, 1)
        test_age_sex_data = np.concatenate((test_age, test_sex), axis=1)

        X_test_harmonized_without_age_sex = X_test_harmonized[:,:-2]
        X_test_ttest_without_age_sex = X_test_harmonized_without_age_sex[:,columns_ttest]
        X_test_ttest_with_age_sex = np.concatenate((X_test_ttest_without_age_sex, test_age_sex_data), axis=1)
        
        # print("X_test_ttest_without_age_sex", X_test_ttest_without_age_sex)
        # print("X_test_ttest_without_age_sex.shape", X_test_ttest_without_age_sex.shape)
        
        # print("X_test_ttest_with_age_sex", X_test_ttest_with_age_sex)
        # print("X_test_ttest_with_age_sex.shape", X_test_ttest_with_age_sex.shape)

    else:
        X_train_ttest_with_age_sex = X_train_harmonized
        X_test_ttest_with_age_sex = X_test_harmonized
        # print("X_train_ttest_with_age_sex", X_train_ttest_with_age_sex)
        # print("X_train_ttest_with_age_sex.shape", X_train_ttest_with_age_sex.shape)
        
        # print("X_test_ttest_with_age_sex", X_test_ttest_with_age_sex)
        # print("X_test_ttest_with_age_sex.shape", X_test_ttest_with_age_sex.shape)

    # clustering step
    integrated_clustering = Clustering()
    integrated_clustering.setNumP(num_P)
    integrated_clustering.setNumS(num_S)

    # array_has_nan = np. isnan(X_train_ttest_with_age_sex).any()
    # min_val = X_train_ttest_with_age_sex.min()
    # print(array_has_nan)
    # print('min val: ', min_val)
    # print(X_train_ttest_with_age_sex.shape)
    # print(y_train.shape)
    X_train_clustered, y_train_new = integrated_clustering.transform(
        X_train_ttest_with_age_sex, y_train)
    
    # print("X_train_clustered", X_train_clustered)
    # print("X_train_clustered.shape", X_train_clustered.shape)
    
    # print("y_train_new", y_train_new)
    # print("y_train_new.shape", y_train_new.shape)
    
    # remove age and sex data before mrmr step
    train_sex = X_train_clustered[:,-1].reshape(-1, 1)
    train_age = X_train_clustered[:,-2].reshape(-1, 1)
    train_age_sex_data = np.concatenate((train_age, train_sex), axis=1)
    
    # print("train_age_sex_data", train_age_sex_data)
    # print("train_age_sex_data.shape", train_age_sex_data.shape)

    X_train_clustered_without_age_sex = X_train_clustered[:,:-2]
    
    # print("X_train_clustered_without_age_sex", X_train_clustered_without_age_sex)
    # print("X_train_clustered_without_age_sex.shape", X_train_clustered_without_age_sex.shape)

    X_test_combined = np.concatenate(
        (X_test_harmonized, np.reshape(y_test, (-1, 1))), axis=1)
    
    # print("X_test_combined", X_test_combined)
    # print("X_test_combined.shape", X_test_combined.shape)

    X_train_clustered_without_age_sex_with_y = np.concatenate((X_train_clustered_without_age_sex, y_train_new.reshape(-1,1)), axis=1)

    # print("X_train_clustered_without_age_sex_with_y", X_train_clustered_without_age_sex_with_y)
    # print("X_train_clustered_without_age_sex_with_y.shape", X_train_clustered_without_age_sex_with_y.shape)    
    
    # feature selection step
    X_train_selected_without_age_sex, y_train_selected, column_indices = mrmrSelection(
        pd.DataFrame(X_train_clustered_without_age_sex_with_y), nFeatures)
    
    # print("X_train_selected_without_age_sex", X_train_selected_without_age_sex)
    # print("X_train_selected_without_age_sex.shape", X_train_selected_without_age_sex.shape)  
    
    # print("y_train_selected", y_train_selected)
    # print("y_train_selected.shape", y_train_selected.shape)   
    
    # print("column_indices", column_indices)
    # print("column_indices.len", len(column_indices))    

    # attach age and sex data back to the X_train after mrmr
    X_train_selected_with_age_sex = np.concatenate((X_train_selected_without_age_sex, train_age_sex_data), axis=1)

    # print("X_train_selected_with_age_sex", X_train_selected_with_age_sex)
    # print("X_train_selected_with_age_sex.shape", X_train_selected_with_age_sex.shape)      
    
    if dataset != 's':
        sub_list = [columns_ttest[i] for i in column_indices]
        # column_indices = columns_ttest[column_indices]
        column_indices = sub_list
        # apply the selected features on test data
        # remove the age and sex data for testing data
    y_test_selected = X_test_combined[:, -1]
    test_sex = X_test_combined[:,-2].reshape(-1, 1)
    test_age = X_test_combined[:,-3].reshape(-1, 1)

    # print('X_test_combined',X_test_combined)
    # print('X_test_combined.shape',X_test_combined.shape)

    test_age_sex_data = np.concatenate((test_age, test_sex), axis=1)

    # print('test_age_sex_data',test_age_sex_data)
    # print('test_age_sex_data.shape',test_age_sex_data.shape)

    # print("len(column_indices)", len(column_indices))
    X_test_selected = X_test_combined[:, column_indices]
    X_test_selected = np.concatenate((X_test_selected,test_age_sex_data), axis=1)

    # print("X_test_selected", X_test_selected)
    # print("X_test_selected.shape", X_test_selected.shape)     
    
    # start grid search

    for classifier in range(n_classifiers):
        if classifier == 0:
            # print('classifier ', classifier)
            for weighting in weightings:
                # print('weighting ', weighting)
                sample_weights = computeSampleWeight(
                    y_train_selected, num_P, weighting)
                clf = SVC()
                clf.fit(X_train_selected_with_age_sex, y_train_selected,
                        sample_weight=sample_weights)
                pred_y = clf.predict(X_test_selected)

                fold_row = {'fold': fold, 'num_P': num_P, 'num_S': num_S, 'num_Features': nFeatures,
                            'classifier': classifier, 'weighting': weighting, 'columns_selected': column_indices, 'pred_y': pred_y, 'act_y': y_test_selected, 'test_index': X_test_i}
                # print('fold_row ', fold_row)
                results_for_each_fold = results_for_each_fold.append(
                    fold_row, ignore_index=True)

        elif classifier == 1:
            for weighting in weightings:
                sample_weights = computeSampleWeight(
                        y_train_selected, num_P, weighting)
                clf = RandomForestClassifier(random_state=0)
                clf.fit(X_train_selected_with_age_sex, y_train_selected,
                        sample_weight=sample_weights)
                pred_y = clf.predict(X_test_selected)

                fold_row = {'fold': fold, 'num_P': num_P, 'num_S': num_S, 'num_Features': nFeatures,
                            'classifier': classifier, 'weighting': weighting, 'columns_selected': column_indices, 'pred_y': pred_y, 'act_y': y_test_selected, 'test_index': X_test_i}

                results_for_each_fold = results_for_each_fold.append(
                        fold_row, ignore_index=True)

        else:
            for weighting in weightings:
                sample_weights = computeSampleWeight(
                        y_train_selected, num_P, weighting)
                clf = LogisticRegression(random_state=0, max_iter=5000)
                clf.fit(X_train_selected_with_age_sex, y_train_selected,
                            sample_weight=sample_weights)
                pred_y = clf.predict(X_test_selected)

                fold_row = {'fold': fold, 'num_P': num_P, 'num_S': num_S, 'num_Features': nFeatures,
                            'classifier': classifier, 'weighting': weighting, 'columns_selected': column_indices, 'pred_y': pred_y, 'act_y': y_test_selected, 'test_index': X_test_i}

                results_for_each_fold = results_for_each_fold.append(
                            fold_row, ignore_index=True)
    # print(results_for_each_fold)
    return results_for_each_fold




################################################# Metric Calculator #################################################

# True positive value

def TP(y_actual, y_pred):
    TP = 0
    for i in range(len(y_actual)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
    return TP

# False positive value


def FP(y_actual, y_pred):
    FP = 0
    for i in range(len(y_actual)):
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
    return FP

# True negative value


def TN(y_actual, y_pred):
    TN = 0
    for i in range(len(y_actual)):
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
    return TN

# False negative value


def FN(y_actual, y_pred):
    FN = 0
    for i in range(len(y_actual)):
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1
    return FN

# tp / (tp + fp)


def precision(y_act, y_pred):
    tp = TP(y_act, y_pred)
    fp = FP(y_act, y_pred)
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def binarize(label, arr):
    arr_copy = arr.copy()
    for i, v in enumerate(arr_copy):
        if v == label:
            arr_copy[i] = 1
        else:
            arr_copy[i] = 0
    return arr_copy

# (1=stable, 2=progressor)


def binarizeP(arr):
    arr_copy = arr.copy()
    for i, v in enumerate(arr_copy):
        if v == 2:
            arr_copy[i] = 1
        else:
            arr_copy[i] = 0
    return arr_copy

# (1=stable, 2=progressor)


def binarizeS(arr):
    arr_copy = arr.copy()
    for i, v in enumerate(arr_copy):
        if v == 1:
            arr_copy[i] = 1
        else:
            arr_copy[i] = 0
    return arr_copy


# P: [0, 1, 2]  num_P = 3
# S: [3, 4, 5]  num_S = 3
# n_class = 6

def computePPV(numP, numS, y_pred, y_act):
    n_class = numP + numS
    P_range, S_range = np.arange(0, numP), np.arange(numP, n_class)
    dic = {}
    # print("y_act",y_act)
    for i in range(n_class):
        # print("class label:", i)
        if (i in P_range):
            # print("%d in P class" %(i))
            y_p = binarize(i, y_pred)
            # print("predicted y after binarization", y_p)
            y_a = binarizeP(y_act)
            # print("actual y after binarization of P", y_a)
            score = precision(y_a, y_p)
        else:
            # print("%d in S class" %(i))
            y_p = binarize(i, y_pred)
            # print("predicted y after binarization", y_p)
            y_a = binarizeS(y_act)
            # print("actual y after binarization of S", y_a)
            score = precision(y_a, y_p)
        dic[i] = score
    return dic

################################################# Create Parameter Grid #################################################

# for multiprocessing
def create_param_grid(numP, numS, n_feature_max, n_feature_min=2):
    grid = ParameterGrid({
        'numP': range(1, numP + 1),
        'numS': range(1, numS + 1),
        'n_features': range(n_feature_min, n_feature_max + 1)
    })
    return grid


# for single threaded processing
def single_param_grid(numP, numS, n_features):
    grid = ParameterGrid({
        'numP': range(1, numP + 1),
        'numS': range(1, numS + 1),
        'n_features': range(n_features, n_features + 1)
    })
    return grid


def isGoodResult(threshold, dic):
    isGood = False
    for value in dic.values():
        if value > threshold:
            isGood = True
            print("Found! PPV:", value)
            break
    return isGood


################################################# Compute Parameter Result #################################################

def computeParamRowResult(dataframe, param, cw):

    num_P = param.get('numP')
    num_S = param.get('numS')
    n_features = param.get('n_features')
    classifier = cw[0]
    weighting = cw[1]
    pred_y = list(dataframe['pred_y'])
    act_y = list(dataframe['act_y'])
    test_index = list(dataframe['test_index'])

    pred_y = [item for items in pred_y for item in items]
    act_y = [item for items in act_y for item in items]
    test_index = [item for items in test_index for item in items]

    dic = computePPV(num_P, num_S, pred_y, act_y)

    row_param = {'num_P': num_P,'num_S': num_S,
                    'num_Features': n_features,'classifier': classifier, 
                    'weighting': weighting, 'pred_y': pred_y,'act_y': act_y,
                    'test_index': test_index,'precision_score':dic
                }
    return row_param

################################################# Main #################################################

if __name__ == "__main__":

    # argument will be 'outerfold' 'number of features'

    # print the argument list
    print ('Number of arguments:', len(sys.argv), 'arguments.')
    print ('Argument List:', str(sys.argv))

    # specify which data set to be worked on
    dataset = sys.argv[1].lower()

    # specify which outer fold
    outer_fold_cv = int(sys.argv[2])

    # specify the number of features to be searched
    num_Features_lower = int(sys.argv[3])

    num_Features_upper = int(sys.argv[4])

    print('Outer fold:', outer_fold_cv)
    print('Features lower: ', num_Features_lower)
    print('Features upper: ', num_Features_upper)

     # save results to csv and pickle
    path_results_for_each_fold = ("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/DKT_results/X{}/results/fold{}/outerfold{}_results_for_each_fold_features_{}_{}.csv").format(dataset, outer_fold_cv, outer_fold_cv, num_Features_lower, num_Features_upper)
    path_results_for_each_param = ("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/DKT_results/X{}/results/fold{}/outerfold{}_results_for_each_param_setting_features_{}_{}.csv").format(dataset, outer_fold_cv, outer_fold_cv, num_Features_lower, num_Features_upper)
    path_good_results = ("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/DKT_results/X{}/results/fold{}/outerfold{}_good_results_features_{}_{}.csv").format(dataset, outer_fold_cv, outer_fold_cv, num_Features_lower, num_Features_upper)

    pkl_path_results_for_each_fold = ("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/DKT_results/X{}/results/fold{}/outerfold{}_results_for_each_fold_features_{}_{}.pkl").format(dataset, outer_fold_cv, outer_fold_cv, num_Features_lower, num_Features_upper)
    pkl_path_results_for_each_param = ("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/DKT_results/X{}/results/fold{}/outerfold{}_results_for_each_param_setting_features_{}_{}.pkl").format(dataset, outer_fold_cv, outer_fold_cv, num_Features_lower, num_Features_upper)
    pkl_path_good_results = ("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/DKT_results/X{}/results/fold{}/outerfold{}_good_results_features_{}_{}.pkl").format(dataset, outer_fold_cv, outer_fold_cv, num_Features_lower, num_Features_upper)

    print('path_results_for_each_fold', path_results_for_each_fold)
    print('path_results_for_each_param', path_results_for_each_param)
    print('path_good_results', path_good_results)

    # load data
    dataset_path = ("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/data_DKT/X{}.csv").format(dataset)

    X = np.genfromtxt(dataset_path, delimiter=',')
    site = np.genfromtxt("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/data_DKT/Site.csv", delimiter=',')
    Y = np.genfromtxt("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/data_DKT/Y.csv", delimiter=',')

    # concate site data to X data
    X = np.concatenate((X, np.reshape(site, (-1, 1))), axis=1)

    # load pre-splitted train-test index
    outer_train_index_pickle = open("/data/shmuel/shmuel1/debbie/environment/parallel_script/MCI/split/outer_train_index.pickle", "rb")
    outer_train_indexes = pickle.load(outer_train_index_pickle)

    # print('All outer fold indexes: ', outer_train_indexes)

    # split the data based on outer fold
    outer_train_index = outer_train_indexes[outer_fold_cv]
    # print('Outer fold: ', outer_fold_cv)
    # print('Fold index: ', outer_train_index)

    # loda the fold specific data
    X_fold = X[outer_train_index]
    Y_fold = Y[outer_train_index]

    # split inner folds
    inner_cv_fold = 12
    skf_inner = StratifiedKFold(n_splits=inner_cv_fold, shuffle=True, random_state=1000)

    inner_train_indexes = []
    inner_test_indexes = []

    for train_outer, test_outer in skf_inner.split(X_fold, X_fold[:,-1]):
        # print("test_outer", test_outer)
        inner_train_indexes.append(train_outer)
        inner_test_indexes.append(test_outer)


    # create paramater grid
    weighting = [1, 5, 10, 20, 50, 100]
    param_grid = create_param_grid(4, 4, num_Features_upper, n_feature_min=num_Features_lower)

    columns_fold_results = ['fold','num_P','num_S','num_Features', 'classifier','weighting','columns_selected','pred_y','act_y','test_index']
    results_for_each_fold = pd.DataFrame(columns = columns_fold_results)

    columns_param_results = ['num_P','num_S','num_Features', 'classifier','weighting','pred_y','act_y','test_index', 'precision_score']
    results_for_each_param = pd.DataFrame(columns = columns_param_results)

    columns_good_results = ['num_P','num_S','num_Features', 'classifier', 'weighting','pred_y','act_y' ,'test_index', 'precision_score']
    good_results = pd.DataFrame(columns = columns_good_results)


    # running grid search
    for param in param_grid:
        start = time.time()
        print('start computing: ', param)

        # create temp dataframe to store the fold results
        fold_result_temp = pd.DataFrame(columns = columns_fold_results)

        # compute 12 folds in parallel and store the combined results
        results = Parallel(n_jobs=-1)(delayed(integrated_grid_search)(X_fold, Y_fold, dataset, inner_train_indexes,
                                                                      inner_test_indexes, fold, param, outer_train_index, weighting)
                                      for fold in range(inner_cv_fold))

        # iterate through the results of 12 folds and append each fold
        for row in results:
            results_for_each_fold = results_for_each_fold.append(row, ignore_index=True)
            fold_result_temp = fold_result_temp.append(row, ignore_index=True)

        # combine inner fold result into outerfold
        cw = [(c, w) for c in range(3) for w in weighting]
        for i in cw:
            combined_fold_df = fold_result_temp[
                (fold_result_temp['classifier'] == i[0]) &
                (fold_result_temp['weighting'] == i[1])
            ]
            row_param = computeParamRowResult(combined_fold_df, param, i)
            if isGoodResult(0.8, row_param.get('precision_score')) == True:
                good_results = good_results.append(row_param, ignore_index=True) 

            results_for_each_param = results_for_each_param.append(row_param, ignore_index=True)
        end = time.time()
        print('Time taken: ', end - start)
        
    results_for_each_fold.to_csv(path_results_for_each_fold)
    results_for_each_param.to_csv(path_results_for_each_param)
    good_results.to_csv(path_good_results)

    results_for_each_fold.to_pickle(pkl_path_results_for_each_fold)
    results_for_each_param.to_pickle(pkl_path_results_for_each_param)
    good_results.to_pickle(pkl_path_good_results)



