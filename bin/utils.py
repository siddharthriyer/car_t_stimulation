import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import SpectralClustering
from sklearn.metrics import r2_score
    
import copy
import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)


start_col_dict = {0: 6, 8: 4}
markers = {'x': 'o', "A": 'o', "D":'s'}


def get_data(minus_car = 'True'): #function to process the raw .csv files into dataframes 
    days = [0,8]
    dtypes = ['bulk', 'MFI']

    dfs = []
    names = []
    for day in days:
        for dtype in dtypes:
            if minus_car and day == 8: #option to remove CAR expression data which is not relevant
                filename = '../data/d{}_{}_car-.csv'.format(day, dtype)
            else:
                filename = '../data/d{}_{}.csv'.format(day, dtype)
            df = pd.read_csv(filename)
            
            if day == 8:
                df = process_df(df, start_col = start_col_dict[day])
            else:
                df = process_df(df, start_col = start_col_dict[day], scale_by_donor = False)
            dfs.append(df)
            names.append('{}_{}'.format(day, dtype))
           
    return dfs,names


def process_df(df, start_col, scale_by_donor = False): #removes nonfunctional samples and also non APCms/Dynabeads samples
    pd.options.mode.chained_assignment = None
    df = df.loc[lambda df: df['Sample'] != 'x', :]
    df = df.loc[lambda df: df['Culture'] != 'T', :]
    df = df.reset_index(drop = True)
    donors = df['Donor'].dropna().unique()
    scaler= StandardScaler();
    
    if scale_by_donor:
        for donor in donors:
            donor_vals = scaler.fit_transform( df.loc[df['Donor'] == donor, df.columns[start_col:]])
            df.iloc[df['Donor'] == donor, start_col:] = donor_vals
    else:
        df.iloc[:,start_col:] = scaler.fit_transform(df.loc[:,df.columns[start_col:]])
        
    return df
   
############################# Part 1 ####################################    
    
#creating color palette for samples and donors

def produce_color_mapping(df): #produces color mapping for scatterplots
    from seaborn import color_palette
    sample_dict = {}

    samples = df['Sample'].dropna().unique()
    colorsD = color_palette('YlGn', n_colors = 3); #different palettes for transact vs Dynabeads vs APCms
    colorsT = color_palette('OrRd', n_colors = 3)
    colorsA = color_palette('BuPu', n_colors = 7)

    d= 0;t = 0; a =0;
    for i in range(0,len(samples)):
        if samples[i][0] == 'D':
            sample_dict[samples[i]] = colorsD[d];
            d +=1;
        elif samples[i][0] == 'T':
            sample_dict[samples[i]] = colorsT[t];
            t += 1;
        else:
            sample_dict[samples[i]] = colorsA[a];
            a += 1;
    
    donor_dict = {};
    colorsHD= color_palette('OrRd_r', n_colors = 12)
    colorsAC = color_palette('BuPu', n_colors = 12)
    for i in range(1, 11):
        hd = 'HD{}'.format(i)
        ac= 'AC{}'.format(i)
        donor_dict[hd] = colorsHD[i];
        donor_dict[ac] = colorsAC[i];
    return donor_dict, sample_dict

def graph_pca_invitro(df, name, palette_dicts, graph_cols = [ 'Stim', 'Sample','Donor']): #graphs PCA plots of T cell products
    
    pca= decomposition.PCA()
    start_col = start_col_dict[int(name.split('_')[0])]
    pca_mat = pca.fit_transform(df.loc[:,df.columns[start_col:]])[:,:3]
    for comp in range(3):
        col = 'PC{}'.format(comp+1)
        df[col] = pca_mat[:,comp]
    
    # making plot for Fig. 1K
    if '8' in name:
        plt.figure(figsize = (10,10))
        palette_dict = {}
        for i in range(1,11):
            hd = 'HD{}'.format(i)
            ac= 'AC{}'.format(i)
            palette_dict[ac] = '#FBC901' #different hues for ALL/CLL and healthy donors
            palette_dict[hd] = '#808080'
            
        sns.scatterplot(x = 'PC1', y = 'PC2', hue = 'Donor', s= 150, palette = palette_dict,
                            markers = markers, style = 'Culture', data = df)
        
    #ends up making plot for 1L
    for col, palette_dict in zip(graph_cols, palette_dicts):
        if palette_dict != 'none':
            plt.figure(figsize = (10,10))
            sns.scatterplot(x = 'PC1', y = 'PC2', hue = col, s= 150, palette = palette_dict,
                            markers = markers, style = 'Culture', data = df)
        else:
            df[col] = df[col].astype(np.float)
            directed_graph(df, markers, col)
            
            plt.figure(figsize = (10,10))
            sns.scatterplot(x = 'PC1', y = 'PC2', hue = col, s = 150,
                            markers = markers, style = 'Culture', data = df)
            
            
        
        plt.title(name + ' Colored by ' + col)  
        plt.savefig('../results/figures/pca_{}_{}.pdf'.format(name,col),
                    transparent = True)
    
    
    return df

def directed_graph(df, markers, col): #plot a directed graph for each group in 'name' showing trend of stimulation
    donors = df['Donor'].dropna().unique()
    plt.figure(figsize = (10,10))
    sns.scatterplot(x = 'PC1', y = 'PC2', hue = col, s= 150,
                            markers = markers, style = 'Culture', data = df)
    style_dict =  {'D': 'dashed', 'A': 'solid'}
    for culture in ['D', 'A']: 
        culture_df = df.loc[lambda df: df['Culture'] == culture, :]
        for donor in donors:
            donor_df = culture_df.loc[lambda culture_df: culture_df['Donor'] == donor, :]
            donor_mat = donor_df.values
            pc1_col = list(donor_df.columns).index('PC1')
            pc2_col = list(donor_df.columns).index('PC2')
            for i in range(donor_mat.shape[0]):
                start = donor_mat[i:i+2,pc1_col];
                end= donor_mat[i:i+2, pc2_col];
                plt.plot(start, end, linewidth = 2, linestyle = style_dict[culture],
                     color = 'k', alpha = 0.2+ 0.1*i);

def donor_separator(df, day): #separates out ALL/CLL vs healthy donor samples into different dataframes
    ac_count = 0
    hd_count = 0
    
    if day == 8:
        df = df.loc[lambda df: df['Culture'] == 'A', :] #keeping only APCms samples at this point
        df = df.reset_index(drop = True)
        
    ac_df = hd_df = pd.DataFrame(columns = df.columns)
    for i in range(0,df.shape[0]):
        if df['Donor'][i][:2] == 'AC':
            ac_df = ac_df.append(df.iloc[i])
            ac_count+=1;
        else:
            hd_df = hd_df.append(df.iloc[i])
            hd_count+=1;
        
    return ac_df, hd_df 

def classify_ac_hd(df): #train classifiers to separate ALL/CLL vs healthy donor for analysis
    

    start_col = start_col_dict[8]
    end_col = list(df.columns).index('PC1')
    data_cols = list(df.columns)[start_col:end_col] + ['Stim']
    X = df[[col for col in data_cols ]]
    y = ['AC' in donor for donor in df['Donor']]
    clf_names = ['Random Forest', 'MLPC', 'Decision Tree',
                 'Naive Bayes',  'Nearest Neighbors']
    clfs= [RandomForestClassifier(), MLPClassifier(), DecisionTreeClassifier(), 
           GaussianNB(), KNeighborsClassifier()]
    
    scores = []
    for clf_name, clf in zip(clf_names, clfs):
        cv_scores = cross_val_score(clf, X, y)
        scores.append(cv_scores)
        
    scores =np.array(scores)                 
    barplot_std(scores, clf_names, yaxis = 'Cross-Validation Score', 
                title = 'Cross-Validation Scores for Classifying Between Healthy/Diseased Samples')
    
    score_df = pd.DataFrame(scores, columns= clf_names)
    return score_df
   
    
    
    
    
#################   Part 2 ######################







def patient_type_analysis(df, patient_type, grid_search = False):  #analyzes separators of clustering in healthy/diseased donors
    
    #first, find optimal # of clusters
    mat = df[['PC1','PC2']].values
    optimal_n = clustering_optimization(mat)
    
    #then make clusters
    cluster_clf = KMeans(n_clusters = optimal_n)
    df['Cluster'] = cluster_clf.fit_predict(mat)
    col = 'Cluster'
    directed_graph(df, markers, col)
    
 
    
    start_col = start_col_dict[8]
    end_col = list(df.columns).index('PC1')
    data_cols = list(df.columns)[start_col:end_col] + ['Stim']
    X = df[[col for col in data_cols ]]
    y = df['Cluster']
    X = X.values
    score_df= clf_analysis(X, y, grid_search, data_cols, patient_type, dataset='d8_cluster_classify')
 
    return score_df

def clf_analysis(X, y, grid_search, data_cols, patient_type, dataset): #training classifiers
    
     #classifier names
    clf_names = ['Random Forest', 'MLPC', 'Decision Tree',
                 'Naive Bayes',  'Nearest Neighbors']
    clfs= [RandomForestClassifier(), MLPClassifier(), DecisionTreeClassifier(), 
           GaussianNB(), KNeighborsClassifier()]
                 
                 
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3) #separating out a test set
    
    if dataset == 'd8_cluster_classify': #different plot titles
        data_task = 'Predicting Cluster from Day 8 data'
    elif dataset == 'd8_cluster_predict':
        data_task = 'Predicting Cluster from Day 0 data and Stimulation'
        
    score_df = pd.DataFrame() #dataframe to store the scores
    
    if grid_search: #optional parameter tuning
        
        best_params = {}
        for clf_name, clf in zip(clf_names, clfs):
            print('Doing grid search on {}...'.format(clf_name))
            grid = get_grid_clf(clf_name)
            print(grid)
            gs = GridSearchCV(estimator = clf, param_grid = grid)
            gs.fit(X_train, y_train)
            clf.set_params(**gs.best_params_) 
            best_params[clf_name] = gs.best_params_
            
            print('Done!')
            
            
        print('Done with all parameter tuning')
        with open('../results/model_params/tuned_params_{}_{}_clf.pickle'.format(patient_type, dataset), 'wb') as handle:
            pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            

    val_scores  = []  
    test_scores = []

    #get the tuned parameters
    with open('../results/model_params/tuned_params_{}_{}_clf.pickle'.format(patient_type, dataset), 'rb') as handle:
        best_params = pickle.load(handle) 
     
    #iterate though classifiers and get the test set score as well as cross-validation scores
    for clf_name, clf in zip(clf_names, clfs):
        clf.set_params(**best_params[clf_name])
        cv_scores = cross_val_score(clf, X_train, y_train)
        val_scores.append(cv_scores)

        clf.fit(X_train, y_train)
        test_score= clf.score(X_test, y_test)
        test_scores.append(test_score)
            
        #get feature importances from trained Random Forest model
        if clf_name == 'Random Forest':
            visualize_patient_clf_features(clf, X_train, y_train, data_cols,
                                           len(data_cols), patient_type)

    val_scores = np.array(val_scores) 
    
    #plot the scores
    barplot_std(val_scores, clf_names, yaxis = 'Cross-Validation Scores', 
            title = 'Cross-Validation Scores - {} - {} Donors '.format(data_task, patient_type))

    test_scores = np.array(test_scores)                 
    barplot_std(test_scores, clf_names, yaxis = 'Test Set Scores', 
            title = 'Test Set Scores - {} - {} Donors '.format(data_task, patient_type), stdev = False)

    score_df = pd.DataFrame(val_scores, columns= clf_names)
        
    return score_df  
        
    
             
def get_grid_clf(clf_name): #initializes parameter grid for grid search
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 4)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 4)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    
    activation = ['identity', 'logistic', 'tanh', 'relu']
    hidden_layer_sizes = [50,100,200]
    solver = ['lbfgs', 'sgd', 'adam']
    alpha = [0.00001, 0.0001, 0.001]
    
    var_smoothing = [1e-10, 1e-9, 1e-8]
    
    knn_algo = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size = [20,30,40]
    p = [1,2]
    
    features = [n_estimators, max_features, max_depth, min_samples_split,
                         min_samples_leaf, bootstrap, 
                         activation, hidden_layer_sizes, solver, alpha,
                         var_smoothing,
                        knn_algo, leaf_size, p]
    
    feature_names = ['n_estimators', 'max_features', 'max_depth', 
                     'min_samples_split', 'min_samples_leaf', 'bootstrap',
                     'activation', 'hidden_layer_sizes', 'solver', 'alpha',
                     'var_smoothing',
                     'algorithm', 'leaf_size', 'p']
    
    features_i = {'Random Forest' : np.arange(0,6),
                  'MLPC': np.arange(6,10),
                  'Decision Tree': np.arange(1,5),
                  'Naive Bayes' : [10],
                  'Nearest Neighbors': np.arange(11,14)}
    
    random_grid = {feature_names[i]: features[i] for i in features_i[clf_name]}
    return random_grid
                   
    
def clustering_optimization(mat): #finds optimal number of clusters
    SSE= [];
    SSE_del= []; #empty arrays to store SSE and SSE_del

    for i in range (1,30): #testing up to 30 clusters
        estimator = KMeans(n_clusters=i); #initializing estimator
        estimator.fit(mat); #fitting estimator and storing inertia
        SSE.append(estimator.inertia_) ;
        if i > 1:
            SSE_del.append(SSE[i-1]-SSE[i-2]); #storing change in SSE

    plt.figure(figsize = (12,8))
    plt.plot(range(1,30), SSE); #showing SSE 
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE');


    # fullpath = PATH + '/SSE-kmeans.png';
    # plt.savefig(fullpath, transparent = True)

    plt.figure(figsize = (12,8));
    plt.plot(range(2,30), SSE_del); #showing change in SSE
    plt.xlabel('Number of clusters');
    plt.ylabel('SSE_delta');
    
    for i in range(1, len(SSE_del)):
        if (SSE_del[i] - SSE_del[i-1]) < 10:   
            optimal_n = i
            break
            
    return optimal_n


def barplot_std(data, col_names, yaxis, title, stdev = True): #makes labeled bar plot from an array showing mean/stdev
    plt.figure(figsize = (12,5))
     
    if stdev:
        avg = np.mean(data, axis= 0)
        stdev = np.std(data, axis= 0)
        xx = np.arange(data.shape[1])
        plt.bar(x = xx, height = avg, yerr = stdev)
        
    else:
        xx = np.arange(data.size)
        plt.bar(x = xx, height = data)
    
    plt.ylabel(yaxis)
    plt.xticks(xx, col_names)
        
        
def visualize_patient_clf_features(clf, X, y,features, n_features, patient_type, n_iters = 50):
    
    #makes plot of feature importances
    feat_impts = np.zeros((n_features, n_iters))
                
    for i in range(n_iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)  
        clf.fit(X_train,y_train)
        feat_impts[:,i] = clf.feature_importances_
    barplot_std(feat_impts.T, features, yaxis = 'Feature Importance', title ='{} Donor Feature Importances for Clustering'.format(patient_type))
    

def make_design_mats(d0_df, d8_df): #creates design matrices from dataframes to use as inputs/outputs in regression/classification
    new_df =  copy.deepcopy(d8_df[[col for col in list(d8_df.columns)[:start_col_dict[8]]]])

    
    donors = new_df['Donor'].dropna().unique()
    donor2rows = {donor: [] for donor in donors}
    for i in range(new_df.shape[0]):
        row = new_df.iloc[i]
        donor2rows[row['Donor']].append(i)

    start_col_0= start_col_dict[0]
    end_col_0 = list(d0_df.columns).index('PC1')
    day_0_cols = list(d0_df.columns)[start_col_0:end_col_0]
    
    d0_mat = np.zeros((new_df.shape[0],end_col_0-start_col_0))
    for i in range(d0_df.shape[0]):
        current_row = d0_df.iloc[i]
        if current_row['Donor'] in donor2rows:
            new_j = donor2rows[current_row['Donor']]
            for j in new_j:
                for ii, col in enumerate(day_0_cols):
                    d0_mat[j,ii] = current_row[col]
                    
    start_col_8 = start_col_dict[8]
    end_col_8 = list(d8_df.columns).index('PC1')
    
    
    d8_mat = d8_df[[col for col in list(d8_df.columns)[start_col_8:end_col_8]]]            
    return d0_mat, d8_mat, [start_col_0, end_col_0], [start_col_8, end_col_8], donor2rows
    

def predict_final_cluster(d0_df, d8_df, patient_type, grid_search = False, use_only_stim = False): #predicts final cluster using functions defined above
    
    d0_mat, d8_mat, [start_col_0, end_col_0], [start_col_8, end_col_8], donor2rows = make_design_mats(d0_df, d8_df)
    
    
    mat = d8_df[['PC1','PC2']].values
    optimal_n = clustering_optimization(mat)
    
    #then make clusters
    cluster_clf = KMeans(n_clusters = optimal_n)
    
    clusters = cluster_clf.fit_predict(mat)
    #first, find optimal # of clusters

    if use_only_stim:
        X = np.array(d8_df['Stim']).reshape(-1,1)
    else:
        X = np.hstack((d0_mat, np.array(d8_df['Stim']).reshape(-1,1)))
    y = clusters

    start_col_0= start_col_dict[0]
    end_col_0 = list(d0_df.columns).index('PC1')
    day_cols = list(d0_df.columns)[start_col_0:end_col_0]
    
    data_cols = day_cols + ['Stim']
    
    
        
    score_df= clf_analysis(X, y, grid_search, data_cols, patient_type, dataset='d8_cluster_predict')
    
    
    
    
################################ Part 3 ############################################

def holdout(X,y,rows): #making defined train/test sets from a specific set of rows to hold out
    X_train = np.zeros((X.shape[0]-len(rows), X.shape[1]))
    X_test = np.zeros((len(rows), X.shape[1]))
    y_train = np.zeros((X.shape[0]-len(rows), 1))
    y_test = np.zeros((len(rows), 1))

    test_i = 0
    train_i = 0
    for i in range(X.shape[0]):
        if i in rows:
            X_test[test_i,:] = copy.deepcopy(X[i,:])
            y_test[test_i] = copy.deepcopy(y[i])
            test_i += 1
        else:
            X_train[train_i,:] = copy.deepcopy(X[i,:])
            y_train[train_i] = copy.deepcopy(y[i])
            train_i += 1

    
    return X_train, X_test, y_train, y_test
        
def donor_holdout_test(rg, X, y, donor2rows): #iteratively trains and tests on held out donors for a given regressor
    
    scores = np.zeros(len(list(donor2rows.keys())))
    max_len = np.max(np.array([len(value) for value in donor2rows.values()]))
    predictions = np.zeros((len(list(donor2rows.keys())),max_len))
    actuals = np.zeros((len(list(donor2rows.keys())),max_len))
    
    sample_nums = []
    sample_donors =[]
    
    for i, donor in enumerate(donor2rows):
        
        if donor == 'AC4': #this donor only has one sample
            continue
        rows = donor2rows[donor]
        X_train, X_test, y_train, y_test = holdout(X,y,rows)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        rg.fit(X_train, y_train)
        
        order = np.argsort(y_test)
        y_test= np.sort(y_test)                   
        X_test = np.array([X_test[j] for j in order])           
        y_pred = rg.predict(X_test)
                            
        scores[i] = rg.score(X_test,y_test)
        predictions[i,:y_pred.size] = y_pred
        actuals[i,:y_pred.size] = y_test
        
        sample_nums.append(i)
        sample_donors.append(donor)          
            
    return scores, predictions, actuals, sample_nums, sample_donors


def get_grid_reg(reg_name): #gets grid for grid search to tune regressor parameters
    
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 4)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num = 4)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]
    
    alpha = [int(x) for x in np.logspace(1, 4, num = 4)]
    max_iter = [int(x) for x in np.logspace(3, 5, num = 4)]
    fit_intercept = [False, True]
    
    activation = ['identity', 'logistic', 'tanh', 'relu']
    hidden_layer_sizes = [50,100,200]
    solver = ['lbfgs', 'sgd', 'adam']
    alpha = [0.00001, 0.0001, 0.001]
    
    features = [n_estimators, max_features, max_depth, min_samples_split,
                         min_samples_leaf, bootstrap, 
                         alpha, max_iter,
                         fit_intercept,
                        activation, hidden_layer_sizes, solver, alpha]
    
    feature_names = ['n_estimators', 'max_features', 'max_depth', 
                     'min_samples_split', 'min_samples_leaf', 'bootstrap',
                     'alpha', 'max_iter', 'fit_intercept',
                    'activation', 'hidden_layer_sizes', 'solver', 'alpha']
    
    features_i = {'Random Forest Regression' : np.arange(0,6),
                  'Linear Regression': np.arange(8,9),
                  'Lasso Regression': np.arange(6,9),
                  'Ridge Regression' : np.arange(6,9),
                  'MLP Regression': np.arange(9,13)}
    
    random_grid = {feature_names[i]: features[i] for i in features_i[reg_name]}
    return random_grid
                   
    
def predict_stimulation_directly(dfs, patient_type, grid_search = False): 
    
    # predicting desired stimulation given an input and output phenotype
    
    d0_df, d8_df = dfs
    d0_mat, d8_mat, [start_col_0, end_col_0], [start_col_8, end_col_8], donor2rows = make_design_mats(d0_df, d8_df)
    
    
    X = np.hstack((d0_mat, d8_mat))
    y = d8_df['Stim'].values.astype(np.float)        
    rg_names = ['Random Forest Regression', 'Ridge Regression', 'Lasso Regression', 'MLP Regression']
    regressors = [RandomForestRegressor(), Ridge(), Lasso(), MLPRegressor()]
    
    if patient_type == 'Healthy':
        holdout_donors = ['HD7','HD8', 'HD6']
    elif patient_type == 'Diseased':
        holdout_donors = ['AC7','AC8', 'AC10']
        
    holdout_donor_rows = []
    for donor in holdout_donors:
        holdout_donor_rows += donor2rows[donor]
        
    X_train, X_test, y_train, y_test = holdout(X,y, holdout_donor_rows)
    
    
    if grid_search:
  
        best_params = {}
        
        for rg_name, rg in zip(rg_names, regressors):
            print('Doing grid search on {}...'.format(rg_name))
            grid = get_grid_reg(rg_name)
            print(grid)
            gs = GridSearchCV(estimator = rg, param_grid = grid)
            gs.fit(X_train, y_train)
            rg.set_params(**gs.best_params_) 
            best_params[rg_name] = gs.best_params_
            
            print('Done!')
            
            
        print('Done with all parameter tuning')
        with open('../results/model_params/tuned_params_{}_reg.pickle'.format(patient_type), 'wb') as handle:
            pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    else:
        with open('../results/model_params/tuned_params_{}_reg.pickle'.format(patient_type), 'rb') as handle:
            best_params = pickle.load(handle)
        test_scores = []
        single_scores = []
        for rg_name, rg in zip(rg_names, regressors):
            
            plt.figure()
            rg.set_params(**best_params[rg_name])
            rg.fit(X_train, y_train)
            
            test_scores.append(rg.score(X_test,y_test))
            
            new_df = pd.DataFrame()
            actual_list = []; predict_list = []; donor_list = [];


            scores, predictions, actuals, sample_nums, sample_donors = donor_holdout_test(rg, X, y, donor2rows)
            
            
            single_scores.append(np.mean(scores))
            
            i2donor = {i: donor for i, donor in zip(sample_nums, sample_donors)}

            plt.title('Actual vs Predicted Stimulation - {}'.format(rg_name))
            plt.xlabel('Actual Stimulation')
            plt.ylabel('Predicted Stimulation')

            for i in range(predictions.shape[0]):
                if i in sample_nums:
                    plt.plot(actuals[i,:], predictions[i,:])
                    for j in range(predictions.shape[1]):
                        actual_list.append(actuals[i,j])
                        predict_list.append(predictions[i,j])
                        donor_list.append(i2donor[i])

            plt.legend(sample_donors)
            plt.savefig('../results/figures/Actual vs Predicted Stimulation - {}.png'.format(rg_name))       
            new_df['Donor'] = donor_list
            new_df['Actual Stimulation'] = actual_list
            new_df['Predicted Stimulation'] = predict_list

            new_df.to_csv('../results/predictions/{} - direct_stimulation_prediction.csv'.format(rg_name))
            
        return test_scores, single_scores
        
