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

import pickle
import warnings
warnings.simplefilter('ignore', UserWarning)


start_col_dict = {0: 6, 8: 4}
markers = {'x': 'o', "A": 'o', "D":'s'}


def get_data(minus_car = 'True'):
    days = [0,8]
    dtypes = ['bulk', 'MFI']

    dfs = []
    names = []
    for day in days:
        for dtype in dtypes:
            if minus_car and day == 8:
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


def process_df(df, start_col, scale_by_donor = False):
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
    
#creating color palette for samples and donors

def produce_color_mapping(df):
    from seaborn import color_palette
    sample_dict = {}

    samples = df['Sample'].dropna().unique()
    colorsD = color_palette('YlGn', n_colors = 3);
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

def graph_pca_invitro(df, name, palette_dicts, graph_cols = [ 'Stim', 'Sample','Donor']):
    
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
            palette_dict[ac] = '#FBC901'
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

def directed_graph(df, markers, col): #plot a directed graph for each group in 'name'
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

def donor_separator(df, day):
    ac_count = 0
    hd_count = 0
    
    if day == 8:
        df = df.loc[lambda df: df['Culture'] == 'A', :]
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

def classify_ac_hd(df):
    clf = RandomForestClassifier(n_estimators = 100)
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
                      
#################


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

def clf_analysis(X, y, grid_search, data_cols, patient_type, dataset):
    
     #classifier names
    clf_names = ['Random Forest', 'MLPC', 'Decision Tree',
                 'Naive Bayes',  'Nearest Neighbors']
    clfs= [RandomForestClassifier(), MLPClassifier(), DecisionTreeClassifier(), 
           GaussianNB(), KNeighborsClassifier()]
                 
                 
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3)
    
    if dataset == 'd8_cluster_classify':
        data_task = 'Predicting Cluster from Day 8 data'
    elif dataset == 'd8_cluster_predict':
        data_task = 'Predicting Cluster from Day 0 data and Stimulation'
        
        
    if grid_search:
        
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
            
            
    else:
        val_scores  = []  
        test_scores = []
        for clf_name, clf in zip(clf_names, clfs):
            cv_scores = cross_val_score(clf, X_train, y_train)
            val_scores.append(cv_scores)
            
            clf.fit(X_train, y_train)
            test_score= clf.score(X_test, y_test)
            test_scores.append(test_score)
            
            if clf_name == 'Random Forest':
                visualize_patient_clf_features(clf, X_train, y_train, data_cols,
                                               len(data_cols), patient_type)
                
        val_scores = np.array(val_scores)                 
        barplot_std(val_scores, clf_names, yaxis = 'Cross-Validation Scores', 
                title = 'Cross-Validation Scores - {} - {} Donors '.format(data_task, patient_type))
        
        test_scores = np.array(test_scores)                 
        barplot_std(test_scores, clf_names, yaxis = 'Test Set Scores', 
                title = 'Test Set Scores - {} - {} Donors '.format(data_task, patient_type), stdev = False)
        
        score_df = pd.DataFrame(val_scores, columns= clf_names)
        
    return score_df  
        
    
                  
def get_grid_clf(clf_name):
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
                     'knn_algo', 'leaf_size', 'p']
    
    features_i = {'Random Forest' : np.arange(0,6),
                  'MLPC': np.arange(6,10),
                  'Decision Tree': np.arange(1,5),
                  'Naive Bayes' : [10],
                  'Nearest Neighbors': np.arange(11,14)}
    
    random_grid = {feature_names[i]: features[i] for i in features_i[clf_name]}
    return random_grid
                   
    
def clustering_optimization(mat):
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


def barplot_std(data, col_names, yaxis, title, stdev = True):
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
    feat_impts = np.zeros((n_features, n_iters))
                
    for i in range(n_iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)  
        clf.fit(X_train,y_train)
        feat_impts[:,i] = clf.feature_importances_
    barplot_std(feat_impts.T, features, yaxis = 'Feature Importance', title ='{} Donor Feature Importances for Clustering'.format(patient_type))
    
################################

def make_design_mats(d0_df, d8_df):
    
    import copy
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
    

def predict_final_cluster(d0_df, d8_df, patient_type, grid_search = False, use_only_stim = False):
    
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
    
    
    
    
###########

def predict_stimulation_directly(dfs, patient_type, grid_search = False):

    d0_df, d8_df = dfs
    d0_mat, d8_mat, [start_col_0, end_col_0], [start_col_8, end_col_8], donor2rows = make_design_mats(d0_df, d8_df)
    

    X = np.hstack((d0_mat, d8_mat))
    y = d8_df['Stim'].values.astype(np.float)        
    
    if grid_search:
        
        rg_names = ['Random Forest Regression', 'Ridge Regression', 'Lasso Regression', 'Linear Regression', 'MLP Regression']
        regressors = [RandomForestRegressor(), Ridge(), Lasso(), MLPRegressor()]
    
        if patient_type == 'Healthy':
            holdout_donors = ['HD7','HD8', 'HD9']
        elif patient_type == 'Diseased':
            holdout_donors = ['AC7','AC8', 'AC9']
        
        holdout_donor_rows = []
        for donor in holdout_donors:
            holdout_donor_rows += donor2rows[donor]
            
        X_train, X_test, y_train, y_test = holdout(X,y, holdout_donor)
        
        best_params = {}
        
        for rg_name, rg in zip(rg_names, regressors):
            print('Doing grid search on {}...'.format(rg_name))
            grid = get_grid_reg(rg_name)
            print(grid)
            gs = GridSearchCV(estimator = rg, param_grid = grid)
            gs.fit(X_train, y_train)
            clf.set_params(**gs.best_params_) 
            best_params[clf_name] = gs.best_params_
            
            print('Done!')
            
            
        print('Done with all parameter tuning')
        with open('../results/model_params/tuned_params_{}_{}_reg.pickle'.format(patient_type, dataset), 'wb') as handle:
            pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    else:
        rg_names = ['Random Forest Regression', 'Ridge Regression', 'Lasso Regression', 'Linear Regression', 'MLP Regression']
        regressors = [RandomForestRegressor(), Ridge(), Lasso(), MLPRegressor()]
        for rg_name, rg in zip(rg_names, regressors):
            new_df = pd.DataFrame()
            actual_list = []; predict_list = []; donor_list = [];


            scores, predictions, actuals, diseased, diseased_donors = donor_holdout_test(rg, X, y, donor2rows)
            i2donor = {i: donor for i, donor in zip(diseased, diseased_donors)}

            plt.title('Actual vs Predicted Stimulation - {}'.format(rg_name))
            plt.xlabel('Actual Stimulation')
            plt.ylabel('Predicted Stimulation')




            print(rg_name, '- Donor Holdout Score:', np.mean(scores))
            for i in range(predictions.shape[0]):
                if i in diseased:
                    plt.plot(actuals[i,:], predictions[i,:])
                    for j in range(predictions.shape[1]):
                        actual_list.append(actuals[i,j])
                        predict_list.append(predictions[i,j])
                        donor_list.append(i2donor[i])

            plt.legend(diseased_donors)
            plt.savefig('../results/figures/Actual vs Predicted Stimulation - {}.png'.format(rg_name))       
            new_df['Donor'] = donor_list
            new_df['Actual Stimulation'] = actual_list
            new_df['Predicted Stimulation'] = predict_list

            new_df.to_csv('../results/predictions/{} - direct_stimulation_prediction.csv'.format(rg_name))
    
def holdout(X,y,rows):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(X.shape[0]):
        if i in rows:
            X_test.append(X[i,:])
            y_test.append(y[i])
        else:
            X_train.append(X[i,:])
            y_train.append(y[i])
            
    X_train= np.array(X_train)
    y_train =np.array(y_train)
    X_train= np.array(X_test)
    y_train =np.array(y_test)
    
    return X_train, X_test, y_train, y_test
        
def donor_holdout_test(rg, X, y, donor2rows):
    scores = np.zeros(len(list(donor2rows.keys())))
    max_len = np.max(np.array([len(value) for value in donor2rows.values()]))
    predictions = np.zeros((len(list(donor2rows.keys())),max_len))
    actuals = np.zeros((len(list(donor2rows.keys())),max_len))
    diseased = []
    diseased_donors =[]
    for i, donor in enumerate(donor2rows):
        rows = donor2rows[donor]
        X_train, X_test, y_train, y_test = holdout(X,y,rows)
        rg.fit(X_train, y_train)
        order = np.argsort(y_test)
        y_test= np.sort(y_test)                   
        X_test = np.array([X_test[j] for j in order])                
        y_pred = rg.predict(X_test)
                            
        scores[i] = r2_score(y_pred, y_test)
        predictions[i,:y_pred.size] = y_pred
        actuals[i,:y_pred.size] = y_test
                      
        if 'AC' in donor:
            diseased.append(i)
            diseased_donors.append(donor)          
            
    return scores, predictions, actuals, diseased, diseased_donors


def get_grid_reg(reg_name):
    
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
    normalize = [False, True]
    
    activation = ['identity', 'logistic', 'tanh', 'relu']
    hidden_layer_sizes = [50,100,200]
    solver = ['lbfgs', 'sgd', 'adam']
    alpha = [0.00001, 0.0001, 0.001]
    
    features = [n_estimators, max_features, max_depth, min_samples_split,
                         min_samples_leaf, bootstrap, 
                         alpha, max_iter,
                         fit_intercept, normalize,
                        activation, hidden_layer_sizes, solver, alpha]
    
    feature_names = ['n_estimators', 'max_features', 'max_depth', 
                     'min_samples_split', 'min_samples_leaf', 'bootstrap',
                     'alpha', 'max_iter', 'fit_intercept', 'normalize',
                    'activation', 'hidden_layer_sizes', 'solver', 'alpha']
    
    features_i = {'Random Forest Regression' : np.arange(0,6),
                  'Linear Regression': np.arange(8,10),
                  'Lasso Regression': np.arange(6,10),
                  'Ridge Regression' : np.arange(6,10),
                  'MLP Regression': np.arange(10,14)}
    
    random_grid = {feature_names[i]: features[i] for i in features_i[reg_name]}
    return random_grid
                   