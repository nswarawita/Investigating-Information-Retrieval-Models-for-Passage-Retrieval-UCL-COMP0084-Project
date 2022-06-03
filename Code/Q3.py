#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import the required packages
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

np.random.seed(15)

# Load the pre processed train and validation datasets from sub task 2
train_df_subset = pd.read_pickle('train_df_subset.pkl')
train_df_subset.head(2)

validation_df_subset = pd.read_pickle('validation_df_subset.pkl')
validation_df_subset.head(2)

# Create the train and test datasets
# Create the training data
vector_size = 50
x_train = np.zeros((len(train_df_subset),2,vector_size))

for i in range(len(train_df_subset)):
    x_train[i][0] = train_df_subset.iloc[i,7]
    x_train[i][1] = train_df_subset.iloc[i,8]

X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
print(X_train.shape)   

y_train = train_df_subset['relevancy'].values
Y_train = np.reshape(y_train, (len(train_df_subset), 1))
print(Y_train.shape)   

# Create the testing data
x_validation = np.zeros((len(validation_df_subset),2,vector_size))

for i in range(len(validation_df_subset)):
    x_validation[i][0] = validation_df_subset.iloc[i,7]
    x_validation[i][1] = validation_df_subset.iloc[i,8]
    
    
X_validation = np.reshape(x_validation, (x_validation.shape[0], x_validation.shape[1] * x_validation.shape[2])) #np.mean(x_test, axis = 2)
print(X_validation.shape)   
y_validation = validation_df_subset['relevancy'].to_numpy()
Y_validation = np.reshape(y_validation, (len(validation_df_subset), 1))
print(Y_validation.shape)   


# Load data into DMatrices
train_mat = xgb.DMatrix(X_train, label=Y_train)
val_mat = xgb.DMatrix(X_validation, label=Y_validation)

# Create the parameter dictionary
parameters = {
    'max_depth':6,
    'min_child_weight': 1,
    'eta':0.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'rank:pairwise'
}


# Tune the hyperparameters using the cv function in XGBoost
# Tune the parameters that add constraints on the tree architechture
gridsearch_parameters_1 = [(7, 1), (7, 2), (7, 3), (8, 1), (8, 2), (8, 3), (9, 1), (9, 2), (9, 3)]
gridsearch_parameters_1

max_map = 0
best_max_depth = None
best_min_child_weight = None
for max_depth, min_child_weight in gridsearch_parameters_1:
    print(f"CV with max_depth={max_depth}, min_child_weight={min_child_weight}")
   # Update the parameters
    parameters['max_depth'] = max_depth
    parameters['min_child_weight'] = min_child_weight
    
    # Run CV
    cv_results = xgb.cv(parameters, train_mat, seed=15, nfold=10)
    
    # Update best MAP
    mean_map = cv_results['test-map-mean'].max()
    boost_rounds = cv_results['test-map-mean'].argmax()
    print(f" MAP:{mean_map}, for {boost_rounds} rounds")
    
    if mean_map > max_map:
        max_map = mean_map 
        best_max_depth = max_depth
        best_min_child_weight = min_child_weight
print(f"MAP:{max_map},max_depth:{best_max_depth}, min_child_weight:{best_min_child_weight}")

# Update the tuned parameters
parameters['max_depth'] = best_max_depth
parameters['min_child_weight'] = best_min_child_weight

# Tune the parameters that control the sampling that is done at each boosting round
gridsearch_parameters_2 = [(1, 1), (1, 0.9), (1, 0.8), (0.9, 1), (0.9, 0.9), (0.9, 0.8), (0.8, 1), (0.8, 0.9), (0.8, 0.8)]
gridsearch_parameters_2

max_map = 0
best_subsample = None
best_colsample = None
for subsample, colsample in gridsearch_parameters_2:
    print(f"CV with subsample={subsample}, colsample={colsample}")
   # Update the parameters
    parameters['subsample'] = subsample
    parameters['colsample_bytree'] = colsample
    
    # Run CV
    cv_results = xgb.cv(parameters, train_mat, seed=15, nfold=10)
    
    # Update best MAP
    mean_map = cv_results['test-map-mean'].max()
    boost_rounds = cv_results['test-map-mean'].argmax()
    print(f" MAP:{mean_map}, for {boost_rounds} rounds")
    
    if mean_map > max_map:
        max_map = mean_map 
        best_subsample = subsample
        best_colsample = colsample
print(f"MAP:{max_map},subsample:{best_subsample}, colsample:{best_colsample}")

# Update the tuned parameters
parameters['subsample'] = best_subsample
parameters['colsample_bytree'] = best_colsample


# Tune the eta parameter - this controls the learning rate
max_map = 0
best_eta = None
for eta in [0.005, 0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.35]:
    print(f"CV with learning rate={eta}")
   # Update the parameters
    parameters['eta'] = eta
    
    # Run CV
    cv_results = xgb.cv(parameters, train_mat, seed=15, nfold=10)
    
    # Update best MAP
    mean_map = cv_results['test-map-mean'].max()
    boost_rounds = cv_results['test-map-mean'].argmax()
    print(f" MAP:{mean_map}, for {boost_rounds} rounds")
    
    if mean_map > max_map:
        max_map = mean_map 
        best_eta = eta
print(f"MAP:{max_map},eta:{best_eta}")

# Update the tuned learning rate
parameters['eta'] = best_eta


# Train the lambdaMART model using the training matrix
xgb_model = xgb.train(parameters, train_mat)


# Test the model using the validation data and save the results
preds = xgb_model.predict(val_mat)

validation_df_subset['predicted relevance values']=pd.Series(preds)
validation_df_subset.head(5)

t3_results_df = validation_df_subset[['qid', 'pid', 'predicted relevance values']]
t3_results_df.head()

t3_results_df = t3_results_df.sort_values(['qid','predicted relevance values'],ascending=False).groupby('qid').head(len(t3_results_df))
t3_results_df

t3_results_df["rank"] = t3_results_df.groupby("qid")["predicted relevance values"].rank("dense", ascending=False)
t3_results_df

t3_results_df['A1'] = 'A1'
t3_results_df['algorithm name'] = 'LM'
t3_results_df=t3_results_df.dropna()

t3_submission_df = t3_results_df[['qid', 'A1', 'pid', 'rank', 'predicted relevance values', 'algorithm name']]
t3_submission_df

# Save as a txt file
np.savetxt('LM.txt', t3_submission_df.values, fmt='%s')


# Use the average precision and NDCG metric to assess the performance of the logistic regression model based on the validation data

with open('relevant_dictionary.pkl', 'rb') as f:
    relevant_dict = pickle.load(f)
    
rankings_dict = {}
for i in range(len(t3_submission_df)):
    qid = t3_submission_df.iloc[i,0]
    pid = t3_submission_df.iloc[i,2]
    if qid in rankings_dict.keys():
        rankings_dict[qid].append(pid)
    else:
        rankings_dict[qid] = [pid]  
        
# Average precision of the lambdaMART model
def calculate_precision_at_n(r, n):
    '''
    Calculates the precision at the nth position
    '''
    r = np.asarray(r)
    return np.mean(r[:n])

def calculate_average_precision(r):
    '''
    Calculate the average precision of a query
    input r is a list of 0s and 1s
    '''
    val_lst = []
    for i in range(len(r)):
        if r[i] == 1:
            val_lst.append(calculate_precision_at_n(r, i+1))
    if len(val_lst) > 0:        
        vals = np.asarray(val_lst)
        return np.mean(vals)
    else:
        return 0
    
def calculate_mean_average_precision(rankings_dict, relevant_dict):
    precision_sum = 0
    for qid_value in rankings_dict.keys():
        relevent_lst = []               
        retrieved_list = rankings_dict[qid_value]                  # Find the list of retrieved passages
        for pid_val in retrieved_list:        
            if pid_val in relevant_dict[qid_value]:
                relevent_lst.append(1)
            else:
                relevent_lst.append(0)
        precision_sum += calculate_average_precision(relevent_lst) # Add the average precision of the query to the precision_sum
    mean_avg_precision = precision_sum / len(rankings_dict)
    return mean_avg_precision

mean_avg_precision = calculate_mean_average_precision(rankings_dict, relevant_dict)
print(f"The mean average precision of the lambdaMART model is {round(mean_avg_precision,4)}")

# NDCG metric for the lambdaMART model
def calculate_DCG(rel_lst):
    dcg = 0.0
    for i in range(len(rel_lst)):
        rel_score = rel_lst[i]
        if rel_score == 1:
            dcg += 1.0 / np.log2(i + 2)
    return dcg

def calculate_ideal_DCG(ideal_lst):
    ideal_dcg = 0.0
    for j in range(len(ideal_lst)):
        ideal_dcg += 1.0 / np.log2(j + 2)
    return ideal_dcg

def calculate_NDCG(rankings, relevant_dict):
    passage_ndcg = 0 
    for qid_val in rankings.keys(): 
        rel_lst = []
        num_passages_relevant = 0
        retrieved_pids = rankings[qid_val]
        for pid in retrieved_pids:
            if pid in relevant_dict[qid_val]:
                rel_lst.append(1)
                num_passages_relevant += 1
            else:
                rel_lst.append(0)
        assert len(retrieved_pids) == len(rel_lst)
        query_dcg = calculate_DCG(rel_lst)

        ideal_lst = [1] * num_passages_relevant
        query_ideal_dcg = calculate_ideal_DCG(ideal_lst)

        if query_ideal_dcg != 0:
            query_ndcg = query_dcg / query_ideal_dcg
        else:
            query_ndcg = 0

        passage_ndcg += query_ndcg

    avg_ndcg = passage_ndcg / len(rankings)
    return avg_ndcg

avg_ndcg = calculate_NDCG(rankings_dict, relevant_dict)   
print(f"The average NDCG metric for the lambdaMART model is {round(avg_ndcg,4)}") 










