# -*- coding: utf-8 -*-
# Import the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

np.random.seed(15)

# Load the pickled train_df_subset and validation_df_subset
train_df_subset = pd.read_pickle('train_df_subset.pkl')
train_df_subset

validation_df_subset = pd.read_pickle('validation_df_subset.pkl')
validation_df_subset


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


# Implement the logistic regression
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def train_logistic_model(X_train, Y_train, num_iterations, lr):
    n_train,d = X_train.shape

    # Initialise w and b to 0
    w = np.random.rand(d,1)
    b = np.random.randint(low=-10, high=10)

    costs = []
    for i in range(num_iterations):
        # Forward pass - calculate the cost function (J)
        a = np.dot(X_train, w) + b         # Multiply w with the feature values and sum them up
        h = sigmoid(a)             # Transform a using the sigmoid function
        J = (-1/n_train) * np.sum((Y_train * np.log(h)) + ((1 - Y_train) * np.log(1 - h))) # Calculate the cost function
        costs.append(J)
        
        # Backward pass - calculate the gradients
        dw =  (1/n_train) * np.dot(X_train.T, (h - Y_train))   # Calculate the gradient of the loss wrt weights
        db = (1/n_train) * np.sum(h - Y_train)                 # Calculate the gradient of the loss wrt bias

        w = w - lr * dw # Update w
        b = b - lr * db # Update b

    return w, b, costs

def test_logistic_model(X_test, w, b): 
    n_train = X_test.shape[0]       
    Y_preds = np.zeros((n_train,1))

    y_hat = sigmoid(np.dot(X_test, w) + b) # Calculate y_hat
    
    for i in range(n_train):
        if y_hat[i][0] >= 0.5:
            Y_preds[i] = 1
        else:
            Y_preds[0] = 1
    return y_hat, Y_preds
            
# Train the logistic regression using the training data
W, B, cost = train_logistic_model(X_train=X_train, Y_train=Y_train, num_iterations=1000, lr=0.1)


# Test the model using the validation data and save the results
rel_probabilities, preds = test_logistic_model(X_validation, W, B)

rel_probabilities = np.reshape(rel_probabilities, (rel_probabilities.shape[0],))
validation_df_subset['predicted probability of relevance']=pd.Series(rel_probabilities)
validation_df_subset.head(5)

t2_results_df = validation_df_subset[['qid', 'pid', 'predicted probability of relevance']]
t2_results_df.head()

t2_results_df = t2_results_df.sort_values(['qid','predicted probability of relevance'],ascending=False).groupby('qid').head(len(t2_results_df))
t2_results_df

t2_results_df["rank"] = t2_results_df.groupby("qid")["predicted probability of relevance"].rank("dense", ascending=False)
t2_results_df

t2_results_df['A1'] = 'A1'
t2_results_df['algorithm name'] = 'LR'
t2_results_df=t2_results_df.dropna()

t2_submission_df = t2_results_df[['qid', 'A1', 'pid', 'rank', 'predicted probability of relevance', 'algorithm name']]
t2_submission_df


# Save as a txt file
#np.savetxt('LR.txt', t2_submission_df.values, fmt='%s')


# Use the average precision and NDCG metric to assess the performance of the logistic regression model based on the validation data
with open('relevant_dictionary.pkl', 'rb') as f:
    relevant_dict = pickle.load(f)

rankings_dict = {}
for i in range(len(t2_submission_df)):
    qid = t2_submission_df.iloc[i,0]
    pid = t2_submission_df.iloc[i,2]
    if qid in rankings_dict.keys():
        rankings_dict[qid].append(pid)
    else:
        rankings_dict[qid] = [pid]  
        
# Average precision of the logistic regression model
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
print(f"The mean average precision of the logistic regression model is {round(mean_avg_precision,4)}")      


# NDCG metric for the logistic regression model
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
print(f"The average NDCG metric for the logistic regression model is {round(avg_ndcg,4)}")  


# Implement the effect of the learning rate on the model's training loss
W1, B1, cost1 = train_logistic_model(X_train=X_train, Y_train=Y_train, num_iterations=1000, lr=100)
W2, B2, cost2 = train_logistic_model(X_train=X_train, Y_train=Y_train, num_iterations=1000, lr=10)
W3, B3, cost3 = train_logistic_model(X_train=X_train, Y_train=Y_train, num_iterations=1000, lr=5)
W4, B4, cost4 = train_logistic_model(X_train=X_train, Y_train=Y_train, num_iterations=1000, lr=1)
W5, B5, cost5 = train_logistic_model(X_train=X_train, Y_train=Y_train, num_iterations=1000, lr=0.1)
W6, B6, cost6 = train_logistic_model(X_train=X_train, Y_train=Y_train, num_iterations=1000, lr=0.01)
W7, B7, cost7 = train_logistic_model(X_train=X_train, Y_train=Y_train, num_iterations=1000, lr=0.001)

num_iterations_lst = list(range(0, 1000))

fig, ax = plt.subplots()
ax.plot(num_iterations_lst, cost1, label='lr=100')
ax.plot(num_iterations_lst, cost2, label='lr=10')
ax.plot(num_iterations_lst, cost3, label='lr=5')
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Training loss')
ax.set_title("Effect of the learning rate on the model's training loss")
ax.legend()

fig, ax = plt.subplots()
ax.plot(num_iterations_lst, cost4, label='lr=1')
ax.plot(num_iterations_lst, cost5, label='lr=0.1')
ax.plot(num_iterations_lst, cost6, label='lr=0.01')
ax.plot(num_iterations_lst, cost7, label='lr=0.001')
ax.set_xlabel('Number of iterations')
ax.set_ylabel('Training loss')
ax.set_title("Effect of the learning rate on the model's training loss")
ax.legend()





















