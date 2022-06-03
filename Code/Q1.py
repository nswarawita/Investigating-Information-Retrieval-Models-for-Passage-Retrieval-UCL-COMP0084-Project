#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import the required packages
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('english') 
import pickle

# Import the validation data set
validation_df = pd.read_csv("../data/validation_data.tsv",sep='\t')
validation_df 

# Create a subset of the validation data
validation_subset_size = 50000
# Select all the rows that have a relevancy score of 1
df1 = validation_df[validation_df.relevancy!=0] 
# Select the rows of the dataset that have a relevancy score of 0
df2 = validation_df[validation_df.relevancy==0].head(validation_subset_size - len(df1))
# Combine the datasets to form a subset of the data
frames = [df1, df2]
validation_df_subset = pd.concat(frames)
validation_df_subset


# Pre-process the data
# (1) Convert both queries and passages to lowercase - 10s
validation_df_subset['pre-processed queries'] = validation_df_subset['queries'].str.lower()
validation_df_subset['pre-processed passage'] = validation_df_subset['passage'].str.lower()

# (2) Strip off punctuation - 30s
validation_df_subset['pre-processed queries'] = validation_df_subset['pre-processed queries'].str.replace(r'[^\w\s]+', '')
validation_df_subset['pre-processed passage'] = validation_df_subset['pre-processed passage'].str.replace(r'[^\w\s]+', '')

# (3) Tokenise - 30s
validation_df_subset['pre-processed queries'] = validation_df_subset['pre-processed queries'].str.split()
validation_df_subset['pre-processed passage'] = validation_df_subset['pre-processed passage'].str.split()

# (4) Stopword removal
validation_df_subset['pre-processed queries'] = validation_df_subset['pre-processed queries'].apply(lambda row: [word for word in row if word not in nltk_stopwords])
validation_df_subset['pre-processed passage'] = validation_df_subset['pre-processed passage'].apply(lambda row: [word for word in row if word not in nltk_stopwords])

# (5) Stemming
# Create a function for stemming
def stemming_fn(sentence):
    new_sentence = [porter_stemmer.stem(word) for word in sentence]
    return new_sentence
validation_df_subset['pre-processed queries'] = validation_df_subset['pre-processed queries'].apply(lambda row:stemming_fn(row))
validation_df_subset['pre-processed passage'] = validation_df_subset['pre-processed passage'].apply(lambda row:stemming_fn(row))


# Number of unique passages (sample size)
N = validation_df_subset['pid'].nunique() 
print(f'N = {N}')


# Create an inverted index to store the number of times a word occurs in a given passage
inverted_index = {}

for i in range(len(validation_df_subset.index)):        
    pid = validation_df_subset.iloc[i,1]                         # Find the pid corresponding to the passage
    words_in_passage = validation_df_subset.iloc[i,6]            # Find the pre processed words in the passage

    for word in words_in_passage:                          
        if word in inverted_index.keys():
            if pid in inverted_index[word]:
                inverted_index[word][pid] += 1
            else:
                inverted_index[word][pid] = 1                     
        else: 
            inverted_index[word] = {} 
            inverted_index[word][pid] = 1
    
            
# Create a dictionary to store the qid and pid values of relevant docs (keys are the qids and values are the pids)
relevant_dict = {}
for i in range(len(validation_df_subset.index)): 
    relevancy =  validation_df_subset.iloc[i,4]
    if relevancy == 1:
        qid = validation_df_subset.iloc[i,0]
        pid = validation_df_subset.iloc[i,1]
        if qid in relevant_dict.keys():
            relevant_dict[qid].append(pid)
        else:
            relevant_dict[qid] = [pid]

# Save relevant dict as it is used in other subtasks
#file = open("relevant_dictionary.pkl","wb") # create a binary pickle file 
#pickle.dump(relevant_dict,file)             # write the dict to the created pickle file
#file.close()   


# Calculate if all queries have relevant passages
len(relevant_dict) # Not all queries have relevant passages


# Create a function to calculate the bm25 score
def calculate_r(word,qid):
    try:
        relevant_passages_list = relevant_dict[qid] 
        passages_contain_word = set(inverted_index[word])
        r = len(set(relevant_passages_list).intersection(passages_contain_word))
    except KeyError:
        r = 0
    return r

def bm25_score(query,passage,qid,K,R,k1=1.2,k2=100,b=0.75):
    '''
    pid - int, unique passage identifier
    passage - list of words, pre processed passage
    qid - int, unique query identifier
    query - list of words, pre processed query
    '''
    bm25_score = 0   
    for word in query:
        try:
            n = len(inverted_index[word])              # Number of passages that contain the word
        except KeyError:
            n = 0
        
        qf = query.count(word)                         # Number of times the word occurs in the given query 
        f = passage.count(word)                        # Number of times the word occurs in the given passage         
        r = calculate_r(word,qid)                  # Number of relevant passages that contain the word
        
        term1 = ( (r + 0.5) * (N - n - R + r + 0.5) ) / ( (n - r + 0.5) * (R - r + 0.5) )
        term2 = ( (k1 + 1) * f ) / (K + f)
        term3 = ( (k2 + 1) * qf ) / (k2 + qf)
        
        bm25_score += np.log(term1) * term2 * term3
        
    return bm25_score


# Calculate the bm25 score for each passage and query pair in validation_df_subset
# Create a column for the passage length
validation_df_subset['passage_length'] = validation_df_subset['pre-processed passage'].str.len()
k1=1.2
k2=100
b=0.75

# Create a df with unique passages
unique_pids_df = validation_df_subset.drop_duplicates(subset=['pid'])
sum_passage_lengths = unique_pids_df['passage_length'].sum()

# Find the average passage length
avdl = sum_passage_lengths / N

# Create a column for K (Parameter required for bm25)
validation_df_subset['K'] = k1 * ((1-b) + (b*(validation_df_subset['passage_length']/avdl)))

def calculate_R(qid):
    try:
        R = len(relevant_dict[qid]) 
    except KeyError:
        R = 0
    return R

# Create a column for R (Number of passages that are relevant to the query)
validation_df_subset['R'] = validation_df_subset['qid'].apply(lambda row: calculate_R(row))

validation_df_subset['bm25 score'] = 0
for i in range(len(validation_df_subset)):    
    validation_df_subset.iloc[i,10] = bm25_score(query=validation_df_subset.iloc[i,5],passage=validation_df_subset.iloc[i,6],qid=validation_df_subset.iloc[i,0],K=validation_df_subset.iloc[i,8],R=validation_df_subset.iloc[i,9],k1=1.2,k2=100,b=0.75)
 
validation_df_subset.head(5)

t1_results_df = validation_df_subset[['qid', 'pid', 'bm25 score']]
t1_results_df.head()

t1_results_df = t1_results_df.sort_values(['qid','bm25 score'],ascending=False).groupby('qid').head(len(t1_results_df))
t1_results_df

# Create a dictionary with the retrieved passages dor each query
rankings_dict = {}
for i in range(len(t1_results_df)):
    qid = t1_results_df.iloc[i,0]
    pid = t1_results_df.iloc[i,1]
    if qid in rankings_dict.keys():
        rankings_dict[qid].append(pid)
    else:
        rankings_dict[qid] = [pid]  
        
        
# Calculate the average precision of the bm25 model
def calculate_precision_at_k(r, k):
    '''
    Calculates the precision at the nth position
    '''
    r = np.asarray(r)
    return np.mean(r[:k])

def calculate_average_precision(r):
    '''
    Calculate the average precision of a query
    input r is a list of 0s and 1s
    '''
    val_lst = []
    for i in range(len(r)):
        if r[i] == 1:
            val_lst.append(calculate_precision_at_k(r, i+1))
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
            if qid_value in relevant_dict.keys():               
                if pid_val in relevant_dict[qid_value]:
                    relevent_lst.append(1)
            else:
                relevent_lst.append(0)
        precision_sum += calculate_average_precision(relevent_lst) # Add the average precision of the query to the precision_sum
    mean_avg_precision = precision_sum / len(rankings_dict)
    return mean_avg_precision

mean_avg_precision = calculate_mean_average_precision(rankings_dict, relevant_dict)
print(f"The mean average precision of the bm25 model is {round(mean_avg_precision,4)}")      


# Calculate the NDCG metric for the bm25 model
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
            if qid_val in relevant_dict.keys():
                if pid in relevant_dict[qid_val]:
                    rel_lst.append(1)
                    num_passages_relevant += 1
                else:
                    rel_lst.append(0)
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
print(f"The average NDCG metric for the bm25 model is {round(avg_ndcg,4)}")  














                               
