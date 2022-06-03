# -*- coding: utf-8 -*-
# Import the required packages
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()
from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('english') 
from gensim.models import Word2Vec
import pickle

# Import train_data
train_df = pd.read_csv("../data/train_data.tsv",sep='\t')
train_df 

# Import validation_data
validation_df = pd.read_csv("../data/validation_data.tsv",sep='\t')
validation_df 


# Create subsets of the training and validation data
train_subset_size = 100000

# Select all the rows that have a relevancy score of 1
df1 = train_df[train_df.relevancy!=0] 

# Select the rows of the dataset that have a relevancy score of 0
df2 = train_df[train_df.relevancy==0].head(train_subset_size - len(df1))

# Combine the datasets to form a subset of the data
frames_1 = [df1, df2]
train_df_subset = pd.concat(frames_1)
train_df_subset

validation_subset_size = 50000

# Select all the rows that have a relevancy score of 1
df3 = validation_df[validation_df.relevancy!=0] 

# Select the rows of the dataset that have a relevancy score of 0
df4 = validation_df[validation_df.relevancy==0].head(validation_subset_size - len(df3))

# Combine the datasets to form a subset of the data
frames_2 = [df3, df4]
validation_df_subset = pd.concat(frames_2)
validation_df_subset


# Preprocess the training and validation data
# (1) Convert both queries and passages to lowercase
train_df_subset['pre-processed queries'] = train_df_subset['queries'].str.lower()
train_df_subset['pre-processed passage'] = train_df_subset['passage'].str.lower()

# (2) Strip off punctuation 
train_df_subset['pre-processed queries'] = train_df_subset['pre-processed queries'].str.replace(r'[^\w\s]+', '')
train_df_subset['pre-processed passage'] = train_df_subset['pre-processed passage'].str.replace(r'[^\w\s]+', '')

# (3) Tokenise 
train_df_subset['pre-processed queries'] = train_df_subset['pre-processed queries'].str.split()
train_df_subset['pre-processed passage'] = train_df_subset['pre-processed passage'].str.split()

# (4) Stopword removal
train_df_subset['pre-processed queries'] = train_df_subset['pre-processed queries'].apply(lambda row: [word for word in row if word not in nltk_stopwords])
train_df_subset['pre-processed passage'] = train_df_subset['pre-processed passage'].apply(lambda row: [word for word in row if word not in nltk_stopwords])
train_df_subset.head(5)

# (1) Convert both queries and passages to lowercase
validation_df_subset['pre-processed queries'] = validation_df_subset['queries'].str.lower()
validation_df_subset['pre-processed passage'] = validation_df_subset['passage'].str.lower()

# (2) Strip off punctuation 
validation_df_subset['pre-processed queries'] = validation_df_subset['pre-processed queries'].str.replace(r'[^\w\s]+', '')
validation_df_subset['pre-processed passage'] = validation_df_subset['pre-processed passage'].str.replace(r'[^\w\s]+', '')

# (3) Tokenise 
validation_df_subset['pre-processed queries'] = validation_df_subset['pre-processed queries'].str.split()
validation_df_subset['pre-processed passage'] = validation_df_subset['pre-processed passage'].str.split()

# (4) Stopword removal
validation_df_subset['pre-processed queries'] = validation_df_subset['pre-processed queries'].apply(lambda row: [word for word in row if word not in nltk_stopwords])
validation_df_subset['pre-processed passage'] = validation_df_subset['pre-processed passage'].apply(lambda row: [word for word in row if word not in nltk_stopwords])
validation_df_subset.head(5)


# Combine the passages and queries as we want to use the same word2vec model to generate vectors for both passages and queries
queries_lst = train_df_subset['pre-processed queries'].tolist()
passage_lst = train_df_subset['pre-processed passage'].tolist()

train_data_lst = queries_lst + passage_lst


# Train the word2vec model using the data
train_embeddings = Word2Vec(train_data_lst, vector_size = 50, min_count = 2, window = 5, workers = 4)
vector_size = 50

vocabulary = list(train_embeddings.wv.index_to_key)
print(f"The length of the vocabulary is {len(vocabulary)} words")

# Compute the query/ passage embeddings by averaging embeddings of all the words in the query/ passage
# Create a function to do this
def create_embedding_fn(word_lst):
    embeddings = []
    for word in word_lst:
        if word in vocabulary:
            embeddings.append(train_embeddings.wv.get_vector(word))
        else:
            embeddings.append(np.random.rand(vector_size))          
    return np.mean(embeddings, axis=0)

train_df_subset['query vector representation'] = train_df_subset['pre-processed queries'].apply(lambda row : create_embedding_fn(row))
train_df_subset['passage vector representation'] = train_df_subset['pre-processed passage'].apply(lambda row : create_embedding_fn(row))

validation_df_subset['query vector representation'] = validation_df_subset['pre-processed queries'].apply(lambda row : create_embedding_fn(row))
validation_df_subset['passage vector representation'] = validation_df_subset['pre-processed passage'].apply(lambda row : create_embedding_fn(row))


# Save the train and validation datasets
#train_df_subset.to_pickle('train_df_subset.pkl')
#validation_df_subset.to_pickle('validation_df_subset.pkl')

