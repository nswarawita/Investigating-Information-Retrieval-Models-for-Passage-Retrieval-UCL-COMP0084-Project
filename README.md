Instructions to run the code

The Code Folder contains 4 files
1. Instructions.txt
2. src (this is a folder)
3. data (this is a folder)
4. source notebooks (this is a folder)


src contains Q1.py, Q2_P1.py, Q2_P2.py, Q3.py, train_df_subset.pkl, validation_df_subset.pkl and relevant_dictionary.pkl

Q1.py contains the code for subtask 1 - Evaluating Retrieval Quality 
Q2_P1.py contains the code for subtask 2 - Logistic Regression (LR) until computing query and passage embeddings for both the train and validation data subsets. This may take awhile.
Q2_P2.py contains the remaining code for subtask 2 - Logistic Regression (LR). This part can be run without running Q2_P1.py as it imports train_df_subset.pkl and validation_df_subset.pkl
Q3.py contains the code for subtask 3 - LambdaMART model (LM)
train_df_subset.pkl, validation_df_subset.pkl and relevant_dictionary.pkl are pickled objects. They are required to run the code. 


data is an EMPTY folder. Download train_data.tsv and validation_data.tsv into the data folder. 


source notebooks include Q1.ipynb, Q2.ipynb and Q3.ipynb. Please refer to these only if the .py files do not work 

-------------------------------------
Instructions to run the code
1. Check that train_data.tsv and validation_data.tsv have been downloaded into the data folder
2. Set the src folder (Code/src) as the working directory 
3. Run Q1.py, Q2_P1.py, Q2_P2.py and Q3.py

