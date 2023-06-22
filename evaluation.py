import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score

import nltk
from nltk.metrics import agreement
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics import masi_distance

"""
    This script help to compute metrics for the multi-label prediction. The script take two arguments:
    1. The path to  the csv file that contain the true label
    2. The path to the csv file that contain the predicted label, geenrated using the fine-tuned BERT model.
"""

def krippendorff_alpha(df):
    """
        This function compute the Krippendorff’s Alpha inter-Annotator(between the manual and BERT-predicted annotation) Agreement.
        source: https://stats.stackexchange.com/questions/407453/krippendorffs-alpha-in-r-for-multi-label-annotation

        The function extract annotation from the df and convert it to a data format compatible with nltk.AnnotationTask to compute later the Krippendorff's Alpha score.
        The funciton convert each row to a frozenset unit, e.g. For a 4-label annotated row: [1,0,1,0] => [ ('annotator1','Item0',frozenset([1,0,1,0])) ]

        :args
            - filename: a pandas dataframe with the following header format: ['prediction', 'annotation']
                where:
                    - prediction is the column that contain the predicted Fine-tuned BERT labels as string e.g. grammar, duplication
                    - annotation is the column that contain the true labels, human annotation as string e.g. slot addition

        :return
            The Krippendorff’s Alpha inter-Annotator score
    """

    task_data = []

    idxc = 0#counter to track non empty row in df
    for index, row in df.iterrows():
        #if prediction is not an empty value
        if row[0]:
            idxc+=1
            #add predicted label
            data  = row[0]
            task_data.append(
                (
                    'coder1',f"Item{idxc}",frozenset(data)
                )
            )

            #add manually annotated label
            data  = row[1]
            task_data.append(
                (
                    'coder2',f"Item{idxc}",frozenset(data)
                )
            )
    
    task = AnnotationTask(distance = masi_distance)
    task.load_array(task_data)
    kripp_alpha = task.alpha()
    return kripp_alpha

def vectorize_data(df):
    """
        This function extracts the value of the columns of each row in df and converts it into a single vector. For example:
        Suppose we have a df with 3 labels:
                A  B  C
            0   1  0  0
            1   1  1  0
            2   0  0  1

        Then this function convert it to a numpy array: arr = [ [1,0,0], [1,1,0], [0,0,1] ]
        
        :args
            - df: the pandas dataframe to be processed and converted.

        :return
            - A numpy array that contains all the required column values as a single vector for each row in df.
    """

    arr = []

    for index, row in df.iterrows():
        values = [v for v in row]#extarct all labels from each row
        arr.append(values)
    
    return np.array(arr)

def convert_label_to_number(df,new_columns,default_value = 0):
    """
        Convert straing label to binary values.
        Example: For 5 labels ['a','e'], ['b','c','d'], ['d']:
            a   b   c   d   e
        0   1   0   0   0   1
        1   0   1   1   1   0
        2   0   0   0   1   0

        :args
            - new_columns (list): A python list of string labels. List of column names for the new DataFrame.
            - default_value (int): The defined value to initialize the new columns.

        :return
    """
    
    new_df_prediction = pd.DataFrame(default_value, index = df.index, columns = new_columns)
    new_df_annotation = pd.DataFrame(default_value, index = df.index, columns = new_columns)

    for index, row in df.iterrows():

        ##convert prediction column values to 1
        if row['prediction']:
            labels = row['prediction']#get list of labels

            for c in labels:
                new_df_prediction.loc[index, c] = 1
        
        #convert annotation column values to 1
        if row['annotation']:
            labels = row['annotation']#get list of labels

            for c in labels:
                new_df_annotation.loc[index, c] = 1
    
    
    y_pred = vectorize_data(new_df_prediction)
    y_true = vectorize_data(new_df_annotation)
    return y_pred,y_true

def error_label_to_column(df, new_columns = ['a', 'b', 'c']):
    """
        This function creates 16 new columns and assigns them a default value of 0. For each error label in the error_category column, the value is replaced by 1 in the corresponding new added column.
        These new 16 added columns will be the output label classes of the model to train.

        :args
            - df: the pandas dataframe to process. The dataframe header is: [prediction,annotation] : [prediction,annotation], where prediction is the GPT-based multilabel prediction column and annotation are the columns for truth labels.
        
        :return
            - Two pandas dataframe with the following header: [ 'semantic', 'spelling', 'grammar', 'redundant', 'duplication', 'incoherent', 'punctuation', 'wrong slot', 'slot addition', 'slot omission', 'wordy', 'answering', 
            'questioning', 'homonym', 'acronym', 'correct']
            First dataframe is the preditcion columns and the second one is the annotation columns. The function will convert string label to binary value in the corresponding column.
            Example:
                annotation  prediction
            0  [X, Y, Z]  [A, B, C]
            1  [A, D, E]  [C, D, y]

            >>> df1,df2 = error_label_to_column(df)
            >>> df1.head()
                A   B   C   D   E   ... X   Y   Z 
            0   0   0   0   0   0       1   1   1
            1   1   0   0   1   1       0   0   0

            >>> df2.head()
                A   B   C   D   E   ... X   Y   Z 
            0   1   1   1   0   0       0   0   0
            1   0   0   1   1   0       0   1   0
    """

    y_pred,y_true = convert_label_to_number(df,new_columns)
    
    return y_pred,y_true

def compute_metric(df,y_pred,y_true):
    """
        Compute various evaluation metrics for predicted and true labels.

        This function takes predicted paraphrases with error labels and true labels as input, and computes various evaluation metrics including
        Krippendorff's alpha inter-agreement, Exact Match Ratio (EMR), Hamming loss, recall, precision, and F1 measure.

        :args
            - df (pandas.DataFrame): The input DataFrame used for calculating Krippendorff's alpha.
            - y_pred (numpy.array-like): Predicted labels.
            - y_true (numpy.array-like): True labels.

        :return
            - tuple: A tuple containing the following metrics:
                - kripp_alpha (float): Krippendorff's alpha inter-agreement.
                - EMR (float): Exact Match Ratio.
                - h_loss (float): Hamming loss.
                - r_score (float): Recall score (averaged over samples).
                - p_score (float): Precision score (averaged over samples).
                - f1 (float): F1 measure (averaged over samples).
    """
    
    kripp_alpha = krippendorff_alpha(df)
    # print(f"Krippendorff's alpha inter-agreement: {kripp_alpha}")

    EMR = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    # print(f'Exact Match Ratio: {EMR}')

    h_loss = hamming_loss(y_true, y_pred)
    # print(f'Hamming loss: {h_loss}')

    r_score = recall_score(y_true=y_true, y_pred=y_pred, average='samples')
    # print(f'Recall: {r_score}') 

    p_score = precision_score(y_true=y_true, y_pred=y_pred, average='samples')
    # print(f'Precision: {p_score}')

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='samples')
    # print(f'F1 Measure: {f1}')

    return kripp_alpha, EMR, h_loss, r_score, p_score, f1