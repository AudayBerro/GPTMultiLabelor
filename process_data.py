import sys
from ast import literal_eval

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


""" This script processes the Transformer Error dataset and converts it into a compatible input format to train a multi-label classifier. """

def clean_columns(df,desired_columns = ['utterance', 'paraphrase', 'list_of_slots', 'error_category']):
    """
        This function removes undesired columns from a pandas dataframe.
        
        :args
            - df: the pandas dataframe to be cleaned.
            - desired_columns : a python list containing the desired columns to keep.

        :return
            - a pandas dataframe without the unwanted columns.
    """

    unwanted_c = list( set(df.columns) - set(desired_columns) )#Get the difference between the df.columns and the lists of desired columns.
    df.drop( unwanted_c,axis = 1, inplace = True)#remove the unwanted columns form the dataframe
    return df


def normalize_error_column(df):
    """
        This function convert the values of the error_category column to a list of errors, e.g. convert "semantic, wordy, slot addition" to [semantic, wordy, slot addition].
        :args
            - df: the pandas dataframe to process.
        
        :return
            - a pandas dataframe with the desired error_category column format
    """

    labels_list = {
        'semantic', 'spelling', 'grammar', 'redundant', 'duplication', 'incoherent', 'punctuation', 'wrong slot', 'slot addition', 'slot omission', 'wordy', 'answering', 'questioning', 'homonym', 'acronym', 'correct'
    }

    for i, row in df.iterrows():
        row_value = row['error_category']#extract the current row value e.g. semantic, wordy, slot addition
        row_value = row_value.lower().split(",")
        row_value = [i.strip() for i in row_value]#remove leading and trailing whitespace
        
        #check that the value of the error_category column is not wrong and that it contains required labels. It is possible that we have an empty cell or wrong data in the current row
        if set(row_value).issubset( labels_list ):
            for label in row_value:
                df.at[i,label] = 1.0
            
            df.at[i,'error_category'] = f"{row_value}"
    return df

def error_label_to_column(df):
    """
        This function creates 16 new columns and assigns them a default value of 0. For each error label in the error_category column, the value is replaced by 1 in the corresponding new added column.
        These new 16 added columns will be the output label classes of the model to train.

        :args
            - df: the pandas dataframe to process.
        
        :return
            - a pandas dataframe with the following header format:
                utterance, paraphrase, list_of_slots, error_category, semantic, spelling, grammar, redundant, duplication, incoherent, punctuation, wrong slot, slot addition, slot omission, wordy,
                answering, questioning, homonym, acronym, correct
    """
    new_columns = [
        'semantic', 'spelling', 'grammar', 'redundant', 'duplication', 'incoherent', 'punctuation', 'wrong slot', 'slot addition', 'slot omission', 'wordy', 'answering', 'questioning', 'homonym', 'acronym', 'correct'
    ]

    for c in new_columns:
        df[ c ] = 0.0
    
    return df

def add_labels_to_bar_plot(x,y):
    """
        Add value labels on a Matplotlib Bar Chart. Adding value labels in the center of each Bar on the Bar Chart.

        :args
            - x: a python list containing the labels of the plot x-axis, the labels.
            - y: a python list containing the number of each label on the x-axis, labels_count.

            e.g., x = ["label-A", "label-B", "label-C", "label-D"] y = [234, 2344, 125, 653]
        :retrun
            - None
    """
    for i in range(len(x)):
        plt.text(i, y[i], y[i], fontsize = 13, ha = 'center', va = 'bottom')

def plot_labels_count(df, labels):
    """
        This function count for each label. Filter all the label/output columns.

        :args
            - df: dataframe to process.
            - labels : a python list containing the labels that will be used to count the number of occurrences in df. This list will also be useful for selecting columns.
    """

    data_to_plot = df[labels]
    print(df.columns)
    print(data_to_plot.head())

    #Using the df data frame, draw the graph that will show the different counts of labels for each row.
    plt.clf()
    fig = plt.figure(figsize = (18, 12))

    # creating the bar plot
    plt.title("Number of labels in the dataset",fontsize = 18)
    plt.xlabel('Labels',fontsize = 18)
    plt.ylabel('Count',fontsize = 18)
    
    data_to_plot.sum(axis=0).plot.bar(width = 0.4, fontsize = 14)
    
    # get labels count
    labels_count = data_to_plot.sum(axis=0).tolist()#sum() return a pandas.core.series.Series object

    add_labels_to_bar_plot(labels,labels_count)
    plt.savefig('./labels.png')#save the ploted figure

def clean_y(df):
    """
        This function processes the data and removes quotation marks preceding list items in the data, thus cleaning up the data. Converts the data from a list of strings containing a list of labels into a list of lists of labels.
        e.g. if data = ["[a,b]","[c,d,e]","[a,e]","[b]"] => data = [[a,b],[c,d,e],[a,e],[b]]

        :args
            - data: a pandas.core.series.Series , where each element is a list of label, e.g, data = [ "[a,b]", "[c,d,e]", "[a,e]", "[b]" ]
        
        :return
            - a processed python list of list elements, e.g data = [[a,b],[c,d,e],[a,e],[b]]
    """

    y = [ literal_eval(v) for i,v in df.items()]#literal_eval convrt a string of list to a list e.g. "['a','b','c']" => ['a','b','c']
    return y


def convert_data_to_skllm_format(df,target_label = "error_category"):
    """
        Convert the df data to a fromat compatible with the Scikit-LLM library. See the get_multilabel_classification_dataset() at https://github.com/iryna-kondr/scikit-llm/blob/main/skllm/datasets/multi_label.py
        This function returns two python lists X and y. To construct X, we'll extract only the paraphrase row from the database. To construct y, we'll extract only the target_label column.

        :args
            - df: the pandas dataframe to process.
            - target_label: a pyhton string specifying the column in the datasframe containing the ground truth labels.

        :return
            - X,y : where X is a python list of paraphrases and y their respective labled errors.
            
    """

    X = df['paraphrase'].values.tolist()# extarct paraprhase column only

    y = df[target_label]# extarct error_category column only
    y = clean_y(y)

    return X,y

def skllm_format_converter(df,labels):
    """
        A wrapper function to convert a pandas dataframe to a format compatible whith the Scikit-LLM library MultiLabelZeroShotGPTClassifier function to a GPT-based model on the Multi-Label Zero-Shot paraphrases error prediction task.

        :args
            - df: pandas dataframe containing the data to process.
            - labels: a python list containing the labels. This list will also be useful for selecting columns.
            -
            -

        :return
            - X,y: returns two python lists X and y. Where X is a python list of paraphrases and y their respective labeled errors.
    """

    #print columns
    # print(df.columns)
    
    #remove unwanted columns
    df = clean_columns(df)

    #add the 16 output classes
    df = error_label_to_column(df)

    # convert columns binary values into a single vector of string labels e.g., if we have 5 labels: [1,0,1,0,1] => [label-A,label-C,label-E] 
    df = normalize_error_column(df)

    plot_labels_count(df, labels)

    X,y = convert_data_to_skllm_format(df)

    assert len(X)==len(y),"X and y must have the same size"

    return X,y


if __name__ == "__main__":
    
    __labels = [
        'semantic', 'spelling', 'grammar', 'redundant', 'duplication', 'incoherent', 'punctuation', 'wrong slot', 'slot addition', 'slot omission', 'wordy', 'answering', 'questioning', 'homonym', 'acronym', 'correct'
    ]

    #read data
    df = pd.read_csv('./TPME_dataset.csv', sep = ',',na_filter= False)

    X,y = skllm_format_converter(df,__labels)