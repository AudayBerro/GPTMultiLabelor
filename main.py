import os
import configparser
import argparse
import time

from skllm.config import SKLLMConfig
from skllm import MultiLabelZeroShotGPTClassifier
import pandas as pd

from process_data import skllm_format_converter
from evaluation import error_label_to_column, compute_metric

"""
    Multi-label zero-shot with Scikit-LLM library https://github.com/iryna-kondr/scikit-llm

    Inspired from:
        - https://medium.com/@fareedkhandev/scikit-llm-sklearn-meets-large-language-models-11fc6f30e530
        - https://towardsdatascience.com/scikit-llm-power-up-your-text-analysis-in-python-using-llm-models-within-scikit-learn-framework-e9f101ffb6d4
    
    To execute the script and Fine-tune a GPY based model run the following command: python  main.py -f TPME_dataset.csv
"""

def configure_API_Key():
    """
        Configure OpenAI API (key). Retreive your personal OpenAI API key from the config.ini file
        If you don't have an OpenAi APY Key please create one https://platform.openai.com/account/api-keys

        args:
            - None

        return:
            - None
    """

    #load configs from config.ini file
    config = configparser.ConfigParser(inline_comment_prefixes="#")
    config.read(os.path.join(os.path.dirname(__file__), ".","config.ini"))
    OPENAI_KEY_CONFIG = config["OpenAIKey"]
    OPENAI_ORG_CONFIG = config["OpenAIOrganization"]

    if "OPENAI_SECRET_KEY" not in OPENAI_KEY_CONFIG and OPENAI_KEY_CONFIG["OPENAI_SECRET_KEY"] == "":
        raise Exception("You need to pass your OpenAI API key to Scikit-LLM in config.ini")
    else:
        OPENAI_SECRET_KEY = OPENAI_KEY_CONFIG['OPENAI_SECRET_KEY']
    
    
    if "OPENAI_ORG_ID" not in OPENAI_ORG_CONFIG and OPENAI_ORG_CONFIG["OPENAI_ORG_ID"] == "":
        raise Exception("OpenAI organization was not found. You need to pass your organization ID to Scikit-LLM in config.ini")
    else:
        OPENAI_ORG_ID = OPENAI_ORG_CONFIG['OPENAI_ORG_ID']
    # configure OpenAI API (key) Set your OpenAI API key
    SKLLMConfig.set_openai_key(OPENAI_SECRET_KEY)
    SKLLMConfig.set_openai_org(OPENAI_ORG_ID)

def load_parser():
    """
        Load argparse library bject

        args:
            - None

        return:
            - parser: <class 'argparse.Namespace'> object
    """
    parser = argparse.ArgumentParser(description='Fine-tune Multi Label ZeroShot GPT Classifier')

    parser.add_argument('-f','--datapath', required=True,
        help='path to the file containing the training dataset') # -f data set file name argument
    
    parser.add_argument('-l','--max_labels', type=int, required=False, default=5,
        help="The maximum number of labels you want to assign to each sample. Define it when you don't have labeled dataset")
    
    parser.add_argument('-m','--openai_model', type=str, required=False, default='gpt-3.5-turbo',
        help="OpenAI model to use. The OpenAI API is powered by a diverse set of models with different capabilities and price points https://platform.openai.com/docs/models/")
    
    return parser.parse_args()

def get_dataset(data_path, label_avalability = True):
    """
        Get classification dataset that we will use to fine-tuning the selected OpenAI model

        args:
            - data_path: pytho nstring, allows you to obtain the path to the training dataset arguments passed in
            - label_avalability: Boolean, if True means the dataset is labeled, False means there is no labeled data.

        return:
            - X,Y: X is a python list of paraphrases and Y is a python list of list of labels.
            e.g.
                X = [
                    "Book a flight from Lyon to Sydney",
                    "Any restaurant servin gLebanese food near me?"
                ]
                
                Y = [
                    ["correct"],
                    ["spelling"]
                ]
    """
    __labels = [
        'semantic', 'spelling', 'grammar', 'redundant', 'duplication', 'incoherent', 'punctuation', 'wrong slot', 'slot addition', 'slot omission', 'wordy', 'answering', 'questioning', 'homonym', 'acronym', 'correct'
    ]

    #read data
    df = pd.read_csv(data_path, sep = ',',na_filter= False)

    X,y = skllm_format_converter(df,__labels)
    return X,y

def get_prediction(clf, data):
    """
        Get classification dataset that we will use to fine-tuning the selected OpenAI model

        args:
            - clf: a skllm.models.gpt_zero_shot_clf.MultiLabelZeroShotGPTClassifier  instance, this object is a wrapper for the fine-tuned GPT-based model
            - data: a python list of sentence strings for which we will predict the error label with the clf model.
            e.g. X = [
                    "Book a flight fromm Lyon to",
                    "Any restaurant servin gLebanese food near me?"
                ]
            
        return:
            - predicted_paraphrases_error: a python list of predicted errors.
            e.g.
                predicted_paraphrases_error = [
                    ['spelling','slot omission'], ---> predicted error for the sentence 1: "Book a flight fromm Lyon to",
                    ['spelling']" ---> predicted error for the sentence 2: "Any restaurant servin gLebanese food near me?"
                ]
    """

    predicted_paraphrases_error = clf.predict(X = data)

    return predicted_paraphrases_error


def merge_predicted_with_true_labels(predicted_labels,annotated_labels):
    """
        Merge predicted and annotated labels into a DataFrame.
        
        :args
            - predicted_labels (list of lists): A python list of lists representing predicted labels. Contains labels predicted by the GPT Multi Label ZeroShot model
            - annotated_labels (list of lists): A python list of lists representing annotated labels. Contains ground truth labels that have been manually annotated in the TPME dataset.

        :return
            pandas.DataFrame: A DataFrame with two columns: 'predicted' and 'annotated'.

            Example:
            >>> predicted = [['A', 'B', 'C'], ['D', 'E', 'F']]
            >>> annotated = [['X', 'Y', 'Z'], ['P', 'Q', 'R']]
            >>> df = merge_predicted_with_true_labels(predicted, annotated)
            >>> df.head()
                annotation  prediction
            0  [X, Y, Z]  [A, B, C]
            1  [P, Q, R]  [D, E, F]
    """
    
    # Create a dictionary with 'prediction' and 'annotation' as keys
    data = {'prediction': predicted_labels, 'annotation': annotated_labels}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    return df

def test():
    """
        This is a simple test function to show how to execute the code to obtain the predicted error label.

        args:
            - None

        return:
            - None

    """
    #Configure OpenAI API (key)
    configure_API_Key()

    #Load parser
    args = load_parser()

    #Load training dataset
    X,y = get_dataset(args.datapath)

    # defining the model
    openai_model = args.openai_model

    #Initialize the classifier with the OpenAI model
    # max_labels = args.max_labels
    max_labels = 5
    clf = MultiLabelZeroShotGPTClassifier(max_labels = max_labels)

    #fitting the data / Train the model 
    clf.fit(X = X[:500], y = y[0:500])

    #Use the trained classifier to predict the error of the paraphrases
    test_set = X[501:550]
    true_label = y[501:550]
    predicted_paraphrases_error = get_prediction(clf, test_set)

    for review, labels, truth in zip(test_set, predicted_paraphrases_error,true_label):
        print(f"Review: {review}\nPredicted Labels: {labels}\nTrue Labels: {truth}\n")
    
    df = merge_predicted_with_true_labels(predicted_paraphrases_error,true_label)

    timestr = time.strftime("%Y%m%d-%H-%M-%S")
    file_name = f"./predicted_vs_annotated-{timestr}.csv"
    df.to_csv(file_name, index = False)

    new_columns = [
        'semantic', 'spelling', 'grammar', 'redundant', 'duplication', 'incoherent', 'punctuation', 'wrong slot', 'slot addition', 'slot omission', 'wordy', 'answering', 'questioning', 'homonym', 'acronym', 'correct'
    ]

    y_pred, y_true = error_label_to_column(df, new_columns)

    kripp_alpha, EMR, h_loss, r_score, p_score, f1 = compute_metric(df, y_pred, y_true)

    with open(f'Metric_score_-{timestr}.txt', "w") as f:
        l = "\tKrippendorff's alpha inter-agreement: {kripp_alpha}"
        l = len(l)+8
        line_separator = "="*l
        f.write(f"\n{line_separator}\n")
        f.write(f"\tKrippendorff's alpha inter-agreement: {kripp_alpha}\n")
        f.write(f'\tExact Match Ratio: {EMR}\n')
        f.write(f'\tHamming loss: {h_loss}\n')
        f.write(f'\tRecall: {r_score}\n')
        f.write(f'\tPrecision: {p_score}\n')
        f.write(f'\tF1 Measure: {f1}\n')
        f.write(f"\n{line_separator}\n")

if __name__ == "__main__":

    test()