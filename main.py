import os
import configparser
import argparse

from skllm.config import SKLLMConfig

"""
    Multi-label zero-shot with Scikit-LLM library https://github.com/iryna-kondr/scikit-llm

    Inspired from:
        - https://medium.com/@fareedkhandev/scikit-llm-sklearn-meets-large-language-models-11fc6f30e530
        - https://towardsdatascience.com/scikit-llm-power-up-your-text-analysis-in-python-using-llm-models-within-scikit-learn-framework-e9f101ffb6d4
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

    if "OPENAI_SECRET_KEY" not in OPENAI_KEY_CONFIG and OPENAI_KEY_CONFIG["OPENAI_SECRET_KEY"] == "":
        raise Exception("You need to pass your OpenAI API key to Scikit-LLM in config.ini")
    else:
        OPENAI_SECRET_KEY = OPENAI_KEY_CONFIG['OPENAI_SECRET_KEY']
    
    # configure OpenAI API (key) Set your OpenAI API key
    SKLLMConfig.set_openai_key(OPENAI_SECRET_KEY)

def load_parser():
    """
        Load argparse library bject

        args:
            - None

        return:
            - parser: <class 'argparse.ArgumentParser'> object
    """

    parser = argparse.ArgumentParser(description='Fine-tune Multi Label ZeroShot GPT Classifier')

    parser.add_argument('-f','--datapath', required=True,
        help='path to the file containing the training dataset') # -f data set file name argument
    
    parser.add_argument('-l','--max_labels', type=int, required=False, default=3,
        help="The maximum number of labels you want to assign to each sample. Define it when you don't have labeled dataset",)
    
    parser.add_argument('-m','--openai_model', type=str, required=False, default='gpt-3.5-turbo',
        help="OpenAI model to use. The OpenAI API is powered by a diverse set of models with different capabilities and price points https://platform.openai.com/docs/models/",)
    
    return parser

def get_dataset(parser, label_avalability = True):
    """
        Get classification dataset that we will use to fine-tuning the selected OpenAI model

        args:
            - parser: <class 'argparse.Namespace'> object, allows you to obtain the path to the training dataset and other optional arguments passed in
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
    pass

if __name__ == "__main__":

    #Configure OpenAI API (key)
    configure_API_Key()

    #Load parser
    parser = load_parser()
    args = parser.parse_args()

    #Load training dataset
    X,Y = get_dataset(args)