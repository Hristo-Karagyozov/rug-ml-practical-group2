import pandas as pd
import nltk
import numpy as np
import preprocessing as prep

def load_tweets():
    """"
    loads the tweets used for training
    """
    with open("Data/train_text.txt", "r", encoding="utf-8") as f:
        tweets = f.readlines()
        df = pd.DataFrame({'tweet': tweets})

    return df


if __name__ == "__main__":
    pipeline = prep.Pipeline(load_tweets())
    pipeline.preprocess()
