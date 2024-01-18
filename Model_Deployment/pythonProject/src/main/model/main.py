import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizer
from sklearn.model_selection import train_test_split

from src.main.preprocessing import pipeline as preprocessing


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metric = evaluate.load("accuracy")

    predictions = np.argmax(logits, axis=-1)
    print(predictions)

    return metric.compute(predictions=predictions, references=labels)


def load_tweets_and_labels():
    # read the tweets and emojis (labels) and put them in a pd df
    with open("../../../data/raw/us_test.labels", "r", encoding="utf-8") as label_file:
        labels = label_file.readlines()

    with open("../../../data/raw/us_test.text", "r", encoding="utf-8") as tweet_file:
        tweets = tweet_file.readlines()

    formatted_df = [{'label': int(label), 'tweet': tweet} for label, tweet in zip(labels, tweets)]

    return formatted_df


def main():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    pipeline = preprocessing.Pipeline(load_tweets_and_labels())
    processed_data = pipeline.preprocess()

    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(processed_data)

    # Tokenize the tweet text and generate attention masks
    tokenized_output = tokenizer(df['tweet'].tolist(), return_tensors="pt", padding=True, truncation=True,
                                 max_length=64)
    input_ids = tokenized_output["input_ids"]
    attention_mask = tokenized_output["attention_mask"]

    # Add input_ids and attention_mask to the DataFrame
    df['input_ids'] = input_ids.tolist()
    df['attention_mask'] = attention_mask.tolist()

    train_dataset, test_dataset = train_test_split(df, train_size=0.1, random_state=42)

    # Convert the two datasets from pandas to dataset type again
    train_dataset = Dataset.from_pandas(train_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=20)
    training_args = TrainingArguments(output_dir="test_trainer")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    main()
