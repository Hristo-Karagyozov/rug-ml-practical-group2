import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

from Model_Deployment.pythonProject.src.main.preprocessing import pipeline as preprocessing


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    metric = evaluate.load("accuracy")

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)


def load_tweets_and_labels():
    # read the tweets and emojis (labels) and put them in a pd df
    with open("C:/Users/david/OneDrive/Desktop/University/Year 3/Machine Learning Practical/rug-ml-practical-group2/Model_Deployment/pythonProject/data/raw/test/us_test.text", "r", encoding="utf-8") as tweet_file:
        tweets = tweet_file.readlines()

    with open("C:/Users/david/OneDrive/Desktop/University/Year 3/Machine Learning "
              "Practical/rug-ml-practical-group2/Model_Deployment/pythonProject/data/raw/test"
              "/us_test.labels", "r", encoding="utf-8") as label_file:
        labels = label_file.readlines()

        # use zip to make instead of 2 dicts 1 list with 50k dicts
        # what we have now: {labels: [1,3,4...2]}, {tweets: [nsdo, sdjf, sdjf, ... sdjf]}
        # what we should have: {label:1, tweet:nsdo}
        #                      {label:3, tweet:sdjf}
        #                              .
        #                              .
        #                              .
        #                              .
        #                     {label:2, tweet:sdfj}

    # df = pd.DataFrame({'tweet': tweets, 'label': labels})
    # print(df.head(10))

    proper_df = [{'label': int(label), 'tweet': tweet} for label, tweet in zip(labels, tweets)]
    #
    # for i, item in enumerate(proper_df):
    #     print(item)
    #     if i == 4:
    #         break

    return proper_df


def main():
    pipeline = preprocessing.Pipeline(load_tweets_and_labels())
    processed_data = pipeline.preprocess()

    dataset = Dataset.from_pandas(processed_data)

    train_dataset, test_dataset = train_test_split(dataset, train_size=0.8, random_state=42)

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
