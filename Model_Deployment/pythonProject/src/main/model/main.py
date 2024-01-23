import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, RobertaTokenizer

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


def freeze_layers(number_of_frozen_layers, model):
    for name, param, in model.roberta.named_parameters():
        layer_index = int(name.split(".")[2]) if 'encoder.layer' in name else None

        if layer_index is not None and layer_index <= number_of_frozen_layers:
            # small check to verify that we are actually freezing
            # print(f"froze layer {layer_index}")
            param.requires_grad = False

    for name, param in model.roberta.embeddings.named_parameters():
        param.requires_grad = False


def main():
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    pipeline = preprocessing.Pipeline(load_tweets_and_labels())
    processed_data = pipeline.preprocess()

    # convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(processed_data)

    # tokenize the tweet text and generate attention masks;

    # the highest tweet character count our computational
    # resources could allow us was 128, a quarter of
    # roberta's potential, half of the max character
    # amount that twitter allows

    chosen_token_length = int(tokenizer.model_max_length / 4)

    tokenized_output = tokenizer(df['tweet'].tolist(),
                                 return_tensors="pt",
                                 padding=True,
                                 truncation=True,
                                 max_length=chosen_token_length)

    input_ids = tokenized_output["input_ids"]
    attention_mask = tokenized_output["attention_mask"]

    # add input_ids and attention_mask to the DataFrame
    df['input_ids'] = input_ids.tolist()
    df['attention_mask'] = attention_mask.tolist()

    # these two splits divide the data into 70% training, 15% evaluation, 15% testing
    train_dataset, rest_dataset = train_test_split(df, train_size=0.7, random_state=42)
    evaluate_dataset, test_dataset = train_test_split(rest_dataset, train_size=0.5, random_state=42)

    # convert the two datasets from pandas to dataset type again
    train_dataset = Dataset.from_pandas(train_dataset)
    evaluate_dataset = Dataset.from_pandas(test_dataset)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=20)
    training_args = TrainingArguments(output_dir="test_trainer")

    # freezing the first n layers depending on the best outcome
    number_of_frozen_layers = 10

    freeze_layers(number_of_frozen_layers, model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=evaluate_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.save_pretrained("src/main/model/trained_model")
    tokenizer.save_pretrained("src/main/model/tokenizer")


if __name__ == "__main__":
    main()
