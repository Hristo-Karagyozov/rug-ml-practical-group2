import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud


def generate_wordcloud(df, emoji):
    tweets_for_emoji = df[df['Emoji'] == emoji]['Tweet'].str.cat(sep=' ')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(tweets_for_emoji)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for Heart Emoji')
    plt.show()


def train_word2vec(df):

    tokenized_tweets = df.tolist()
    word2vec_model = Word2Vec(sentences=tokenized_tweets,
                              vector_size=100, window=5,
                              min_count=1, workers=4)
    word2vec_model.save("word2vec_model.model")


def tokenized_dataframe():
    text_path = '../us_trial.text'
    df = pd.read_csv(text_path, header=None, names=['Tweet'], sep='\t')
    tokenized_df = df['Tweet'].apply(lambda x: word_tokenize(x))
    return tokenized_df


def load_data():
    labels_path = '../us_trial.labels'
    text_path = '../us_trial.text'

    labels_df = pd.read_csv(labels_path, header=None, names=['Emoji'], sep='\t')
    text_df = pd.read_csv(text_path, header=None, names=['Tweet'], sep='\t')

    df = pd.concat([labels_df, text_df], axis=1)
    return df


def make_frequency_histogram(df):
    emoji_frequency = df['Emoji'].value_counts()
    emoji_frequency.plot(kind='bar')
    plt.title('Distribution of Tweets per Emoji')
    plt.xlabel('Emoji')
    plt.ylabel('Frequency')
    plt.show()


def main():
    # nltk.download('punkt')

    # Loading the data with the feature and label
    df = load_data()
    make_frequency_histogram(df)
    # tokenized_df = tokenized_dataframe()
    # train_word2vec(tokenized_df)
    generate_wordcloud(df, 0)


if __name__ == "__main__":
    main()
