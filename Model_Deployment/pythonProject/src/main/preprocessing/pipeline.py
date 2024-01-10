import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


class Pipeline:
    def __init__(self, df):
        self.df = df
        self.punctuation = "[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]"
        self.stop_words = set(stopwords.words('english'))
        self.custom_stop_words = ["user", "tweet"]
        self.stop_words.update(self.custom_stop_words)

    def remove_stop_words(self, data):
        d = data.copy()

        for i in range(len(d)):
            words = d.iloc[i]["tweet"].split()
            filtered = [word for word in words if word not in self.stop_words]
            d.at[i, "tweet"] = ' '.join(filtered)

        return d

    def tokenize(self, data):
        d = data.copy()
        for idx, text in d["tweet"].items():
            tokens = word_tokenize(text)
            d.loc[idx, 'tweet'] = ' '.join(tokens)

        return d

    def lemmatize(self, data):
        d = data.copy()
        lemmatizer = WordNetLemmatizer()

        for idx in range(len(d)):
            words = nltk.word_tokenize(str(d.iloc[idx]["tweet"]))
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            d.at[idx, "tweet"] = ' '.join(lemmatized_words)

        return d

    def remove_punctuation(self, data):
        d = data.copy()
        transtab = str.maketrans(dict.fromkeys(self.punctuation, ''))

        for idx in range(len(d)):
            tweet = str(d.iloc[idx]["tweet"])
            d.at[idx, "tweet"] = tweet.translate(transtab)

        return d

    def pad(self, data):
        d = data.copy()

        for idx in range(len(d)):
            current_tweet = str(d.iloc[idx]["tweet"])

            # I put a stock number here, but we should
            # decide on what to put based on avg
            # string length or something like that

            padded_tweet = current_tweet.ljust(128)
            d.at[idx, "tweet"] = padded_tweet

        return d

    def preprocess(self):
        tokenized_data = self.tokenize(self.df)
        removed_stop_words = self.remove_stop_words(tokenized_data)
        lemmatized_data = self.lemmatize(removed_stop_words)
        data_no_punctuation = self.remove_punctuation(lemmatized_data)
        padded_data = self.pad(data_no_punctuation)

        # return padded_data.to_csv("out.csv")
        return padded_data
