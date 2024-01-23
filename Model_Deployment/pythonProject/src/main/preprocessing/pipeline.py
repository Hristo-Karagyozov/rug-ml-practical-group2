import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class Pipeline:
    def __init__(self, data_list):
        self.data_list = data_list
        self.punctuation = "[!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~]"
        self.stop_words = set(stopwords.words('english'))
        self.custom_stop_words = ["user", "tweet"]
        self.stop_words.update(self.custom_stop_words)

    def remove_stop_words(self, data):
        for entry in data:
            words = entry["tweet"].split()
            filtered = [word for word in words if word not in self.stop_words]
            entry["tweet"] = ' '.join(filtered)

        return data

    def tokenize(self, data):
        for entry in data:
            tokens = word_tokenize(entry["tweet"])
            entry['tweet'] = ' '.join(tokens)

        return data

    def lemmatize(self, data):
        lemmatizer = WordNetLemmatizer()

        for entry in data:
            words = nltk.word_tokenize(str(entry["tweet"]))
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            entry["tweet"] = ' '.join(lemmatized_words)

        return data

    def remove_punctuation(self, data):
        transtab = str.maketrans(dict.fromkeys(self.punctuation, ''))

        for entry in data:
            tweet = str(entry["tweet"])
            entry["tweet"] = tweet.translate(transtab)

        return data

    def pad(self, data):
        max_tweet_length = max(len(entry["tweet"]) for entry in data)
        # print("max tweet length:", max_tweet_length)

        for entry in data:
            current_tweet = str(entry["tweet"])
            padded_tweet = current_tweet.ljust(max_tweet_length)
            entry["tweet"] = padded_tweet

        return data

    def preprocess(self):
        removed_stop_words = self.remove_stop_words(self.data_list)
        tokenized_data = self.tokenize(removed_stop_words)
        lemmatized_data = self.lemmatize(tokenized_data)
        data_no_punctuation = self.remove_punctuation(lemmatized_data)
        padded_data = self.pad(data_no_punctuation)

        # Convert labels to integers
        for entry in padded_data:
            entry["label"] = int(entry["label"])

        return padded_data
