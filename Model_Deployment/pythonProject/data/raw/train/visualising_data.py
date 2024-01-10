import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

labels_path = 'F:/University/Courses/ML_Practical/Data/trial/us_trial.labels'
text_path = 'F:/University/Courses/ML_Practical/Data/trial/us_trial.text'

labels = pd.read_csv('your_data.labels', header=None, names=['Emoji'])
tweets = pd.read_csv('your_data.text', header=None, names=['Tweet'])

df = pd.concat([tweets, labels], axis=1)

df.head()
