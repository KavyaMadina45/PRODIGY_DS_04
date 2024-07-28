# Twitter and News Sentiment Analysis

This project involves cleaning and analyzing tweets and news articles for sentiment analysis. The project includes data preprocessing, removing unnecessary data, handling missing values, performing sentiment analysis using VADER, and visualizing the results.

## Project Description

The project demonstrates how to:
1. Load and inspect a Twitter dataset.
2. Clean the dataset by removing URLs, mentions, special characters, and stopwords.
3. Perform sentiment analysis on cleaned tweets using the VADER sentiment analyzer.
4. Fetch and analyze sentiment of news articles using the News API and VADER sentiment analyzer.
5. Visualize sentiment distribution using various plots such as bar plots and pie charts.

## Installation Instructions

To run this project, you need to have Python installed along with the following libraries:
- pandas
- nltk
- scikit-learn
- matplotlib
- seaborn
- requests

You can install these libraries using pip:
```sh
pip install pandas nltk scikit-learn matplotlib seaborn requests
```
**Usage**
1) Place your dataset files (twitter.csv, news_sentiment_analysis.xlsx) in an accessible directory.
2) Update the file paths in the code to match the locations of your dataset files.
3) Run the script to perform data cleaning, sentiment analysis, and visualization.

**Example code**
```sh
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Download stopwords list
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv('/home/kavya/Downloads/twitter.csv')

def clean_tweet(tweet):
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = ' '.join(tweet.split())
    return tweet

def remove_stop_words(tweet):
    words = tweet.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

df['cleaned_tweet'] = df['tweet'].apply(clean_tweet)
df['cleaned_tweet'] = df['tweet'].apply(remove_stop_words)
df.to_csv('/home/kavya/Downloads/cleaned_twitter.csv', index=False)

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['cleaned_tweet'].apply(lambda x: 'Positive' if sia.polarity_scores(x)['compound'] >= 0.05 else ('Negative' if sia.polarity_scores(x)['compound'] <= -0.05 else 'Neutral'))

plt.figure(figsize=(8, 6))
color=['deeppink','green','blue']
sns.countplot(x='sentiment', data=df, palette=color, hue='sentiment', legend=False)
plt.title('Sentiment Distribution of Tweets')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# News API sentiment analysis
url = 'https://newsapi.org/v2/everything'
params = {
    'q': 'climate change',
    'apiKey': '9b74470fa0df4a8fbc205ad9fbe9b95e'
}
response = requests.get(url, params=params)
articles = response.json()['articles']
analyzer = SentimentIntensityAnalyzer()

for article in articles:
    title = article['title']
    sentiment = analyzer.polarity_scores(title)
    print(f"Title: {title}")
    print(f"Sentiment: {sentiment}")

# Plot sentiment analysis from Excel data
x = '/home/kavya/news_sentiment_analysis.xlsx'
df = pd.read_excel(x)
grouped_df = df.groupby('Title').mean().reset_index()
grouped_df = grouped_df.head(30)

plt.figure(figsize=(12, 8))
sns.barplot(y='Title', x='Sentiment_Positive', data=grouped_df, color='blue', alpha=1.0, label='Positive')
sns.barplot(y='Title', x='Sentiment_Neutral', data=grouped_df, color='gray', alpha=0.8, label='Neutral')
sns.barplot(y='Title', x='Sentiment_Negative', data=grouped_df, color='red', alpha=0.8, label='Negative')
plt.xticks(rotation=90)
plt.title('Sentiment Analysis of News Titles')
plt.ylabel('Title')
plt.xlabel('Sentiment Score')
plt.legend()
plt.show()

# Pie chart for sentiment distribution
sum_positive = df['Sentiment_Positive'].sum()
sum_negative = df['Sentiment_Negative'].sum()
sum_neutral = df['Sentiment_Neutral'].sum()
sum_compound = df['Sentiment_Compound'].sum()
dict = {'Positive': sum_positive, 'Negative': sum_negative, 'Neutral': sum_neutral, 'Compound': sum_compound}
labels = dict.keys()
sizes = dict.values()
colors = ["green", "red", "gray", "deeppink"]
explode = [0.1, 0, 0, 0]
plt.pie(sizes, labels=labels, colors=colors, explode=explode, shadow=True, autopct='%1.1f%%')
plt.show()
```
**Features**
* Data Loading: Efficiently loads Twitter and news datasets from CSV and Excel files.
* Data Cleaning: Removes URLs, mentions, special characters, and stopwords from tweets.
* Sentiment Analysis: Performs sentiment analysis on cleaned tweets and news titles using VADER.
* Visualization: Visualizes sentiment distribution using bar plots and pie charts.

**Contributing**

If you want to contribute to this project, please follow these steps:

1) Fork the repository.
2) Create a new branch (git checkout -b feature-branch).
3) Make your changes.
4) Commit your changes (git commit -m 'Add new feature').
5) Push to the branch (git push origin feature-branch).
6) Create a new Pull Request

**License**
This project is licensed under the MIT License - see the LICENSE file for details.

**Contact Information**
For any questions or issues, please contact Kavya at madinakavya6@gmail.com
