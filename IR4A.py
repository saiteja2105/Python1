# IR4A.py CS5154/6054 cheng 2021
# following Chapter 1 of Blueprints for Text Anlytics Using Python
# Blueprints for Word Frequency Analysis
# using un-general-debates-blueprint.csv

import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from matplotlib import pyplot as plt

stopwords = set(nltk.corpus.stopwords.words('english'))

df = pd.read_csv("un-general-debates-blueprint.csv")

def tokenize(text):
	return re.findall(r'[\w-]*\p{L}[\w-]*', text)

def remove_stop(tokens):
	return [t for t in tokens if t not in stopwords]

pipeline = [str.lower, tokenize, remove_stop]

def prepare(text, pipeline):
	tokens = text
	for transform in pipeline:
		tokens = transform(tokens)
	return tokens

df['tokens'] = df['text'].apply(prepare, pipeline=pipeline)

counter = Counter()
df['tokens'].map(counter.update)

wc = WordCloud()
wc.generate_from_frequencies(counter)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

