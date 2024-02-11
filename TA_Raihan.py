pip install python

import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

#page
st.set_page_config(
    page_title="SentimentAnalysis_MuhammadRaihanPermadi",
    layout="wide"
)

#PART1
#header1
st.header("SENTIMENT ANALYSIS PADA MEDIA SOSIAL TWITTER TERHADAP PEMILU TAHUN 2024")

#column1
with st.container(border=True):
    st.write("Labeling (Positif, Negatif, dan Netral) Tweet Publik Sebelum Menggunakan Metode Naive Bayes Classifier")

col_1, col_2 = st.columns([1.5, 1])

#csv
with col_1:
    with st.container(border=True):
        data1 = pd.read_csv('pemilu_2024_class.csv')
        st.dataframe(data1)

#for col_2_1
import preprocessor as p
from textblob import TextBlob
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

data_labeling = pd.read_csv('pemilu_2024_ok.csv', index_col=0)

data_tweet = list(data_labeling['full_text_eng'])
polaritas = 0
        
status = []
total_positif = total_negatif = total_netral = total = 0
        
for i, tweet in enumerate(data_tweet):
    analysis = TextBlob(tweet)
    polaritas += analysis.polarity

    if analysis.sentiment.polarity > 0.0:
        total_positif += 1
        status.append('Positif')
    elif analysis.sentiment.polarity == 0.0:
        total_netral += 1
        status.append('Netral')
    else:
        total_negatif += 1
        status.append('Negatif')
    
    total += 1

#for col_2_2
sns.set_theme()

labels = ['Positif', 'Negatif', 'Netral']
counts = [total_positif, total_negatif, total_netral]

def show_bar_chart(labels, counts, title):
    fig, ax = plt.subplots(figsize=(12, 9))
    bars = ax.bar(labels, counts, color=('#0962ba', '#db0202', '#8c8c8c'))

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Jumlah')
        ax.set_title('Tanggapan Positif, Negatif, dan Netral Publik Twitter Tentang Pemilu 2024')

show_bar_chart(labels, counts, "Sentiment Analysis")


with col_2:
    with st.container(border=True):
        st.write(f'Hasil Analisis Data :\nPositif = {total_positif},\nNetral = {total_netral},\nNegatif = {total_negatif}')
        st.write(f'\nTotal Data : {total}')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig=False)
        show_bar_chart(labels, counts, "Sentiment Analysis")

#PART2
#column2
with st.container(border=True):
    st.write("Labeling (Positif, Negatif, dan Netral) Tweet Publik Seseudah Menggunakan Metode Naive Bayes Classifier")

col_1, col_2 = st.columns([1.5, 1])

#csv
with col_1:
    with st.container(border=True):
        data2 = pd.read_csv('pemilu_2024_naivebayes.csv')
        st.dataframe(data2)

#for col_2_1
data_class = pd.read_csv('pemilu_2024_class.csv')        

dataset = data_class.drop(['full_text'], axis=1, inplace=False)
dataset = [tuple(x) for x in dataset.to_records(index=False)]

##########
import random

set_positif = []
set_negatif = []
set_netral = []

for n in dataset:
    if(n[1] == 'Positif'):
        set_positif.append(n)
    elif(n[1] == 'Negatif'):
        set_negatif.append(n)
    else:
        set_netral.append(n)

set_positif = random.sample(set_positif, k=int(len(set_positif)/2))
set_negatif = random.sample(set_negatif, k=int(len(set_negatif)/2))
set_netral = random.sample(set_netral, k=int(len(set_netral)/2))

train = set_positif + set_negatif + set_netral

train_set = []

for i in train:
    train_set.append(n)


from textblob.classifiers import NaiveBayesClassifier
cls = NaiveBayesClassifier(train_set)

#for col_2_2
data_tweet = list(data_class['full_text_eng'])
polaritas = 0

status = []
total_positif = total_negatif = total_netral = total = 0

for i, tweet in enumerate(data_tweet):
    analysis = TextBlob(tweet, classifier=cls)

    if analysis.classify() == 'Positif':
        total_positif += 1
    elif analysis.classify() == 'Netral':
        total_netral += 1
    else:
        total_negatif += 1

    status.append(analysis.classify())
    total += 1

#for col_2_3
sns.set_theme()

labels = ['Positif', 'Negatif', 'Netral']
counts = [total_positif, total_negatif, total_netral]

def show_bar_chart(labels, counts, title):
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(labels, counts, color=('#0962ba', '#db0202', '#8c8c8c'))

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(f'{count}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
        
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Jumlah')
        ax.set_title('Tanggapan Positif dan Negatif Publik Twitter Tentang Pemilu 2024')

show_bar_chart(labels, counts, "Sentiment Analysis")

with col_2:
    with st.container(border=True):
        st.write("Akurasi Test", cls.accuracy(dataset))
        st.write(f'\nHasil Analisis Data:\nPositif = {total_positif}\nNetral = {total_netral}\nNegatif = {total_negatif}')
        st.write(f'\nTotal Data : {total}')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig=False)
        show_bar_chart(labels, counts, "Sentiment Analysis")
