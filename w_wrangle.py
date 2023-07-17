#Standard imports

import pandas as pd
import numpy as np

#Imports for webscraping

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from webdriver_manager.chrome import ChromeDriverManager

import time as t
import os

import requests
from bs4 import BeautifulSoup

#NLP tokens for processing

import unicodedata
import re
import json
import nltk

import spacy

from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

#Sentiment Analysis

import nltk.sentiment

#Visualizations

import matplotlib.pyplot as plt
import seaborn as sns

def acquire_commentary():
    test_df_2 = pd.read_csv('player_commentaries_health.csv', index_col=0)
    df = pd.read_csv('player_commentaries.csv', index_col=0)

    combined_df = df.merge(test_df_2, left_on = 'player_name', right_on = 'player_name')

    combined_df['player_commentary_z'] = combined_df['player_commentary_x'] + combined_df['player_commentary_y']

    combined_df = combined_df[['player_name', 'player_commentary_z']].rename(columns={'player_commentary_z': 'player_commentary'})

    cap = pd.read_csv('pivot_final_2.csv')

    cap = cap[['name', 'year', 'percent_of_cap']].sort_values(by=['name', 'year']).drop_duplicates('name', keep='last')

    cap = cap.merge(combined_df, left_on='name', right_on='player_name')

    conditions = [
    (cap['percent_of_cap'] < 2.7),
    (cap['percent_of_cap'] >= 2.7) & (cap['percent_of_cap'] < 12.9),
    (cap['percent_of_cap'] >= 12.9)
    ]

    # create a list of the values we want to assign for each condition
    values = ['low', 'mid', 'high']

    # create a new column and use np.select to assign values to it using our lists as arguments
    cap['tier'] = np.select(conditions, values) 

    cap = cap.drop(columns='name')[['player_name', 'year', 'player_commentary', 'percent_of_cap', 'tier']]

    return cap

def clean_strings(string, exclude_words=[], extra_words=[]):    
    #Initialize NLP spacy object for lemmatization
    nlp = spacy.load('en_core_web_sm')
    
    #Basic clean
    lower_string = string.lower()
    
    normal_string = unicodedata.normalize('NFKD', lower_string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    
    normal_no_chars_string = re.sub(r'[^a-z0-9\s]', '', normal_string.replace("'", " "))
    
    #Tokenize
    ttt = ToktokTokenizer()
    tokens = ttt.tokenize(normal_no_chars_string, return_str=True)
    
    doc = nlp(tokens)
    
    #Lemmatize
    lemmatized_text = ' '.join([token.lemma_ for token in doc])
    
    lemmatized_text
    
    #Remove stopwords
    stopword_list = stopwords.words('english')
    
    #Removing words from list
    stopword_list = [word for word in stopword_list if word not in exclude_words]
    
    #Adding words to list
    
    for word in extra_words:
        stopword_list.append(word)
    
    no_stop_words = [word for word in lemmatized_text.split() if word not in stopword_list]
    
    no_stop_string = ' '.join(no_stop_words)
    
    return no_stop_string

unigram_stopwords = [' game',
                    ' nfl',
                    ' season',
                    ' super',
                    ' bowl',
                    ' quarterback',
                    ' play',
                    ' team',
                    ' I',
                    ' performance',
                    ' year',
                    ' week',
                    'first',
                    'take', 
                    'last', 
                     'good',
                     'time',
                     'start',
                     'two',
                     'qb',
                     'new',
                     'throw',
                     'make', 
                     'pass',
                     'get',
                     'go', 
                     'touchdown',
                     'yard', 
                     'one', 
                     'win']

bi_tri_stopwords = [' I', ' blake',
 ' bortle',
 ' cody',
 ' kessler',
 ' mike',
 ' white',
 ' gardner',
 ' minshew',
 ' joe',
 ' flacco',
 ' philip',
 ' river',
 ' ryan',
 ' fitzpatrick',
 ' josh',
 ' rosen',
 ' drew',
 ' lock',
 ' deshaun',
 ' watson',
 ' ben',
 ' roethlisberger',
 ' sam',
 ' darnold',
 ' carson',
 ' palmer',
 ' matt',
 ' barkley',
 ' trevor',
 ' lawrence',
 ' taylor',
 ' heinicke',
 ' jalen',
 ' hurt',
 ' geno',
 ' smith',
 ' marcus',
 ' mariota',
 ' ryan',
 ' tannehill',
 ' patrick',
 ' mahome',
 ' tua',
 ' tagovailoa',
 ' justin',
 ' field',
 ' nick',
 ' mullen',
 ' baker',
 ' mayfield',
 ' david',
 ' blough',
 ' alex',
 ' smith',
 ' brock',
 ' purdy',
 ' daniel',
 ' jones',
 ' jared',
 ' goff',
 ' kirk',
 ' cousin',
 ' tyler',
 ' huntley',
 ' andrew',
 ' luck',
 ' jay',
 ' cutler',
 ' dwayne',
 ' haskin',
 ' aaron',
 ' rodger',
 ' russell',
 ' wilson',
 ' carson',
 ' wentz',
 ' matt',
 ' ryan',
 ' brock',
 ' osweiler',
 ' josh',
 ' mccown',
 ' colin',
 ' kaepernick',
 ' tyrod',
 ' taylor',
 ' cooper',
 ' rush',
 ' mac',
 ' jones',
 ' joe',
 ' burrow',
 ' drew',
 ' stanton',
 ' eli',
 ' manning',
 ' kyle',
 ' allen',
 ' josh',
 ' allen',
 ' dak',
 ' prescott',
 ' matthew',
 ' stafford',
 ' mitchell',
 ' trubisky',
 ' andy',
 ' dalton',
 ' brett',
 ' hundley',
 ' nick',
 ' fole',
 ' tom',
 ' brady',
 ' c.j.',
 ' beathard',
 ' sam',
 ' bradford',
 ' robert',
 ' griffin',
 ' iii',
 ' blaine',
 ' gabbert',
 ' mason',
 ' rudolph',
 ' justin',
 ' herbert',
 ' jamei',
 ' winston',
 ' jimmy',
 ' garoppolo',
 ' brian',
 ' hoyer',
 ' kyler',
 ' murray',
 ' brandon',
 ' allen',
 ' case',
 ' keenum',
 ' jacoby',
 ' brissett',
 ' bryce',
 ' petty',
 ' trevor',
 ' siemian',
 ' jeff',
 ' driskel',
 ' drew',
 ' bree',
 ' tom',
 ' savage',
 ' mike',
 ' glennon',
 ' deshone',
 ' kizer',
 ' teddy',
 ' bridgewater',
 ' zach',
 ' wilson',
 ' derek',
 ' carr',
 ' kenny',
 ' pickett',
 ' cam',
 ' newton',
 ' devlin',
 ' hodge',
 ' davis',
 ' mill',
 ' lamar',
 ' jackson',
 ' poll 0 vote quick link',
 ' follow share show comment',
 ' feedback thank first one comment',
 ' view post instagram',
 ' instagram instagram post',
 ' quote please credit',
 ' w w w',
 ' new england patriot',
 ' new orleans saint', 
 ' new york jet',
 ' kansas city chief',
 ' green bay packer',
 ' san francisco 49er',
 ' tampa bay buccaneer',
 ' los angeles ram',
 ' las vegas raider',
 ' new orleans saints',
 ' new york giants',
 ' new york jets',
 ' los angeles charger',
 ' washington football team',
 ' los angeles rams',
 ' san francisco 49ers',
 ' kansas city chiefs',
 ' carolina panther',
 ' denver broncos',
 ' pittsburgh steeler',
 ' chicago bear',
 ' philadelphia eagle',
 ' buffalo bill',
 ' houston texans',
 ' indianapolis colt',
 ' cleveland brown',
 ' 2023 arnold classic',
 ' mr olympia',
 ' miami dolphin',
 ' jacksonville jaguar',
 ' let take look',
 ' san diego charger',
 ' today2835 80420 yards12 ypa6 tdabsolute masterclass team',
 ' need 10ajmccarron post one great spring football',
 ' dallas cowboy',
 ' seattle seahawks',
 ' baltimore raven',
 ' washington commander',
 ' tennessee titans',
 ' detroit lion',
 ' arizona cardinal',
 ' lebron james',
 ' golden state warrior',
 ' new york giant',
 ' prime videoalso available',
 ' nfl nfl november 18 2022',
 ' gmt18 nov 2022',
 ' yard one touchdown',
 ' yard two touchdown',
 ' yard three touchdown',
 ' yard four touchdown',
 ' mccarron statline time performance',
 ' l l l',
 ' nation jason kelce say still beat', 
 ' sb loss speak still',
 ' twittercomjameslarsenpfnjame larsenjameslarsenpfnaj xfl',
 ' 2022 nfl season',
 ' 2021 nfl season',
 ' 2020 nfl season',
 ' 2022 nfl season',
 ' 20202021 nfl season',
 ' 2022 nfl draft',
 ' 2021 nfl draft',
 ' minnesota viking',
 ' ohio state',
 ' week 1']

def get_grams(df):
    high_words = ' '.join(df[df.tier == 'high'].player_commentary)
    mid_words = ' '.join(df[df.tier == 'mid'].player_commentary)
    low_words = ' '.join(df[df.tier == 'low'].player_commentary)

    unigram_high_words = high_words
    unigram_mid_words = mid_words
    unigram_low_words = low_words

    bi_tri_high_words = high_words
    bi_tri_mid_words = mid_words
    bi_tri_low_words = low_words   

    for word in unigram_stopwords:
        unigram_high_words = unigram_high_words.replace(word, '')
        unigram_mid_words = unigram_mid_words.replace(word, '')
        unigram_low_words = unigram_low_words.replace(word, '')

    for word in bi_tri_stopwords:
        bi_tri_high_words = bi_tri_high_words.replace(word, '')
        bi_tri_mid_words = bi_tri_mid_words.replace(word, '')
        bi_tri_low_words = bi_tri_low_words.replace(word, '')

    return unigram_high_words, unigram_mid_words, unigram_low_words, bi_tri_high_words, bi_tri_mid_words, bi_tri_low_words

def viz_unigrams(unigram_high_words, unigram_mid_words, unigram_low_words):
    plt.figure(figsize=(12,5))
    plt.suptitle('Top 20 Most Common Unigrams')

    plt.subplot(131)
    pd.Series(unigram_high_words.split()).value_counts().head(20).plot.barh()
    plt.title('High Percentage Caps')

    plt.subplot(132)
    pd.Series(unigram_mid_words.split()).value_counts().head(20).plot.barh()
    plt.title('Mid Percentage Caps')

    plt.subplot(133)
    pd.Series(unigram_low_words.split()).value_counts().head(20).plot.barh()
    plt.title('Low Percentage Caps')

    plt.tight_layout()
    plt.show()


def viz_bigrams(bi_tri_high_words, bi_tri_mid_words, bi_tri_low_words):
    plt.figure(figsize=(12,5))
    plt.suptitle('Most Common Bigrams')

    plt.subplot(131)
    top_20_bigrams_high = (pd.Series(nltk.ngrams(bi_tri_high_words.split(), 2))
                        .value_counts()
                        .head(20))
    top_20_bigrams_high[1:20].plot.barh()
    plt.title('High Percentage Caps')

    plt.subplot(132)
    top_20_bigrams_mid = (pd.Series(nltk.ngrams(bi_tri_mid_words.split(), 2))
                        .value_counts()
                        .head(20))
    top_20_bigrams_mid[1:20].plot.barh()
    plt.title('Mid Percentage Caps')

    plt.subplot(133)
    top_20_bigrams_low = (pd.Series(nltk.ngrams(bi_tri_low_words.split(), 2))
                        .value_counts()
                        .head(20))
    top_20_bigrams_low[1:20].plot.barh()
    plt.title('Low Percentage Caps')

    plt.tight_layout()
    plt.show()  

def viz_trigrams(bi_tri_high_words, bi_tri_mid_words, bi_tri_low_words):
    plt.figure(figsize=(12,5))
    plt.suptitle('Most Common Trigrams')

    plt.subplot(131)
    top_20_trigrams_high = (pd.Series(nltk.ngrams(bi_tri_high_words.split(), 3))
                        .value_counts()
                        .head(20))
    top_20_trigrams_high.plot.barh()
    plt.title('High Percentage Caps')

    plt.subplot(132)
    top_20_trigrams_mid = (pd.Series(nltk.ngrams(bi_tri_mid_words.split(), 3))
                        .value_counts()
                        .head(20))
    top_20_trigrams_mid.plot.barh()
    plt.title('Mid Percentage Caps')

    plt.subplot(133)
    top_20_trigrams_low = (pd.Series(nltk.ngrams(bi_tri_low_words.split(), 3))
                        .value_counts()
                        .head(20))
    top_20_trigrams_low.plot.barh()
    plt.title('Low Percentage Caps')

    plt.tight_layout()
    plt.show()

def get_sia_scores(df):

    sia = nltk.sentiment.SentimentIntensityAnalyzer()

    df['sentiment'] = df['player_commentary'].apply(lambda doc: sia.polarity_scores(doc)['compound'])

    return df.groupby('tier').mean('sentiment')