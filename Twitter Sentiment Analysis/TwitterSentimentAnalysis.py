# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:12:08 2019

@author: Vishal
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import seaborn as sns
import time
import csv
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import nltk
#from nltk.stem.porter import *
translator = Translator()

analyser = SentimentIntensityAnalyzer()

"""analyser.polarity_scores("This movie is worst")
analyser.polarity_scores("This movie is WORST")
analyser.polarity_scores("This is the WORST MOVIE EVER!!")



translator.translate('कशी आहेस?').text

text = translator.translate('bahot bekar movie he').text
text

analyser.polarity_scores(text)"""

def sentiment_analyzer_scores(text, engl = True):
	if engl:
		trans = text
	else:
		trans = translator.translate(text).text
	
	score = analyser.polarity_scores(trans)
	lb = score['compound']
	if (lb >= 0.05):
		return 1
	elif ((lb > -0.05) and (lb < 0.05)):
		return 0
	else:
		return -1
	

import tweepy

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

"""tweets = api.user_timeline('@narendramodi', count = 5, tweet_mode = 'extended')
for t in tweets:
    print(t.full_text)
    print()"""

def list_tweets(user_id, count, prt = False):
	tweets = api.user_timeline("@" + user_id, count = count, tweet_mode = 'extended')
	tw = []
	for t in tweets:
		tw.append(t.full_text)
		if prt:
			print(t.full_text)
			print()
	return tw

user_id = 'realDonaldTrump'
count = 200
tw_trump = list_tweets(user_id, count)

tw_trump[2]

import re
import numpy as np

def remove_pattern(input_txt, pattern):
	r = re.findall(pattern, input_txt)
	for i in r:
		input_txt = re.sub(i, '', input_txt)
	return input_txt

def clean_tweets(lst):
	#remove twitter return handles (RT @xxx:)
	lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
	#Remove twitter handles (@xxx)
	lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
	#Remove URL links
	lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
	#Remove special chars, nos., punctuations (except #)
	lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
	return lst

tw_trump = clean_tweets(tw_trump)									   
print(tw_trump[2])										        	

print(sentiment_analyzer_scores(tw_trump[2]))

def anl_tweets(lst, title = 'Tweets Sentiment', engl = True):
	sents = []
	for tw in lst:
		try:
			st = sentiment_analyzer_scores(tw, engl)
			sents.append(st)
		except:
			sents.append(0)
	ax = sns.distplot(
			sents,
			kde = False,
			 bins = 3)
	ax.set(xlabel = 'Negative           Neutral              Positive',
		ylabel = '#Tweets',
		title = "Tweets of @" + title)
	return sents

tw_trump_sent = anl_tweets(tw_trump, user_id)

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def word_cloud(wd_lst):
	stopwords = set(STOPWORDS)
	all_words = ' '.join([text for text in wd_lst])
	wordcloud = WordCloud(
			background_color = 'black',
			stopwords = stopwords,
			width = 1600,
			height = 800,
			random_state = 21,
			colormap = 'jet',
			max_words = 50,
			max_font_size = 200).generate(all_words)
	
	plt.figure(figsize = (12, 10))
	plt.axis('off')
	plt.imshow(wordcloud, interpolation = "bilinear")
	
word_cloud(tw_trump)


#Let's try for Modi
user_id = "narendramodi"
tw_modi = list_tweets(user_id, count)
tw_modi = clean_tweets(tw_modi)
tw_modi_sent = anl_tweets(tw_modi, user_id)
word_cloud(tw_modi)

user_id = "RahulGandhi"
tw_rahul = list_tweets(user_id, count)
tw_rahul = clean_tweets(tw_rahul)
tw_rahul_sent = anl_tweets(tw_rahul, user_id)
word_cloud(tw_rahul)

def twitter_stream_listener(file_name,
							filter_track,
							follow = None,
							locations = None,
							languages = None,
							time_limit = 20):
	class CustomStreamListener(tweepy.StreamListener):
		def __init__(self, time_limit):
			self.start_time = time.time()
			self.limit = time_limit
			#self.saveFile = open('a.json', 'a')
			super(CustomStreamListener, self).__init__()
			
		def on_status(self, status):
			if(time.time() - self.start_time) < self.limit:
				print(".", end = "")
				#Writing status data
				with open(file_name, 'a', encoding="utf-8") as f:
					writer = csv.writer(f)
					writer.writerow([
							status.author.screen_name, status.created_at, status.text])
			else:
				print("\n\n[INFO] Closing file and ending streaming")
				return False
		
		def on_error(self, status_code):
			if status_code == 420:
				print("Encountered error 420. Disconnecting the stream")
				#Returning false in on_data disconnects the stream
				return False
			else:
				print("Encountered error with status code: {}".format(status_code))
				return True  #Don't kill the stream
		
		def on_timeout(self):
			print("Timeout...")
			return True   #Don't kill the stream
		
		
	#Writing csv titles
	print("\n[INFO] Open file: [{}] and starting {} seconds of streaming for {}\n"
	   .format(file_name, time_limit, filter_track))
	with open(file_name, 'w', encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(['author', 'date', 'text'])
		
		streamingAPI = tweepy.streaming.Stream(
				auth, CustomStreamListener(time_limit = time_limit))
		streamingAPI.filter(
				track = filter_track,
				follow = follow,
				locations = locations,
				languages = languages,)
		f.close()
		
filter_track = ['trump', 'wall']
file_name = 'tweets-trump_wall.csv'
twitter_stream_listener(file_name, filter_track, time_limit = 60)
			
df_tws = pd.read_csv(file_name)
df_tws.shape
df_tws.head()

df_tws['text'] = clean_tweets(df_tws['text'])

df_tws['sent'] = anl_tweets(df_tws.text)

df_tws.head()

tws_pos = df_tws['text'][df_tws['sent'] == 1]
word_cloud(tws_pos)

tws_neg = df_tws['text'][df_tws['sent'] == -1]
word_cloud(tws_neg)

def hashtag_extract(x):
	hashtags = []
	#Loop over the words in tweets
	for i in x:
		ht = re.findall(r"#(\w+)", i)
		hashtags.append(ht)
	return hashtags

#Extracting hashtags from positive tweets
HT_positive = hashtag_extract(df_tws['text'][df_tws['sent'] == 1])

#Extracting hashtags from negative tweets
HT_negative = hashtag_extract(df_tws['text'][df_tws['sent'] == -1])
  
#unnesting list
HT_positive = sum(HT_positive, [])
HT_negative = sum(HT_negative, [])

#Positive Tweets
a = nltk.FreqDist(HT_positive)
d = pd.DataFrame({'Hashtag': list(a.keys()),'Count': list(a.values())})

sns.set()
#Selecting top 10 most frequent hashtags
d = d.nlargest(columns = "Count", n = 10)
plt.figure(figsize = (16, 5))
ax = sns.barplot(data = d, x = "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
ax.set(title = 'Positive hashtags')
plt.show()

# Negative Tweets
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 10 most frequent hashtags
e = e.nlargest(columns = "Count", n = 10)   
plt.figure(figsize = (16,5))
ax = sns.barplot(data = e, x = "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
ax.set(title = 'Negative hashtags')
plt.show()