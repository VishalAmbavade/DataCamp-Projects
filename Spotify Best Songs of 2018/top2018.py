# -*- coding: utf-8 -*-
"""
Created on Tue May 21 13:12:08 2019

@author: Vishal
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import pandas as pd

cid = "" 
secret = ""

client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

spotify = spotipy.Spotify()

def get_audio_features(spotify, ids):
    music_features = pd.DataFrame(spotify.audio_features(ids))
    return music_features

playlists = sp.user_playlists('vishal_a')
while (playlists):
    for i, playlist in enumerate(playlists['items']):
        print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['uri'], playlist['name']))
    if(playlists['next']):
        playlists = sp.next(playlists)
    else:
        playlists = None
    
track_results = sp.user_playlist("vishal_a", "0Kzab90DeFe6dOIqcdw0mS")


df_tracks = pd.DataFrame([[t["track"]["id"], t["track"]["name"], t["track"]["artists"][0]["id"],
                           t["track"]["artists"][0]["name"], t["track"]["album"]["name"], 
                           t["track"]["popularity"]]
    for t in track_results['tracks']['items']],
    columns = ["id", "song_name", "artist_id", "artist_name", "album_name", "popularity"])
    
df_tracks["norm_popularity"] = df_tracks["popularity"] / 100

print(df_tracks.head())

import plotly.plotly as py
import plotly.graph_objs as go

trace = go.Histogram(x = df_tracks['artist_name'])
data = [trace]
py.iplot(data)

#print(df_tracks['artist_name'][:2])
#print(df_tracks.info())

for i in range(len(df_tracks)):
    try:
        if (df_tracks['popularity'][i] >= 60):
            print(df_tracks['song_name'][i], df_tracks['popularity'][i])
    except Exception:
        pass
    
"""def get_features_df(sp, track_ids):
    feature_list = []
    i = 0
    while track_ids:
        print("Call #{} for audio features", format(i + 1))
        features_results = sp.audio_features(track_ids[:API_LIMIT])"""
        
print(len(df_tracks['artist_name'].unique()))
track_ids = df_tracks['id'].unique().tolist()

df_features = get_audio_features(sp, track_ids)
print(df_features.head())

print(df_features.describe())

#Normalizing temp column
df_features['norm_tempo'] = (df_features['tempo'] - 24) / 176

df_features['popularity'] = df_tracks['norm_popularity']

df_features.drop(['analysis_url', 'track_href', 'uri'], axis = 1, inplace = True)

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(df_features.corr(), annot = True, ax = ax)

sns.set()
f, axes = plt.subplots(1, 3, sharey = True, figsize = (15, 4))
sns.boxplot(df_features['valence'], ax = axes[0])
sns.boxplot(df_features['danceability'], ax = axes[1])
sns.boxplot(df_features['loudness'], ax = axes[2])

f, axes = plt.subplots(1, 3, sharey = True, figsize = (15, 4))
sns.boxplot(df_features['acousticness'], ax = axes[0])
sns.boxplot(df_features['energy'], ax = axes[1])
sns.boxplot(df_features['liveness'], ax = axes[2])

f, axes = plt.subplots(1, 3, sharey = True, figsize = (15, 4))
sns.boxplot(df_features['speechiness'], ax = axes[0])
sns.boxplot(df_features['instrumentalness'], ax = axes[1])
sns.boxplot(df_features['norm_tempo'], ax = axes[2])

plt.show()

f, axes = plt.subplots(1, 3, sharey = True, figsize = (15, 4))
sns.distplot(df_features['valence'], ax = axes[0])
sns.distplot(df_features['danceability'], ax = axes[1])
sns.distplot(df_features['loudness'], ax = axes[2])

f, axes = plt.subplots(1, 3, sharey = True, figsize = (15, 4))
sns.distplot(df_features['acousticness'], ax = axes[0])
sns.distplot(df_features['energy'], ax = axes[1])
sns.distplot(df_features['liveness'], ax = axes[2])

f, axes = plt.subplots(1, 3, sharey = True, figsize = (15, 4))
sns.distplot(df_features['speechiness'], ax = axes[0])
sns.distplot(df_features['instrumentalness'], ax = axes[1])
sns.distplot(df_features['popularity'], ax = axes[2])

plt.show()

print(df_features.describe())

#===================================================================================#
#TOP SONGS OF 2018#

top_2018_df = pd.read_csv('top2018.csv')

top_2018_df = top_2018_df.drop(['duration_ms', 'time_signature'], axis = 1)
top_2018_df.head()

trace1 = go.Histogram(x = top_2018_df['artists'])
data1 = [trace1]
py.plot(data1)

top_2018_df.loc[4]

#Top 4 artists with most number of songs in 2018
top_2018_df['artists'].value_counts().head(4)

trace2 = go.Histogram(x = top_2018_df['key'])
data2 = [trace2]
py.plot(data2)

top_2018_df['name']

f, axes = plt.subplots(1, 3, sharey = True, figsize = (15, 4))

sns.distplot(top_2018_df['valence'], ax = axes[0])
sns.distplot(top_2018_df['danceability'], ax = axes[1])
sns.distplot(top_2018_df['loudness'], ax = axes[2])

f, axes = plt.subplots(1, 3, sharey = True, figsize = (15, 4))
sns.distplot(top_2018_df['acousticness'], ax = axes[0])
sns.distplot(top_2018_df['energy'], ax = axes[1])
sns.distplot(top_2018_df['liveness'], ax = axes[2])

f, axes = plt.subplots(1, 3, sharey = True, figsize = (15, 4))
sns.distplot(top_2018_df['speechiness'], ax = axes[0])
sns.distplot(top_2018_df['instrumentalness'], ax = axes[1])
plt.show()

sns.heatmap(top_2018_df.corr())

valence_avg = top_2018_df['valence'].mean()
dance_avg = top_2018_df['danceability'].mean()
acoustic_avg = top_2018_df['acousticness'].mean()
energy_avg = top_2018_df['energy'].mean()
speech_avg = top_2018_df['speechiness'].mean()
liveness_avg = top_2018_df['liveness'].mean()

valence_avg2 = df_features ['valence'].mean()
dance_avg2 = df_features ['danceability'].mean()
acoustic_avg2 = df_features ['acousticness'].mean()
energy_avg2 = df_features ['energy'].mean()
speech_avg2 = df_features ['speechiness'].mean()
liveness_avg2 = df_features ['liveness'].mean()

Data = [go.Scatterpolar(r = [valence_avg, dance_avg, acoustic_avg, energy_avg, speech_avg, liveness_avg], 
                        theta = ['Valence', 'Danceability', 'Acousticness', 'Energy', 'Speechiness', 'liveness'],
                        fill = 'toself', name = 'Group A'), 
    go.Scatterpolar(r = [valence_avg2, dance_avg2, acoustic_avg2, energy_avg2, speech_avg2, liveness_avg2], 
                    theta = ['Valence', 'Danceability', 'Acousticness', 'Energy', 'Speechiness', 'liveness'], 
                    fill = 'toself', name = 'Group B')]
layout = go.Layout(
        polar = dict(radialaxis = dict(visible = True, range = [0.00, 1.00])),
        showlegend = False)

fig = go.Figure(data = Data, layout = layout)
py.plot(fig, filename = "Basic_Radar")


