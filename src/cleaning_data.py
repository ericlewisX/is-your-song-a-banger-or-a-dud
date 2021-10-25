# Imports
import pandas as pd
import numpy as np
from datetime import datetime


## Importing Files ##

# Billboard Hot 100 Dataset and Song Features. Importing tracks dataset to balance dataset before model building. 
df = pd.read_csv('Hot Stuff.csv')                            # (contains y-target)
features_df = pd.read_excel('Hot 100 Audio Features.xlsx')   # (contains X-features)
df2 = pd.read_csv('data/tracks.csv')

hot100, hot100_features, tracks = df.copy(), features_df.copy(), df2.copy()


## Cleaning ## 

# url, weekly position, instance, previous week position are all irrelevant to my goals.
hot100.drop(['url', 'Week Position', 'Instance', 'Previous Week Position'], axis=1, inplace=True)
# Url not needed. Album could be used for future projects, not for time crunch tasks. Popularity -> Data Leakage
hot100_features.drop(['spotify_track_preview_url', 'spotify_track_album', 'spotify_track_popularity'], axis=1, inplace=True)
# popularity might lead to data leakage
tracks.drop(['popularity'], axis=1, inplace=True)

# Rename Columns and extract relevant strings
tracks['artists'] = tracks['artists'].apply(lambda x: x[2:-2])
tracks['id_artists'] = tracks['id_artists'].apply(lambda x: x[2:-2])
tracks.rename(columns={'name':'Song', 'artists':'Performer', 'id':'spotify_track_id'}, inplace=True)

beta = hot100.groupby('SongID', group_keys=True).apply(lambda x: x.loc[x['Weeks on Chart'].idxmax()])
beta.reset_index(inplace =True, drop = True)

# Converts WeekID entries (type str) to (type datetime)
beta['WeekID'] = pd.to_datetime(beta['WeekID'])

hot100_features.dropna(inplace =True)

print("beta shape : ", beta.shape)
print("hot100_features shape : ", hot100_features.shape)

# Combine dataframes
combo = pd.merge(perfect100, hot100_features, how='right', on ='SongID')
print("combo shape : ", combo.shape)

combo.dropna(inplace =True)

# Fix Duplicate Columns resultant from Merge. 
combo.drop(['Song_y','Performer_y'], axis = 1, inplace=True)
combo.rename(columns={'Song_x':'Song', 'Performer_x':'Performer'}, inplace=True)

# Set new column for future binary classification
combo['top100'] = 1

# Fix Genre column to actual lists
combo['spotify_genre'] = combo['spotify_genre'].apply(lambda x: ast.literal_eval(x))

## Balancing the Dataset
# Combo, contains all the features of the unique Hot 100 songs
# Tracks contains all the features of songs that may or may not be in the Hot 100.
print("combo shape : ", combo.shape)
print("tracks shape : ", tracks.shape)

# Merge tracks and combo. More specifically this time to avoid duplicate columns. 
model_ready = pd.merge(combo, tracks, how='outer', 
                   left_on=['Song', 'Performer', 'spotify_track_id', 'danceability', 'energy', 'key', 
                            'loudness', 'mode',
                            'speechiness','acousticness','instrumentalness', 'liveness', 'valence',
                            'tempo', 'time_signature'], 
                   right_on=['Song', 'Performer', 'spotify_track_id', 'danceability', 'energy', 'key', 
                             'loudness', 'mode',
                            'speechiness','acousticness','instrumentalness', 'liveness', 'valence',
                            'tempo', 'time_signature'])
print("model_ready shape : ", model_ready.shape)


# Replace nan values with 0. (mainly for column 'top100')
model_ready['top100'] = model_ready['top100'].fillna(0)

## Optional ##
# Check if nan's are converted/ nan is in any column we care about.
# model_ready[model_ready.isna().any(axis=1)]


# Check count for songs in the Top 100
# model_ready[model_ready['top100'] == 1.0].count()

# Check count for songs not in the Top 100
# model_ready[model_ready['top100'] == 0.0].count()

model_ready.sort_values('top100', ascending =False, inplace=True)

# Merge for a perfect Dataset to predict Top100.
model_ready_df = pd.concat([model_ready[:24185],
                          model_ready[24185:].sample(24190)])

## Optional ##
# Check New dataset
# model_ready_df[model_ready_df['top100'] == 1.0].count()
#model_ready_df[model_ready_df['top100'] == 0.0].count()



## --- End --- ##
## Save Perfect Dataset to file
model_ready_df.to_csv("cleaned_and_balanced_top100")






