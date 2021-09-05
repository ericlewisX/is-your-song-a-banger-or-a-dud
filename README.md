# Is Your Song a Banger or a Dud?
Galvanize Capstone Project: Is Your Song a Banger or a Dud?

## Objective
The objecive of this project is to make a model that can predict if a song can make it to the Billboard Hot 100 Chart. The model we end up with is a Hyperparameter Tuned Random Forest Model that uses a song's features such as tempo, valence, liveliness etc., to make it's prediction. 

## Understanding the Motivation / Potential Use-Cases TODO
The primary motivation of this project is money with the target enitities being Record Labels or Music Artists.
It can be extremely lucrative if there is a way to stop either a Record Label or an individual artist from wasting their capital and time on songs that are mathematically proven to not be a success and instead divert those valuable resources to songs that can be popular.

## Proposed Solution 
The proposed solution is to make a model that can predict the success of a song. Success for this predictive model is defined as if the song in question can land any spot on Billboard's U.S Hot 100 Chart.

## Data - Datasets Origin
The data was scraped by another data scientist and comes from Billboard’s The Hot 100 chart:  
* The Hot 100 Chart covers the top songs in the United States every week all the way since the Chart’s inception in 1958. 
  * [https://data.world/kcmillersean/billboard-hot-100-1958-2017](https://data.world/kcmillersean/billboard-hot-100-1958-2017)
* I also used a kaggle dataset to splice in random tracks so there is a balance between the number of tracks that made it to the top and those that didn’t.
  * [https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks) [Removed as of September 2, 2021]

## Data
The structure of the cleaned data was a list of 48 thousand songs where each entry is a song with some of its properties. This is a list of all the properties:
![The Track Properties](images/TrackProperties.png)
An example of an entry of the dataset is: 
![Sample Track](images/TrackSampleElvis.png)

An additional column is added where the value is one hot encoded if the song reached the top 100 chart. 
This plot shows the structure of the cleaned dataset.
![Balanced_DF](plots/balanced_df.png)

## Methodology
### The Models
We'll start with a `train_test_split` to get our Training and Test Data.

Our baseline will be a Logistic Regression Model where the features are the track properties columns and the target is the one hot encoded column that denotes if the track is a member of the Hot 100 Chart.

We will then use a Decision Tree, followed by a Random Forest. Hyper Parameter tuning will be tried for every applicable model.

#### Logistic Regression

#### Decision Trees

#### Random Forest

## Results

## Results
The results of the each of the models are shown in the table below:


|  Original Alpha | Corrected Alpha for Bonferroni method | 
| :---: | :---: |
| 0.05 | 0.004545454545454546 |

| Gender vs Phobia | Original p-val | Bonferroni Corrected p-val| Reject Null? |
| :---: | :---: | :---: | :---: |
| Flying             | 7.208e-04  | 7.93e-03 | True |
| Thunder, Lightning | 4.6241𝑒−22 | 5.09e-21 | True |
| Darkness           | 9.9074𝑒−23 | 1.09e-21 | True |
| Heights            | 0.0989     | 1.0      | False|


## Conclusions / Future-Steps
Out of 10 different phobias surveyed, 9 of them show us that there is a relationship between a person’s gender and their phobias. 
<!-- 
Future Steps - Would be to go back to the beginning and impute those 2% of missing values I dropped and see if there is a significant difference. 

Post-Hoc Testing- Pair-wise comparisons. 
To see which gender is more afraid of those phobias and where the relationship is between the levels of the variables. -->
