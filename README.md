# Is Your Song a Banger or a Dud?
Galvanize Capstone Project: Is Your Song a Banger or a Dud?

This project implements several machine learning algorithms such as Logistic Regression, Decision Trees, and Random Forests. The algorithms use a song's features such as tempo, valence, liveliness etc., to predict if the song will be able to reach the Billboard Hot 100 Chart. 

## Objective
The objecive of this project is to make a model that can predict if a song can make it to the Billboard Hot 100 Chart. The model we end up with is a Hyperparameter Tuned Random Forest Model that uses a song's features such as tempo, valence, liveliness etc., to make it's prediction. 

## Understanding the Motivation / Potential Use-Cases TODO
The primary motivation of this project is money. How can 

## Proposed Solution 


## Data - Datasets
The data was scraped by another data scientist and comes from Billboard’s The Hot 100 chart:  
* The Hot 100 Chart covers the top songs in the United States every week all the way since the Chart’s inception in 1958.
* I also used a the kaggle dataset []() to splice in random tracks so there is a balance between the number of tracks that made it to the top and those that didn’t.


## Data - Entry
Since the amount of null values in some columns were at most only 2%, I ended up dropping those entries. It was a long survey so any presence of null values I figured was due to human-error such as skipping a question or data-loss from transferring the electronic or written records.

I also made a subset of the dataframe built from the survey so I can extract certain feature columns more easily and conduct seperate hypothesis tests more efficiently. 

## Methodology
### The Models
We will tackle our objective by doing a hypothesis test between Gender and Phobias.

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
