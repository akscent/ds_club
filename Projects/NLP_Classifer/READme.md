
=========

# Description of task

Practical assignment related to the course [Natural Language Processing course (stream 5, autumn 2023)](https://ods.ai/tracks/nlp-course-autumn-23). 

**Practical Assignment 2**: Text multiclass classification: store's review rating.


> This task is to classify the store's review rating into 5 classes. The metric is *`F1-score`*.

Task organizers presented 4 baseline solutions based on logistic regression, catboost, LSTM and Transformers. As part of the club meetings at school 21, we analyzed the baselines and wrote our own notebook based on the predictions of transformers and recurrent networks, coupled with generative features.

# Our solution

We created [Dataset on kaggle](https://www.kaggle.com/datasets/akscent/ods-huawei) and make analysis in kaggle-notebooks. 

We used the following approach to solve the problem:
1) Minimal cleaning of the dataset (because models trained on the same dirty data were used to generate features and predictions).
  - in particular, such sentences were cleaned up, for example `alovtdytvldvm++===amalot`
  - using a pre-trained summer of Russian texts, we reduced the few texts with a word length > 150
- we removed zero rows from the training dataset
2) Next, we trained the LSTM and Transformers models with the best hyperparameters. We received predictions for 5 classes as proba estimates.
3) We generated features using ready-made analysis tools for NLP tasks: pre-trained models, TF-IDF analysis, nlp_profiler.
4) We trained an ensemble of boosting models with optimal hyperparameters and feature selection.

As a result, we received a f-speed metric of 0.7

# Notes

We can say that we have achieved good accuracy in recognizing comments corresponding to 5 stars. Also good for 1 star, absolutely terrible for 2 stars and below average for 3 and 4 stars.

We can say that, if there was more time, it would be possible to achieve a more accurate identification of problem classes using a more detailed consideration of the dataset and an analytical approach to feature generation.