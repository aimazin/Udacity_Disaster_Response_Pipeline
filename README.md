# Udacity_Disaster_Response_Pipeline
Second Project of Udacity's Data Scientist Nanodegree


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Table of Contents

app.tar.gz: - run.py
            - master.html in templates folder
            - go.html in templates folder

data.tar.gz: - process_data.py
             - disaster_messages.csv
             - disaster_categories.csv
             - NLP.db

models.tar.gz: - train_classifier.py
               - modelxgboost.pkl
               
               
## Installation

install latest version of python version 3.8 and a Programming developer IDE

also pip install xgboost in IDE for the model


## Project Motivation

Second Project of Udacity's Data Scientist Nanodegree
Discerning crisis response data from Figure Eight's labeled data, then inserting results into app


## File Descriptions

#### app
run.py - used to run web application
masster.html - helps launch app on
go.html -helps launch app

#### data
process_data.py - used to perform ETL on data
disaster_messages.csv - structured messsage data from Figure Eight
disaster_categories.csv - structured category/label data from Figure Eight
NLP.db - created database pos ETL

#### models
train_classifier.py - used to perform NLP on created database and produce model
modelxgboost.pkl - produced model used in web application


## Issues

host website may be busy, you may need a udacity space_id to access hosting site

too little clean data came through in NLP.db only a little more than 40% of the data was useable


## Results

A functioning web app, but poor model resulted in subpar classifications even with tuning (reason: lack of data)


## Acknowledgements

Udacity Team and the Figure Eight Team for providing the dataset


## Licence

I authorize any none malicious use of this code.
