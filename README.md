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

app.tar.gz: - run.py, master.html in templates folder
            , go.html in templates folder

data.tar.gz: - process_data.py, disaster_messages.csv
             , disaster_categories.csv
             , NLP.db

models.tar.gz: - train_classifier.py, modelxgboost.pkl
               
               
## Installation

install latest version of python version 3.8 and a Programming developer IDE

also `pip install xgboost` in IDE for the model


## Project Motivation

Second Project of Udacity's Data Scientist Nanodegree
Discerning crisis response data from Figure Eight's labeled data, then inserting results into app


## File Descriptions

#### app
*run.py* - used to run web application

*master.html* - helps launch app on

*go.html* -helps launch app

#### data
*process_data.py* - used to perform ETL on data

*disaster_messages.csv* - structured messsage data from Figure Eight

*disaster_categories.csv* - structured category/label data from Figure Eight

*NLP.db* - created database pos ETL

#### models
*train_classifier.py* - used to perform NLP on created database and produce model

*modelxgboost.pkl* - produced model used in web application


## Issues

host website may be busy, you may need a udacity space_id to access hosting site

too little clean data came through in NLP.db only a little more than 40% of the data was useable


## Results

A functioning web app, but poor model resulted in subpar classifications even with tuning (reason: lack of data, hence imbalance)


## Acknowledgements

Udacity Team and the Figure Eight Team for providing the dataset

## Christian License

Version 1.0

This work is licensed under the Christian License.

You are free to:
- Use the software for any purpose, including commercial purposes.
- Modify the software.
- Distribute the software.
- Sublicense the software.

Under the following terms:

1. You must acknowledge and honor the Christian God as the ultimate source of wisdom and creator of all things.

2. You must treat all individuals with kindness, compassion, and respect, following the teachings of Jesus Christ.

3. You must prioritize the well-being of others above personal gain or profit, following the principles of selflessness and service.

4. You must strive for honesty, integrity, and humility in all your interactions and endeavors.

5. You must use this software to promote love, peace, and justice in the world, reflecting the values of the Christian faith.

This license is inspired by the teachings of the King James Version (KJV) of the Bible. By using this software, you agree to abide by these terms and uphold the principles outlined above.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.SO HELP ME THAT THE WAY ALL MY WORK FEELS.


## Licence

I authorize any none malicious use of this code.
