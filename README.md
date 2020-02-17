# Disaster-Response-Pipeline
Disaster Response Pipeline project for data science nano-degree of Udacity


### Project Description:
The main goal of this project is to facilitate processing the messages that are sent during disaster events by people who need help, and classifying those messages to their appropriate categories in order to link them with the relevant relief agenencies.


### Files Dedcription:

The project includes three folders wit the following files:
#### 1. data:
- disaster_categories.csv: a csv file includes the data about the 36 categories 
- disaster_messages.csv: a csv file includes the raw messages that were sent during previous disasters
- cleaned_dataset.db: SQLit database contains the processed data about messages and cateories 
- process_data.py: a Python code of ETL pipeline to clean the raw data and saved the processed data in a SQLite database 

#### 2. models:
- train_classifier.py: a Python code of machine learning pipeline to train a classifier usin grid search
- model.pkl: the trained classifier 

#### 3. app:
- run.py: a Flask web file to run the application
- templates: HTML files of the web application


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

