# RECIPE CLASSIFIER

Author: Andrew Yew<br>
Contact: [LinkedIn](https://www.linkedin.com/in/andrewyewcy/) [Website](https://andrewyewcy.com/)

## Brief Introduction
Given supply disruptions due to recent global events such as the Covid 19 pandemic and the war in Ukraine, Canadian food prices have risen at an alarming rate of 11% year on year in 2022, putting pressure on the average consumer’s budget. This combined with the benefits of home cooking has undoubtedly led to many busy working Canadians to cook more at home. However, assuming a busy work life, many adults need a way to ensure whatever they choose to cook is worth the precious time and effort after work. The recipe classifier seeks to address this issue for busy working adults by classifying recipes as worth the time and effort or not worth it, given the different elements available in online food recipes. Natural Language Processing (NLP) and other techniques were applied to 40,000 food recipes gathered from allrecipes.com to train a logistic regression model, achieving a final accuracy of 76% in determining if a recipe is worth it.

## Access to Web-app:
Streamlit Web-app: [Link](https://andrewyewcy-recipe-classifier-recipe-classifier-ojen7a.streamlit.app)<br>
The web-app allows users to select recipes and generates a real time prediction from the model.
![recipe_classifier_webapp.png](/22_assets/images/recipe_classifier_webapp.png)

## Presentation and Report
A visual [presentation deck](https://github.com/andrewyewcy/recipe_classifier/blob/main/22_assets/presentations/presentation.pdf) and a short [business report](https://github.com/andrewyewcy/recipe_classifier/blob/main/22_assets/presentations/summary_report.pdf) are available in the respective links.

## Environments and Setup
The classifier is currently packaged as a Streamlit application and hosted on Streamlit cloud. A transition is being made to port all past code from running on Anaconda environments to packaged Docker containers. That being said, Docker and Docker-compose installed machines are required to ensure smooth setup and replication of repo environments.

**Setup instructions**<br>
1. Installed GIT, Docker and Docker-compose on local machine. 
2. Clone the git repository
3. In terminal, go to directory of repository, then run:
   - `app_setup.yaml` for setup of production environment (Streamlit app)
   - `dev_setup.yaml` for setup of development environment, which includes Streamlit app, Jupyter Hub, phpMyAdmin, and MySQL server

- For the app in production on Streamlit cloud: 
    - `requirements.txt` provides all the required packages.
    - `recipe_classifier.py` Python script containing production app
- For notebooks: `dev_setup.yaml` is a docker-compose file that reads requirements from the folder `01_dev_requirements` to replicate the Docker container that runs all notebooks.
    - After forking the repository, in terminal within the repo's directory run the following
    - `docker-compose -f dev_setup.yaml up`
    - Instructions on how to access Jupyter will appear in the terminal
        - Container 1 `sel`: contains Selenium with Google Chrome installed for web-scraping
        - Container 2 `dev`: contains Pyspark notebook with most machine learning packages installed 

## Tools involved
- Docker
- Python
- Streamlit
- Flask

## Repository Table of Contents
- `01_dev_requirements`
Contains Dockerfile to recreate Docker containers for running all notebooks.
Required by `dev_setup.yaml` when running docker-compose.

- `02_app_requirements` *in construction*
Contains Dockerfile to recreate Docker container for application in production
Required by `app_setup.yaml` when running docker-compose

- `11_raw_data` *not available on GitHub*
The raw data web-scraped from Allrecipes.com between 2023 FEB and 2023 MAR
For access, please contact the owner of this repo

- `12_processed_data` *in construction*
Contains data that has been processed for analysis and machine learning use

- `13_models`
Contains the trained machine learning models used throughout development and production

- `14_streamlit_data` *to be merged into processed data*
Currently contains the processed data used by the model in production on Streamlit cloud

- `21_notebooks`
Contains all Jupyter notebooks used for data processing and model training
Below are core notebooks of the project:
    - `010_data_acquisition.ipynb`: Web-scraping process to gather list of recipe urls and data from each url.
    - `011_recipe_labels_upload_data_to_aws_s3.ipynb`: Process data from 010 and upload to AWS S3 bucket
    - `020_eda_feature_engineering.ipynb`: This notebook details all exploratory data analysis, data processing, and feature engineering before the modelling stage.
    - `021_recipe_labels_analysis_count_vectorizer`: Bivariate analysis of recipe label frequencies and ratings.
    - `030_modelling.ipynb`: This notebook records all experimental procedure and results regarding modelling decisions and model training. Also includes final preprocessing steps such as Term Frequency-Inverse Document Frequency(TF-IDF) vectorization, KNN imputer, One Hot Encoding (OHE), and train test split.

- `22_assets`
Contains all assets for the project. For example: images, presentations, reports

`23_logs` *in construction*
Contains all log files, such as those generated during model training

- `24_testing`
Contains all notebooks and code in development.

- `30_automation` *in construction*
Contains Python files and Makefile for automation processes
    - `get_data.py`: Given an allrecipe url, webscrape and convert data into DataFrame
    - `preprocessing.py`: Preprocess data from `get_data.py` for prediction
    - `prediction.py`: Feds output from `preprocessing.py` into final model for prediction