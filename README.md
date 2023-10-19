#################<br>
RECIPE CLASSIFIER<br>
#################<br>

Author        : Andrew Yew<br>
Date initiated: 2023-04-11<br>
[LinkedIn](https://www.linkedin.com/in/andrewyewcy/) [Website/Blog](https://andrewyewcy.com/)

**Brief Introduction:**<br>
Given supply disruptions due to recent global events such as the Covid 19 pandemic and the war in Ukraine, Canadian food prices have risen at an alarming rate of 11% year on year in 2022, putting pressure on the average consumer’s budget. This combined with the benefits of home cooking has undoubtedly led to many busy working Canadians to cook more at home. However, assuming a busy work life, many adults need a way to ensure whatever they choose to cook is worth the precious time and effort after work. The recipe classifier seeks to address this issue for busy working adults by classifying recipes as worth the time and effort or not worth it, given the different elements available in online food recipes. Natural Language Processing (NLP) and other techniques were applied to 33,691 food recipes gathered from allrecipes.com to train a logistic regression model, achieving a final accuracy of 76% in determining if a recipe is worth it.

**Access the Web-app here:**<br>
Streamlit Web-app: [Link](https://andrewyewcy-recipe-classifier-recipe-classifier-ojen7a.streamlit.app)
![030_regression_modelling_001.png](../assets/images/030_regression_modelling_001.png){:class="img-responsive"}

**Presentation and Report**<br>
A visual [presentation deck](https://github.com/andrewyewcy/recipe_classifier/blob/main/22_assets/presentations/presentation.pdf) and a short [business report](https://github.com/andrewyewcy/recipe_classifier/blob/main/22_assets/presentations/summary_report.pdf) are available in the respective links.

**Environment and Setup**<br>
The classifier is currently packaged as a Streamlit application and hosted on Streamlit cloud. A transition is being made to port all past code from running on Anaconda environments to packaged Docker containers. That being said, Docker and Docker-compose installed machines are required to ensure smooth setup and replication of repo environments.

- For the app in production on Streamlit cloud: 
    - `requirements.txt` provides all the required packages.
    - `recipe_classifier.py` Python script containing production app
- For notebooks: `dev_setup.yaml` is a docker-compose file that reads requirements from the folder `01_dev_requirements` to replicate the Docker container that runs all notebooks.
    - After forking the repository, in terminal within the repo's directory run the following
    - `docker-compose -f dev_setup.yaml up`
    - Instructions on how to access Jupyter will appear in the terminal

**Below is the table of contents for this repository:**<br>
1) `01_dev_requirements`
- contains Dockerfile to recreate Docker containers for running all notebooks
- required by `dev_setup.yaml` when running docker-compose

2) `02_app_requirements` *in construction*
- contains Dockerfile to recreate Docker container for application in production
- required by `app_setup.yaml` when running docker-compose

3) `11_raw_data` *not available on GitHub*
- the raw data web-scraped from Allrecipes.com between 2023 FEB and 2023 MAR
- for access, please contact the owner of this repo

4) `12_processed_data` *in construction*
- contains data that has been processed for analysis and machine learning use

5) `13_models`
- contains the trained machine learning models used throughout development and production

6) `14_streamlit_data` *to be merged into processed data*
- Currently contains the processed data used by the model in production on Streamlit cloud

7)  `21_notebooks`
- Contains all Jupyter notebooks used for data processing and model training
- Below are core notebooks of the project:
    - `010_data_acquisition.ipynb`
        - This notebook contains all code and details required to scrape recipe data from allrecipes.com

    - `020_eda_feature_engineering.ipynb`
        - This notebook details all exploratory data analysis, data processing, and feature engineering before the modelling stage.

    - `030_modelling.ipynb`
        - This notebook records all experimental procedure and results regarding modelling decisions and model training. Also includes final preprocessing steps such as Term Frequency-Inverse Document Frequency(TF-IDF) vectorization, KNN imputer, One Hot Encoding (OHE), and train test split.

8) `22_assets`
- Contains all assets for the project. For example: images, presentations, reports

9) `23_logs` *in construction*
- contains all log files, such as those generated during model training

9) `24_testing`
- Contains all notebooks and code in development.