#################<br>
RECIPE CLASSIFIER<br>
#################

Author        : Andrew Yew<br>
Date initiated: 2023-04-11<br>
Description   : This is a repository for capstone project: recipe classifier.

**Data is available at**: <br>
[Google Drive Link](https://drive.google.com/drive/folders/11XaB4Jp29zT76OUpaiiNFoq8xqQx9EWS?usp=sharing)<br>
If denied access, please contact Andrew Yew at andrewyewcy@gmail.com

**Brief Introduction:**<br>
Given supply disruptions due to recent global events such as the Covid 19 pandemic and the war in Ukraine, Canadian food prices have risen at an alarming rate of 11% year on year in 2022, putting pressure on the average consumer’s budget. This combined with the benefits of home cooking has undoubtedly led to many busy working Canadians to cook more at home. However, assuming a busy work life, many adults need a way to ensure whatever they choose to cook is worth the precious time and effort after work. The recipe classifier seeks to address this issue for busy working adults by classifying recipes as worth the time and effort or not worth it, given the different elements available in online food recipes. Natural Language Processing (NLP) and other techniques were applied to 33,691 food recipes gathered from allrecipes.com to train a logistic regression model, achieving a final accuracy of 76% in determining if a recipe is worth it.

**Below is the table of contents for this repository:**

1) requirements.txt
- a text file that can be used to recreate the python environment used for this project
- bash code to run the file available within the file (open in plain text editor)


2) Andrew_Yew_capstone_final_report.pdf
- Contains a 3 page written report describing the entire project at a high non-technical level


3) Andrew_Yew_final_presentation.pdf
- Contains a PDF version of the final presentation which is a high level summary of the project to be presented in approximately 5 minutes.


4) 010_data_acquisition.ipynb
- This notebook contains all code and details required to scrape recipe data from allrecipes.com


5) 020_eda_feature_engineering.ipynb
- This notebook details all exploratory data analysis, data processing, and feature engineering before the modelling stage.


6) 030_modelling.ipynb
- This notebook records all experimental procedure and results regarding modelling decisions and model training. Also includes final preprocessing steps such as Term Frequency-Inverse Document Frequency(TF-IDF) vectorization, KNN imputer, One Hot Encoding (OHE), and train test split.


7) images folder
- images generated from the notebooks

8) model folder
- final model stored as a pickle file
