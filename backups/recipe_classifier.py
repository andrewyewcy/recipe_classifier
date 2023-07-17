### IMPORT REQUIRED PACKAGES ###
# Packages for processing data
import numpy as np
import pandas as pd

# Packages for plotting
import streamlit as st
import plotly.express as px

# Packages for pickling and loading pickle files.
import joblib 
#import time

# # Packages for statistical testing
# from scipy import stats # chi-square test

# # Packages for data preprocessing
# from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
# from sklearn.decomposition import PCA

# # Packages for natural language processing
# import nltk # Natural Language Tool Kit
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# import string
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Packages for models
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier

# Packages for model hyperparameter optimization
# from sklearn.model_selection import train_test_split, GridSearchCV
# #from sklearn.model_selection import cross_val_score
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline # To set up a temporary directory for caching pipeline results
# from tempfile import mkdtemp

# Packages for model evaluation
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# from sklearn.metrics import classification_report
# import shap

### INITIAL SETTING OF STREAMLIT APP #######################################################################################
st.set_page_config(layout="wide", page_title="Recipe Classifier")

#st.title('Recipe Classifier')
col1, col2  = st.columns(2)

with col1.container():
    subcol1, subcol2 = st.columns(2)
    subcol1.subheader('Recipe Classifier by Andrew Yew')
    subcol2.image('images/allrecipes.png')

### BAR CHART FOR TAGS ####################################################################################################
# Load tags data
tags_df = pd.read_pickle('streamlit_data/tags_df.pkl')

# Create DataFrame for plot
plot_df = tags_df.head(5)

# Create plotly express bar chart
fig = px.bar(plot_df, 
             x = 'tags', 
             y = ['not_worth_it', 'worth_it'],
             labels = {'tags' : 'Recipe Tags',
                       'variable' : 'Target Feature',
                       'value' : 'Recipe Count'},
             hover_name = "tags",
             hover_data = {'variable':False,
                           'tags' : False,
                           'value' : False,
                           'total recipes '    : (':f', plot_df['total_count'].tolist()),
                           '# worth it '     : (':f', plot_df['worth_it'].tolist()),
                           '# not worth it ' : (':f', plot_df['not_worth_it'].tolist())
                          })

# Format plotly express bar chart
fig.update_layout(
    legend=dict(
        x=0.6,
        y=.95,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    )
)

# Display created plotly express bar chart within col1 
col1.plotly_chart(fig, 
                  use_container_width = True, 
                  sharing = "streamlit",
                  theme = None)

### Create scatter plot of data ###########################################################################################
# Excluded for now because processed_df is 399MB in size, larger than the github file size limit of 100MB
# temp_df = pd.read_pickle('data/processed_df.pkl')
# temp_df = temp_df[['recipe_title', 
#          'average_rating', 
#          'recipe_worth_it',
#          'number_of_ratings',
#          'number_of_reviews']].copy()
# temp_df['log_number_of_ratings'] = np.log(temp_df['number_of_ratings']+1)
# temp_df['log_number_of_reviews'] = np.log(temp_df['number_of_reviews']+1)
# temp_df['recipe_worth_it'] = ["worth it" if recipe == 1 else "not worth it" for recipe in temp_df['recipe_worth_it'].tolist()]
# fig = px.scatter(temp_df,
#                  x = "log_number_of_ratings",
#                  y = "log_number_of_reviews",
#                  color = "recipe_worth_it",
#                  labels = {'recipe_worth_it': 'Target Feature',
#                           'log_number_of_reviews':'LOG Number of Reviews',
#                           'log_number_of_ratings':'LOG Number of Ratings'
#                          },
#                  opacity = 0.5,
#                  hover_name = "recipe_title",
#                  hover_data = {"recipe_worth_it" : False,
#                                "log_number_of_reviews" : False,
#                                "log_number_of_ratings" : False,
#                                "average_rating":True,
#                                "number_of_ratings":True,
#                                "number_of_reviews":True},
#                  )

# fig.update_layout(
#     legend=dict(
#         x=0.05,
#         y=.9,
#         traceorder="normal",
#         font=dict(
#             family="sans-serif",
#             size=12,
#             color="black"
#         ),
#     )
# )

# col1.plotly_chart(fig, 
#                   use_container_width = True, 
#                   sharing = "streamlit",
#                   theme = None)

### MODEL #################################################################################################################
# Load models and scalers
model = pd.read_pickle('model/final_model.pkl')
scaler = pd.read_pickle('model/final_robust_scaler.pkl')

# Load data
X_t = pd.read_pickle('streamlit_data/X_t_for_streamlit.pkl')
processed_df = pd.read_pickle('streamlit_data/processed_data_for_streamlit.pkl')
processed_df.reset_index(drop = True, inplace = True)


col2.header(f"Model Prediction")

recipe_title = col2.selectbox("Select Recipe:", 
                              options = ["Authentic Louisiana Red Beans and Rice",
                                         "Banana Banana Bread", # TP Strong
                                        "To Die For Blueberry Muffins", # TP Strong
                                        "Easy Meatloaf", # TP Strong
                                        "Too Much Chocolate Cake", # TP Strong
                                        "Clone of a Cinnabon", # TP Strong
                                        "Classic Chicken Cordon Bleu", #TP Strong
                                        "Sweet Breakfast Hash with Apple and Rosemary", # TP Weak
                                        "Tilapia All-in-One Casserole",# TP weak
                                        "Easy Cabbage Rolls", #TN
                                        "Pannukakku (Finnish Oven Pancake)", #TN,
                                        "Whole Wheat Beer Bread", #FN
                                        "Vegetarian Sloppy Joe", #FN
                                        "Simple White Cake", #FP
                                        "Salisbury Steak" #FP
                                        ])

display_df = processed_df[processed_df['recipe_title'] == recipe_title]
recipe_index = display_df.index[0]

#####Model Elements######
X_tl = np.log(X_t+1)
X_tlr = scaler.transform(X_tl)
X_tlr = pd.DataFrame(X_tlr,columns = X_t.columns)

y_pred = model.predict(X_tlr.loc[[recipe_index],:])
y_pred_prob = model.predict_proba(X_tlr.loc[[recipe_index],:])

if y_pred == 1:
    y_pred_display = "Worth It" 
    col2.header(f"Model Prediction: {y_pred_display} \N{slightly smiling face}")
    col2.header(f"Prediction Confidence: {np.round(y_pred_prob[0,1]*100,1)} %")
else:
    y_pred_display = "Not Worth It"
    col2.header(f"Model Prediction: {y_pred_display} \U0001FAE0")
    col2.header(f"Prediction Confidence: {np.round(y_pred_prob[0,0]*100,1)} %")

#np.round(y_pred_prob*100,2)
#####Display Elements#####


col2.write("---")
col2.subheader(display_df.iloc[0,0])
col2.write(f"**Average rating:** {str(display_df.iloc[0,1])}, **# of Ratings**: {str(int(display_df.iloc[0,2]))}, **Total Time**: {str(int(display_df.iloc[0,-6]))} min.")
col2.write(display_df.iloc[0,3])

#subcol1, subcol2, subcol3, subcol4 = st.columns(4)

with col2.container():
    col3,col4 = st.columns(2)
    col3.text('Ingredients:')
    col3.write(display_df.iloc[0,11])

    col4.text('Nutritional Information:')
    col4.write(display_df.iloc[0,13:21])

col2.text('Cooking Directions')
col2.write(display_df.iloc[0,9])