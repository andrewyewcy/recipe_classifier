### IMPORT REQUIRED PACKAGES ###
# Packages for processing data
import numpy as np
import pandas as pd

# Packages for plotting
import streamlit as st
import plotly.express as px

# Packages for pickling and loading pickle files.
import joblib 

# Packages for data preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler

# Packages for models
from sklearn.linear_model import LogisticRegression


### INITIAL SETTING OF STREAMLIT APP ###
st.set_page_config(layout="wide", page_title="Recipe Classifier")

#st.title('Recipe Classifier')
col1, col2  = st.columns(2)

with col1.container():
    subcol1, subcol2 = st.columns(2)
    subcol1.subheader('Recipe Classifier by Andrew Yew')
    subcol2.image('images/allrecipes.png')

### BAR CHART FOR TAGS ###
# Load tags data
tags_df = pd.read_csv('streamlit_data/tags_df.csv')

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

### MODEL ###
# Load models and scalers
model = joblib.load('model/final_model.pkl')
scaler = joblib.load('model/final_robust_scaler.pkl')

# Load data
X_t = pd.read_csv('streamlit_data/X_t_for_streamlit.csv')
processed_df = pd.read_csv('streamlit_data/processed_data_for_streamlit.csv')
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