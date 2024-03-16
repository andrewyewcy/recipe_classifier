### IMPORT REQUIRED PACKAGES ###
import numpy as np                                  # data processing
import pandas as pd                                 # data loading and processing
import streamlit as st                              # for creating minimum viable app on streamlit
import joblib                                       # for importing pickled model 
from sklearn.preprocessing import RobustScaler      # final scaler
from sklearn.linear_model import LogisticRegression # final model


### INITIAL SETTINGS OF STREAMLIT APP ###
st.set_page_config(
    layout = "wide", 
    page_title = "Recipe Classifier"
)

# Create columns that split the Streamlit App
col1, col2 = st.columns(2)


### LEFT CONTAINER ###
with col1.container():

    # Create two sub-columns within container
    subcol1, subcol2 = st.columns(2)

    # Set visible title
    subcol1.header('Recipe Classifier')
    subcol1.write('by **Andrew Yew** <br> [LinkedIn](https://www.linkedin.com/in/andrewyewcy/),  [Website](https://andrewyewcy.com/)',
                 unsafe_allow_html = True
                 )

    # Set stock icon
    # <a href="https://www.flaticon.com/free-icons/recipe" title="recipe icons">Recipe icons created by photo3idea_studio - Flaticon</a>
    subcol2.image(
        '22_assets/images/recipe.png',
        width = 135
    )

col1.write('This projects seeks to help busy working adults identify which recipes are **worth their time and effort** given Canadian food prices have risen at an alarming rate of [11% year on year in 2022](https://www150.statcan.gc.ca/n1/pub/62f0014m/62f0014m2022014-eng.htm) and continue to do so.')

col1.write('***')

col1.write(
    """A recipe is classified as `worth it` if it has a rating of 4.5 or higher and more than 1 ratings. To train the model, 40,000 recipes were web-scraped from [allrecipes.com](https://www.allrecipes.com/). 

KNN-imputer and natural language processing (NLP) methods such as TF-IDF were used to preprocess the data before training with supervised learning models such as logistic regression, decision trees, and ensemble models such as XGBoost. 

Ultimately, logistic regression was chosen as final model due to it using a more balanced combination of features compared to other models."""
)

col1.write("For more details:")
col1.write("[Summary presentation](https://github.com/andrewyewcy/recipe_classifier/blob/main/presentation.pdf)")
col1.write("[GitHub repository for project](https://github.com/andrewyewcy/recipe_classifier)")


### MODEL ###
# Load final model and scaler
model  = joblib.load('13_models/final_model.pkl')
scaler = joblib.load('13_models/final_robust_scaler.pkl')

# Load data
X_t          = pd.read_csv('14_streamlit_data/X_t_for_streamlit.csv')
processed_df = pd.read_csv('14_streamlit_data/processed_data_for_streamlit.csv', index_col = False)

col2.subheader(f"Model Demo")

# Define select box for users to select recipes
recipe_title = col2.selectbox(
    "Select Recipe:", 
    options = [
        "Select a recipe",
        "Authentic Louisiana Red Beans and Rice",
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
        ]
)

if recipe_title == 'Select a recipe':
    col2.write('Please select a recipe from above to run the model')
else:
    # Define a dataframe for display
    display_df = processed_df[processed_df['recipe_title'] == recipe_title]
    recipe_index = display_df.index[0]
    
    ###Model Elements###
    X_tl  = np.log(X_t+1)
    X_tlr = scaler.transform(X_tl)
    X_tlr = pd.DataFrame(X_tlr, columns = X_t.columns)
    
    y_pred      = model.predict(X_tlr.loc[[recipe_index],:])
    y_pred_prob = model.predict_proba(X_tlr.loc[[recipe_index],:])

    # Define actual label to print
    if display_df.iloc[0,1] >= 4.5:
        actual_label = 'Worth it'
    else:
        actual_label = 'Not worth it'

    # Define if a recipe is true positive or negative
    if y_pred == 1 and actual_label == 'Worth it':
        pred_type = 'True Positive'
    elif y_pred == 1 and actual_label != 'Worth it':
        pred_type = 'False Positive'
    elif y_pred != 1 and actual_label == 'Worth it':
        pred_type = 'False Negative'
    else:
        pred_type = 'True Negative'
        
    if y_pred == 1:
        y_pred_display = "Worth It" 
        col2.write(f"""
            **Model Prediction**: {y_pred_display} \N{slightly smiling face}, {np.round(y_pred_prob[0,1]*100,1)} % conf. 
            
            **Actual Label**    : {actual_label}, {pred_type}
            """)
        
    else:
        y_pred_display = "Not Worth It"
        col2.write(f"""
            **Model Prediction**: {y_pred_display} \U0001FAE0, {np.round(y_pred_prob[0,1]*100,1)} % conf. 
            
            **Actual Label**    : {actual_label}, {pred_type}
            """)

    col2.subheader(f"Data: {display_df.iloc[0,0]}")
    col2.write(f"**Average rating:** {str(display_df.iloc[0,1])}, **# of Ratings**: {str(int(display_df.iloc[0,2]))}, **Total Time**: {str(int(display_df.iloc[0,-6]))} min.")
    col2.write(display_df.iloc[0,3])
    
    with col2.container():
        col3,col4 = st.columns(2)
        col3.write('Ingredients:')
        col3.write(display_df.iloc[0,11])
    
        col4.write('Nutritional Information:')
        col4.write(display_df.iloc[0,13:21])
    
    col2.text('Cooking Directions')
    col2.write(display_df.iloc[0,9])