import numpy as np
import pandas as pd
import streamlit as st

def main():
    st.write("Hello World")

    # Load data
    df = pd.read_csv("12_processed_data/recipes_pivot.csv")

    # Convert data to 
    pivot_df = df.pivot(
        columns = "recipe_name",
        index   = "ing_name",
        values = "cost"
    )

    
    st.data_editor(pivot_df)
    
    

if __name__ == '__main__':
    main()
