import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt

from utils.preprocess import load_data
from utils.st_exploratory import run_eda  

# '''
# root _____________
#         |----------app.py
#         |__________utils
#             |------preprocess.py
#             |------exploratory.py
#             |------st.exploratory.py
#             |------model_training.py
#             |------st.model_training.py


# '''

def main():
    
    #set page configuration
    st.set_page_config(
        page_title = "climate trend predictor",
        layout = "wide"    
    )

    #App title and description
    st.title("Climate Trend Analysis and Prediction")

    st.markdown("Analyse historical tempreature and predict trend")


    #side bar

    st.sidebar.title("Navigation Page")
    page = st.sidebar.radio("Go to", ["EDA", "Model Training", "Prediction"])

    df = load_data()

    if page == "EDA":
        run_eda(df)   # call EDA part

    elif page == "Model Training":
        pass   #call algo part for model training

    else:
        pass #call prediction




if __name__ == "__main__":
    main()