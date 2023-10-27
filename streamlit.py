import streamlit as st
from LitRevSentences.pipeline.prediction_pipeline import PredictionPipeline


st.write("""
### Literature Sentences Classification Model

#### Description
This model will categorize sentences within literature review, making it easier to extract crucial themes & insights.

***
""")

LitRev = st.text_area('Please write the Literature review in below box')


if st.button('Submit'):
    if LitRev == '':
        st.write('No Literature review added')
    else:
        prediction_pipeline = PredictionPipeline()
        result_df = prediction_pipeline.run_pipeline(LitRev=LitRev)    
        st.dataframe(result_df)

else:
    st.write('Press submit button')