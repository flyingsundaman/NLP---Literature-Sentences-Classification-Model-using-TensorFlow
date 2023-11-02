import streamlit as st
from LitRevSentences.pipeline.prediction_pipeline import PredictionPipeline

# Title
st.title("Literature Sentences Classification Model")

# Description
st.markdown("""
This model categorizes sentences within a literature review, making it easier to extract crucial themes and insights.
***
""")

# User Input
st.sidebar.header("User Input")
st.sidebar.markdown("""
Please write the Literature review in the box below and press "Submit".
""")
LitRev = st.sidebar.text_area('')

# Button
if st.sidebar.button('Submit'):
    if LitRev == '':
        st.error('No Literature review added. Please input a review in the sidebar.')
    else:
        prediction_pipeline = PredictionPipeline()
        result_df = prediction_pipeline.run_pipeline(LitRev=LitRev)  
        def dataframe_to_markdown_table(df):
            from tabulate import tabulate
            return tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
        st.markdown(dataframe_to_markdown_table(result_df), unsafe_allow_html=True)
        st.markdown("""
        #### Result Explanation
        The table above shows the categorized sentences from your literature review. Each row represents a sentence and its corresponding category. This classification can help you understand the key themes and insights in the literature review.
        """)
else:
    st.info('Please input a review and press "Submit"')

# Additional Information
st.subheader("Additional Information")
st.markdown("""
This tool uses a machine learning model trained on a large corpus of literature reviews. It analyzes the text you input and categorizes each sentence based on its content.

If you have any questions or feedback, please don't hesitate to reach out.
            
[LINKEDIN](https://www.linkedin.com/in/naqibasri/)
""")

# About
st.subheader("About")
st.markdown("""
This tool is part of a larger project aimed at making literature reviews more accessible and easier to analyze. It was developed by [naqibasri](https://www.linkedin.com/in/naqibasri/) and is powered by the latest advancements in natural language processing.
""")
st.subheader("Credit")
st.markdown("""
1. Where our data is coming from: [*PubMed 200k RCT: a Dataset for Sequential Sentence Classification in Medical Abstracts*](https://arxiv.org/abs/1710.06071)
2. Where our model is coming from: [*Neural networks for joint sentence classification in medical paper abstracts*](https://arxiv.org/pdf/1612.05251.pdf).
            """)