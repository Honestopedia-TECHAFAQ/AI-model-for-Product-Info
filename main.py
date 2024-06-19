import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from transformers import pipeline
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('product_dataset.csv')
        return data
    except Exception as e:
        st.error("Error loading data: " + str(e))
        logging.error("Error loading data: " + str(e))
        return pd.DataFrame(columns=['Product_ID', 'Question', 'Answer'])

data = load_data()
@st.cache_resource
def load_nlp_model():
    try:
        qa_model = pipeline("question-answering")
        return qa_model
    except Exception as e:
        st.error("Error loading NLP model: " + str(e))
        logging.error("Error loading NLP model: " + str(e))
        return None

qa_model = load_nlp_model()
st.title("Custom AI Q&A Model")

product_id = st.selectbox("Select Product ID", data['Product_ID'].unique())

question = st.text_input("Enter your question:")

if question and qa_model is not None:
    product_data = data[data['Product_ID'] == product_id]
    
    if not product_data.empty:
        context = " ".join(product_data['Answer'].tolist())
        result = qa_model(question=question, context=context)
        
        answer = result.get('answer', None)
        score = result.get('score', 0)
        dynamic_threshold = 0.6  
        
        if score > dynamic_threshold:
            st.write(f"Answer: {answer}")
            logging.info(f"Question: {question} | Answer: {answer} | Product ID: {product_id}")
        else:
            st.write("Sorry, I couldn't find a matching answer. Please refine your question or contact us for further assistance.")
            logging.info(f"Question: {question} | Answer: Not found | Product ID: {product_id}")
            if st.button("Contact Us"):
                st.write("Please contact us at support@example.com")
    else:
        st.write("No data available for the selected product.")
else:
    if qa_model is None:
        st.write("NLP model is not available. Please try again later.")
    else:
        st.write("Please enter your question.")
st.write("Please enter your question related to the selected product. If an exact match is not found, you'll be prompted to contact us for further assistance.")
