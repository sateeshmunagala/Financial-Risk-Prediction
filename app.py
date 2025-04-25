import os
from langchain.document_loaders import CSVLoader
import streamlit as st

import Task1_DataIngestion_Preprocessing as data_ingestion
import Task2_Vector_Indexing as vector_indexing
import Task3_RAG_Pipeline_Implementation as rag_pipeline

## The below code is for setting up the Streamlit app with User Interface.
## Due to time constraint, I am not able to implement the UI in a better way but still basic UI is implemented.
## JUST ENTER THE COMMAND -> streamlit run app.py in the terminal to run the app.

## set up Streamlit 
st.title("Financial-Risk-Prediction")
#st.write("Upload Pdf's and chat with their content")

## Input the Groq API Key
api_key=st.text_input("Enter your Huggingface API key:",type="password")

if api_key:
    os.environ["Huggingface_API_KEY"] = api_key
    st.success("API key set successfully!")
else:
    st.warning("Please enter your Huggingface API key to proceed.")
    

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

def my_function():
    #st.write("Button clicked!")
    st.session_state.button_clicked = True

vector_index = None
if st.button("Click to Create Embeddings", on_click=my_function, key="my_button"):
    if st.session_state.button_clicked:
        text_directory = 'data/text_data'
        text_corpus = data_ingestion.load_and_preprocess_text_data(text_directory)
        vector_index = vector_indexing.create_vector_index(text_corpus)
        st.success("Vector Index & Embedding are created successfully!")
    
## MAin interface for user input
st.write("Ask Question about the company. Please mention the company name in the question")
user_input=st.text_input("Query:")

csv_file = 'data/company.csv'
text_directory = 'data/text_data'

if user_input :
    company_df = data_ingestion.load_and_preprocess_csv(csv_file)
    company_df = company_df[company_df['company_id']== user_input]
    company_name = company_df['company_id']
    company_ratios = company_df[['debt_equity_normalized', 'current_ratio_normalized', 'ROE_normalized']].to_dict()
    
    if  vector_index is None:
        st.warning("Please create the vector index first.")
        st.stop()
        
    with st.spinner("Waiting...", show_time=True):
        risk_assessment = rag_pipeline.rag_pipeline(company_name, company_ratios, vector_index)
        st.write(f"\nRisk Assessment for {company_name}: {risk_assessment}")
    
else:
    st.warning("Please provide the company name to get the risk assessment.")