
# import Task1 python file
import Task1_DataIngestion_Preprocessing as data_ingestion
# import Task2 python file
import Task2_Vector_Indexing as vector_indexing
# import Task3 python file
import Task3_RAG_Pipeline_Implementation as rag_pipeline

from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from huggingface_hub import login

def rag_pipeline(company_name, company_ratios, vector_index, k=3):
    """
    Implements the Retrieval-Augmented Generation (RAG) pipeline for risk prediction.
    """
    print('company_ratios')
    print(company_ratios)
    
    # 1. Retrieval step
    relevant_docs = vector_index.similarity_search(f"company: {company_name}", k=k)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # 2. Augmentation step (we will do it in the prompt)

    # 3. Generation step (using Hugging Face Hub LLM)
    
    login(token="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")
    
    llm = HuggingFaceHub(repo_id="microsoft/Phi-3-mini-4k-instruct",
                         huggingfacehub_api_token="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") # Replace with your token and model

    prompt_template = """You are a financial risk analyst. Based on the following financial ratios:\n
    Debt to Equity: {debt_to_equity} \n
    Current Ratio: {current_ratio} \n
    ROE: {roe} \n
    And the following relevant information:
    {context}
    \nAssess the financial risk level of the company as either "Low", "Medium", or "High" and provide a brief justification for your prediction.
    \nRisk Level: """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["debt_to_equity", "current_ratio", "roe", "context"])
    rag_chain = prompt | llm

    prediction_and_justification = rag_chain.invoke({
        "debt_to_equity": company_ratios['debt_equity_normalized'],
        "current_ratio": company_ratios['current_ratio_normalized'],
        "roe": company_ratios['ROE_normalized'],
        "context": context
    })

    return prediction_and_justification

# --- Example Usage Below---
csv_file = 'data/company.csv'
text_directory = 'data/text_data'

if __name__ == "__main__":
    company_df = data_ingestion.load_and_preprocess_csv(csv_file)
    text_corpus = data_ingestion.load_and_preprocess_text_data(text_directory)
    vector_index = vector_indexing.create_vector_index(text_corpus)

    sample_company = company_df.iloc[0]
    company_name = sample_company['company_id']
    company_ratios = sample_company[['debt_equity_normalized', 'current_ratio_normalized', 'ROE_normalized']].to_dict()

    risk_assessment = rag_pipeline(company_name, company_ratios, vector_index)
    print(f"\nRisk Assessment for {company_name}:\n {risk_assessment}")
