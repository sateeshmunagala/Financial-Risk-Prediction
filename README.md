# Financial-Risk-Prediction
Design and implement a Retrieval-Augmented Generation (RAG) pipeline using open‑source models to predict financial risk levels for companies based on both structured data (financial ratios) and unstructured text (e.g. earnings call transcripts, news)

## 1. Problem Statement Overview

Design and implement a Retrieval-Augmented Generation (RAG) pipeline using open‑source models to predict financial risk levels for companies based on both structured data (financial ratios) and unstructured text (e.g. earnings call transcripts, news). You will build the pipeline end‑to‑end, evaluate its performance, and document your solution.

## 2. Objectives
Retrieval: Ingest and index heterogeneous financial data (CSV tables + text documents).
Generation: Leverage an open‑source LLM (e.g. Llama‑2, Mistral, or similar) to produce risk assessments.
Prediction: Classify companies into risk categories (e.g. Low, Medium, High) and justify predictions with retrieved evidence.
Evaluation: Measure and report predictive performance using appropriate metrics.

## 3. Data & Environment
Structured Data:
A CSV of ~1,000 companies with historical financial ratios (e.g company_name, debt‑equity, current ratio, ROE and the target column is : risk_level with 3 possible values: high, low , medium).
Unstructured Data:
A folder of text files containing quarterly earnings call transcripts and relevant news articles.

Tech Stack:
Python 3.10+
FAISS (or alternative) for vector retrieval
Hugging Face Transformers (open‑source LLM)
LangChain (or your own orchestration code)
Jupyter Notebook or Python scripts

## 4. Task Requirements
Data Ingestion & Preprocessing
Load and clean the CSV; normalize numerical features.
Preprocess text (tokenization, basic cleaning).
Vector Indexing
Encode text documents into embeddings (e.g. with Sentence‑Transformers).
Store them in a FAISS (or equivalent) index.
RAG Pipeline Implementation
Write a function that:
Retrieves the top‑k relevant text passages for a given company query.
Augments the financial ratios with retrieved context.
Generates a risk prediction and justification via the LLM.
Prediction Module
Based on LLM output, map its qualitative assessment to one of three risk levels.
Alternatively, train a simple classifier (e.g. logistic regression) on ratio features and combine with LLM signals.
Evaluation
Split your structured data into train/test (e.g. 80/20).
Report classification metrics: accuracy, precision, recall, F1‑score for each risk category.
Evaluate end‑to‑end performance: does retrieval+generation improve over ratios‑only?

Documentation & Code Quality
Provide clear README with setup & run instructions.
Include a short design diagram (e.g. in Markdown or as an image).
Comment your code; adhere to PEP8

## 5. Deliverables
GitHub Repository containing:
notebooks/ or src/ with all code.
data/ (or scripts to download data).


### Setup & dependency steps
all dependencies are in the requirements.txt file. Just run below comamnd in terminal
pip install -r requirements.txt

### Usage examples
each file contains example usage code at the end

## Important Notes
Though everything can be executed but one may face issues in below ways
1. Due to memory connstraints we can not use any model we want .
   In this example microsoft/Phi-3-mini-4k-instruct is used but ideally we must use FinGPT or FinBERT  as they are more relavent for financial data analysis.
2. CSV file has 1000 companies but we dont have quality earnnign call scripts and news articles for each company . I put only 10 text files with call scripts and news articles
3. We need to make 1000 calls to hugging face LLM to get each company level signal. Due to this repetative nature we get hugging face service unavailbe error.
   We may need to go for hugging face paid service to avoid this.

