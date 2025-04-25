import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def load_and_preprocess_csv(csv_path):
    """Loads and preprocesses the company CSV data."""
    df = pd.read_csv(csv_path)

    # handle missing values - replace with mean/median or drop
    for col in ['debt_equity', 'current_ratio', 'roe']:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True) # fill with mean

    # Normalize numerical features
    scaler = MinMaxScaler()
    df[['debt_equity_normalized', 'current_ratio_normalized', 'ROE_normalized']] = scaler.fit_transform(df[['debt_equity', 'current_ratio', 'roe']])

    return df

def load_and_preprocess_text_data(text_folder):
    """Loads and preprocesses text data from the specified folder."""
    text_data = {}
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for filename in os.listdir(text_folder):
        if filename.endswith('.txt'):
            company_name = os.path.splitext(filename)[0] # Assuming filename is company name
            filepath = os.path.join(text_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().lower()
                # Basic cleaning: remove non-alphanumeric characters and extra spaces
                #text = re.sub(r'[^a-z0-9\s]', '', text)
                text = re.sub(r"\s+", " ", text)
                tokens = word_tokenize(text)
                tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 2]
                text_data[company_name] = " ".join(tokens)
    return text_data

# --- Example Usage Below---
csv_file = 'data/company.csv'
text_directory = 'data/text_data'

if __name__ == "__main__":
    
    company_df = load_and_preprocess_csv(csv_file)
    text_corpus = load_and_preprocess_text_data(text_directory)

    print("Processed CSV DataFrame:")
    print(company_df.head())
    print("\nProcessed Text Corpus - top 3 companies")
    print(dict(list(text_corpus.items())[:3]))