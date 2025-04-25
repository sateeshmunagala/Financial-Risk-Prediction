# 4.Prediction Module
# Based on LLM output, map its qualitative assessment to one of three risk levels. 
# --> This has been done in the Task3_RAG_Pipeline_Implementation.py file
# Alternatively, train a simple classifier (e.g. logistic regression) on ratio features and combine with LLM signals. 
# --> This is implemented in this current file.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

# import Task1 python file
import Task1_DataIngestion_Preprocessing as data_ingestion

# import Task2 python file
import Task2_Vector_Indexing as vector_indexing

# import Task3 python file
import Task3_RAG_Pipeline_Implementation as rag_pipeline

def train_and_evaluate_classifier(company_df, llm_text_features):
    """Trains and evaluates a Logistic Regression classifier with combined features."""
    # Prepare features (normalized ratios + LLM text features)
    ratio_features = company_df[['debt_equity_normalized', 'current_ratio_normalized', 'ROE_normalized']].values
    X_combined = np.concatenate((ratio_features, llm_text_features), axis=1)
    y = company_df['risk_level'].map({'low': 0, 'medium': 1, 'high': 2}).values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    model = LogisticRegression(random_state=42, solver='liblinear', multi_class='ovr')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    return model, accuracy, precision, recall, f1

# ---  Usage ---

csv_file = 'data/company.csv'
text_directory = 'data/text_data'

company_df = data_ingestion.load_and_preprocess_csv(csv_file)
text_corpus = data_ingestion.load_and_preprocess_text_data(text_directory)
vector_index = vector_indexing.create_vector_index(text_corpus)

# --- Prepare LLM text features for all companies ---
all_company_names = company_df['company_id'].tolist()
all_ratios = company_df[['debt_equity_normalized', 'current_ratio_normalized', 'ROE_normalized']].to_dict()

all_llm_responses = [
    rag_pipeline.rag_pipeline(row['company_id'], row[['debt_equity_normalized', 'current_ratio_normalized', 'ROE_normalized']].to_dict(), vector_index)
    for index, row in company_df.iterrows()
]
tfidf_vectorizer_all = TfidfVectorizer(stop_words='english')
all_llm_text_features = tfidf_vectorizer_all.fit_transform(all_llm_responses).toarray()

# --- Train and Evaluate ---
trained_model, accuracy, precision, recall, f1 = train_and_evaluate_classifier(company_df, all_llm_text_features)

print("--- Classifier Evaluation ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted): {recall:.4f}")
print(f"F1-score (weighted): {f1:.4f}")
