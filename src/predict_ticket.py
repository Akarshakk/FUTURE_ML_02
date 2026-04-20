import joblib
import re
import os
import argparse

def clean_text(text):
    """Clean the input text identically to the training process."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_models():
    """Load the trained models and vectorizer from the models folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, '..', 'models')
    
    try:
        vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        categorizer = joblib.load(os.path.join(models_dir, 'ticket_categorizer.pkl'))
        prioritizer = joblib.load(os.path.join(models_dir, 'ticket_prioritizer.pkl'))
        return vectorizer, categorizer, prioritizer
    except FileNotFoundError:
        print("Error: Model files not found. Please run the Jupyter Notebook first to train and save the models.")
        exit(1)

def predict_ticket(subject, description):
    """Predicts category and priority given ticket subject and description."""
    vectorizer, categorizer, prioritizer = load_models()
    
    # Preprocess
    full_text = subject + " " + description
    cleaned_text = clean_text(full_text)
    
    # Vectorize
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Predict
    category = categorizer.predict(vectorized_text)[0]
    priority = prioritizer.predict(vectorized_text)[0]
    
    return category, priority

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Support Ticket Category and Priority")
    parser.add_argument('--subject', type=str, required=True, help="Ticket Subject")
    parser.add_argument('--description', type=str, required=True, help="Ticket Description")
    
    args = parser.parse_args()
    
    cat, prio = predict_ticket(args.subject, args.description)
    print("\n--- Ticket Prediction ---")
    print(f"Subject: {args.subject}")
    print(f"Description: {args.description}")
    print(f"-> Predicted Category: {cat}")
    print(f"-> Predicted Priority: {prio}")
    print("-------------------------\n")
