import streamlit as st
import joblib
import re
import os

# Set page config for a wider, cleaner layout
st.set_page_config(
    page_title="Support Ticket Classifier & Prioritizer",
    page_icon="🎫",
    layout="centered"
)

# --- Helper Functions (Same as predict_ticket.py) ---
def clean_text(text):
    """Clean the input text identically to the training process."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_models():
    """Load the trained models from the models folder. Cached for performance."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, '..', 'models')
    
    try:
        vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        categorizer = joblib.load(os.path.join(models_dir, 'ticket_categorizer.pkl'))
        prioritizer = joblib.load(os.path.join(models_dir, 'ticket_prioritizer.pkl'))
        return vectorizer, categorizer, prioritizer
    except FileNotFoundError:
        st.cache_resource.clear()
        return None, None, None

def predict_ticket(subject, description, vectorizer, categorizer, prioritizer):
    """Predicts category and priority given ticket subject and description."""
    full_text = subject + " " + description
    cleaned_text = clean_text(full_text)
    
    vectorized_text = vectorizer.transform([cleaned_text])
    category = categorizer.predict(vectorized_text)[0]
    priority = prioritizer.predict(vectorized_text)[0]
    
    # Calculate probabilities if available (optional enhancement)
    try:
        cat_prob = max(categorizer.predict_proba(vectorized_text)[0]) * 100
        prio_prob = max(prioritizer.predict_proba(vectorized_text)[0]) * 100
    except AttributeError:
        cat_prob, prio_prob = None, None
        
    return category, priority, cat_prob, prio_prob

# --- UI Setup ---

st.title("🎫 Support Ticket AI")
st.markdown("""
Welcome to the internal Support Support decision-system. 
Enter a customer ticket below, and the machine learning model will automatically **categorize it** and **assign a priority level**.
""")

st.divider()

# Model loading status
vectorizer, categorizer, prioritizer = load_models()

if not vectorizer:
    st.error("🚨 Models not found! Please run the Jupyter Notebook first to train and export the models to the `/models` directory.")
    st.stop()

# Input fields
with st.form(key="ticket_form"):
    st.subheader("New Customer Ticket")
    subject_input = st.text_input("Subject", placeholder="e.g. Cannot login to dashboard")
    description_input = st.text_area("Description", placeholder="I am getting an Error 500 when accessing the portal with my new credentials...", height=150)
    
    submit_button = st.form_submit_button(label="Analyze Ticket")

# Processing and Results
if submit_button:
    if not subject_input or not description_input:
        st.warning("Please provide both a Subject and a Description for accurate analysis.")
    else:
        with st.spinner("Analyzing text using NLP models..."):
            cat, prio, cat_prob, prio_prob = predict_ticket(
                subject_input, description_input, vectorizer, categorizer, prioritizer
            )
            
        st.success("Analysis Complete!")
        
        # Display Results in columns
        col1, col2 = st.columns(2)
        
        # Determine color for priority dynamically
        prio_color = "green"
        if prio.lower() == "medium":
            prio_color = "orange"
        elif prio.lower() in ["high", "critical"]:
            prio_color = "red"
            
        with col1:
            st.markdown("### Category")
            st.info(f"**{cat}**")
            if cat_prob:
                st.caption(f"Confidence: {cat_prob:.1f}%")
                
        with col2:
            st.markdown("### Priority")
            st.markdown(f"<h4 style='color:{prio_color};'>{prio.upper()}</h4>", unsafe_allow_html=True)
            if prio_prob:
                st.caption(f"Confidence: {prio_prob:.1f}%")

st.divider()
st.caption("Built with Streamlit & Scikit-Learn | Support Ticket Classification & Prioritization System")
