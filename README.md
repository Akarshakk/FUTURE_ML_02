# Support Ticket Classification & Prioritization

This project implements a Machine Learning decision-support system to automatically classify and prioritize customer support tickets. By analyzing the textual content of a ticket (Subject + Description), the system predicts the most relevant category (e.g., Billing, Technical issue) and assigns an appropriate urgency level (Low, Medium, High). 

This helps SaaS companies and IT teams respond faster, reduce their backlog, and ensure urgent issues are not overlooked.

## Project Structure

```
.
├── data/                                 # Datasets are placed here.
│   └── customer_support_tickets.csv      # Customer Support Ticket Dataset
├── notebooks/
│   └── Ticket_Classification_Prioritization.ipynb  # Main ML Jupyter Notebook
├── src/
│   ├── generate_mock_data.py             # Script to generate a synthetic dataset
│   ├── save_notebook.py                  # Utility to dynamically build the IPython Notebook
│   └── predict_ticket.py                 # Standalone script for testing new tickets
├── models/                               # Saved models (.pkl) will be generated here
├── requirements.txt                      # Project dependencies
└── README.md                             # You are here
```

## Methodology

### 1. Data Cleaning & Feature Extraction
We start by combining the `Ticket Subject` and `Ticket Description` into a single textual feature called `Full_Text` to enrich the signal for the models.
The text goes through a preprocessing pipeline:
- Converted to lowercase.
- Punctuation and special characters removed.
- Extra spaces stripped.

Then, we transition the texts into numerical representations using **TF-IDF (`TfidfVectorizer`)**. TF-IDF gives more weight to words that are strongly associated with specific tickets across the entire dataset rather than words that are universally common.

### 2. Categorization
The **Ticket Category** defines the domain of the problem. We use **Logistic Regression** for the multi-class categorization task due to its efficiency and interpretability with TF-IDF features. 

### 3. Prioritization
The **Ticket Priority** dictates urgency. We utilize a **Random Forest Classifier**, an ensemble model that handles complex combinations of features effectively, allowing it to better capture the nuances indicating a "Critical" vs. "Low" priority ticket.

### 4. Evaluation Metrics
The notebook measures the model using:
- **Accuracy**: Percentage of globally correct predictions.
- **Precision/Recall**: Per-class metrics via Scikit-Learn `classification_report` showing our effectiveness in handling imbalanced priority classes.
- **Confusion Matrix**: Visual heatmap displaying exactly where the model gets confused (e.g., mixing up "General Inquiry" and "Account Access").

## How to Run Locally

**1. Install Dependencies**
Ensure you have Python 3 installed. Install required packages using:
```bash
pip install -r requirements.txt
```

**2. Open the Jupyter Notebook**
```bash
jupyter notebook notebooks/Ticket_Classification_Prioritization.ipynb
```
Follow the notebook cells to process the data, train the models, and inspect the classification metrics and heatmaps. This will also save `tfidf_vectorizer.pkl`, `ticket_categorizer.pkl`, and `ticket_prioritizer.pkl` in the `models/` folder.

**3. Test the Inference Script**
Once the models are saved, run:
```bash
python src/predict_ticket.py --subject "Cannot login to dashboard" --description "I am getting an Error 500 when accessing the portal with my new credentials."
```
