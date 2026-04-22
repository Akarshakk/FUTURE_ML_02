# Support Ticket Classification & Prioritization System

![Architecture](https://img.shields.io/badge/Machine_Learning-Decision_Support-blue) ![NLP](https://img.shields.io/badge/NLP-TF--IDF-orange)

## Deployed app link: https://futureml02-9lfj5gomxb4z8y9exmmqeb.streamlit.app

## Executive Summary: Business Impact
In modern SaaS companies and IT support operations, support teams waste thousands of hours manually reading, tagging, and routing incoming tickets. This manual triage causes critical issues (like outages or account compromises) to be delayed in the queue behind general inquiries.

This project implements an **Automated Machine Learning Decision-Support System** that acts as an intelligent first-responder. When a customer submits a ticket, this system reads the text and instantly:
1. **Categorizes the issue** (e.g., Billing, Technical, Account Access) so it reaches the right department immediately.
2. **Assigns a Priority Level** (Low, Medium, High) so urgent issues are bumped to the top of the queue.

This solution directly reduces response latency, clears backlog, and optimizes support operations. 

---

## Technical Methodology

### 1. How Tickets Are Categorized
When a new ticket arrives, the system concatenates the **Subject** and **Description**. 
- **Text Processing**: It cleans the text (removing punctuation, normalizing lower-case, stripping extra whitespace).
- **Feature Engineering**: It converts the text into a mathematical format using **TF-IDF (Term Frequency-Inverse Document Frequency)**. This algorithm identifies words that strongly indicate a specific category (e.g., "invoice" implies a Billing ticket, "crashes" implies a Technical Issue) while ignoring meaningless frequent words like "the" or "and".
- **The Categorization Model**: We use a **Logistic Regression** multi-class model. It is highly efficient for high-dimensional text data and securely maps the extracted keywords to structured categorical buckets with high confidence probabilities.

### 2. How Priority is Decided
Assigning priority is often more complex than just identifying a topic. A "Technical Issue" might be a Low-priority UI bug, or a Critical server outage.
- **The Priority Model**: We utilize a **Random Forest Classifier**. This is an ensemble model made up of hundreds of decision trees. It is brilliant at recognizing complex, non-linear relationships. For instance, the combination of "account" + "locked" + "urgent" might trigger a High priority prediction, whereas "account" + "change picture" registers as Low.

---

## Evaluation Results & Insights

During training and validation, the models are evaluated based on real-world operational metrics:
- **Accuracy**: Indicates the overall percentage of tickets routed perfectly without human intervention.
- **Precision & Recall (Class-Wise)**: It's vital not to miss *High Priority* tickets. We track "Recall" on the High-priority class to ensure false negatives are minimized. A false positive (a medium ticket flagged as high) is acceptable, but a false negative (a high ticket flagged as low) damages customer trust.
- **Confusion Matrix**: A visual heat map that shows exactly where the model gets "confused." 
  - *Business Insight*: If the model frequently confuses "General Query" with "Feature Request", it indicates the actual support team might also struggle to visually separate these, suggesting those two tags could be combined operationally into a single bucket.

---

## Project Structure

```
.
├── data/                                 # Datasets (CSV)
├── notebooks/
│   └── Ticket_Classification_Prioritization.ipynb  # Main ML Jupyter Notebook containing Model Training & Evaluation
├── src/
│   ├── app.py                            # Streamlit GUI Web Application
│   ├── predict_ticket.py                 # CLI Inference script 
│   └── generate_mock_data.py             # Synthetic data generation fallback
├── models/                               # Serialized (.pkl) trained ML models sit here
├── requirements.txt                      # Project dependencies
└── README.md                             # You are here
```

## How to Run Locally

**1. Install Dependencies**
Ensure you have Python 3 installed.
```bash
pip install -r requirements.txt
```

**2. Train the Models (Jupyter Notebook)**
Launch the notebook to see the data processing, model training, and evaluation heatmaps. Running this saves the needed models into the `/models` directory.
```bash
jupyter notebook notebooks/Ticket_Classification_Prioritization.ipynb
```

**3. Launch the Web UI Frontend**
I've built a sleek UI to visually demonstrate the AI to stakeholders.
```bash
streamlit run src/app.py
```

**4. Test via CLI**
```bash
python src/predict_ticket.py --subject "Cannot login to dashboard" --description "I am getting an Error 500 when accessing the portal with my new credentials."
```
