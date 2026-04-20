import pandas as pd
import random
import numpy as np
import os

def generate_mock_data(num_samples=1000):
    """
    Generates a synthetic customer support ticket dataset mimicking the Kaggle dataset structure.
    Target Columns for ML: Subject, Description, Ticket Type, Priority
    """
    ticket_types = ["Billing", "Technical issue", "Account access", "General inquiry", "Feature Request", "Cancellation"]
    priorities = ["Low", "Medium", "High", "Critical"]
    
    # Templates for synthetic generation based on Ticket Type
    subjects_templates = {
        "Billing": ["Overcharged on my last invoice", "Need a refund for {product}", "Double billing issue", "Payment failed", "Update billing details"],
        "Technical issue": ["App crashes on {product}", "Error 500 when logging in", "Cannot sync my {product}", "Integration not working", "Slow performance"],
        "Account access": ["Forgot password", "Account is locked", "Cannot verify email", "Change email address", "Suspended account"],
        "General inquiry": ["How does {product} work?", "Do you have student discounts?", "Contacting sales team", "Where is the documentation?", "Pricing question"],
        "Feature Request": ["Please add {feature} to {product}", "Dark mode support", "API rate limit increase", "Export to CSV feature", "Mobile app request"],
        "Cancellation": ["Cancel my subscription", "How to delete account?", "Stop auto-renewal", "Refund and cancellation", "Opt out of service"]
    }
    
    descriptions_templates = {
        "Billing": ["Hi, I noticed I was billed twice this month. Please help.", "My payment failed but money was deducted.", "I need a refund for my recent purchase of {product}. It is not what I expected.", "Please update my credit card info.", "I do not understand the recent charges on my invoice."],
        "Technical issue": ["Every time I open {product}, it crashes.", "I am getting an Error 500 when accessing the dashboard.", "The sync feature for {product} is broken.", "My data is not loading. Please fix this soon.", "The application is extremely slow today."],
        "Account access": ["I forgot my password and the reset link is not arriving.", "My account says it is locked due to multiple attempts.", "I lost my 2FA device. Please help me log in.", "I need to change my associated email address.", "Why is my account suspended?"],
        "General inquiry": ["Can you explain the difference between the pro and basic plans?", "I am a student, do I get a discount for {product}?", "How can I contact sales for an enterprise deal?", "I am looking for documentation on how to use {product}.", "What are the pricing tiers?"],
        "Feature Request": ["It would be great if you could add {feature} to {product}.", "When are you guys adding dark mode?", "Please increase the API rate limit for our startup.", "I really need an export to CSV feature for reporting.", "Any plans for an Android app?"],
        "Cancellation": ["I want to cancel my subscription immediately.", "Please delete all my data and close my account.", "How do I turn off auto-renew?", "I am not using the service, please refund and cancel.", "I want to opt out of your service and emails."]
    }

    products = ["Cloud Storage", "Analytics Dashboard", "Mobile App", "Desktop Client", "Email Client"]
    features = ["Single Sign-On", "Multi-factor authentication", "Audit logs", "Custom domains", "Webhooks"]
    
    data = []
    
    for i in range(num_samples):
        # Pick a random ticket type
        t_type = random.choice(ticket_types)
        
        # Pick subject and description
        subject = random.choice(subjects_templates[t_type])
        desc = random.choice(descriptions_templates[t_type])
        
        # Format strings if they contain placeholders
        p = random.choice(products)
        f = random.choice(features)
        
        subject = subject.replace("{product}", p).replace("{feature}", f)
        desc = desc.replace("{product}", p).replace("{feature}", f)
        
        # Determine Priority based on Ticket Type + some randomness
        if t_type == "Technical issue":
            priority = random.choice(["Medium", "High", "Critical"])
        elif t_type in ["Billing", "Cancellation"]:
            priority = random.choice(["Medium", "High"])
        elif t_type == "Account access":
            priority = random.choice(["High", "Critical"])
        else:
            priority = random.choice(["Low", "Medium"])
            
        data.append({
            "Ticket ID": f"TKT-{1000 + i}",
            "Customer Name": f"Customer {i}",
            "Ticket Type": t_type,
            "Subject": subject,
            "Description": desc,
            "Priority": priority,
        })
    
    df = pd.DataFrame(data)
    
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    output_path = 'data/customer_support_tickets_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Generated {num_samples} mock tickets and saved to {output_path}")

if __name__ == "__main__":
    generate_mock_data(2500)
