# Customer Churn Prediction (End-to-End MLOps Pipeline)

## 📌 Overview
This project is an end-to-end machine learning system that predicts customer churn for a telecom business. Beyond just building a predictive model, it covers the complete ML lifecycle from data exploration and cleaning, to model training and experiment tracking, to deployment as a production-ready API. The goal is to demonstrate real-world MLOps practices: containerization, CI/CD automation, and model monitoring.

## 🎯 Problem Statement
Customer churn means when a customer stops using a company's service. This is one of the most costly problems telecom businesses face. Acquiring a new customer is significantly more expensive than retaining an existing one. This project builds a machine learning system that identifies customers at high risk of churning, enabling the business to take proactive retention action before it's too late.

## 🗂️ Dataset
- **Source:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 customer records
- **Features:** Customer demographics, account information, services subscribed (phone, internet, streaming, etc.), and billing details
- **Target:** Whether the customer churned (Yes/No)

## 🏗️ Project Architecture
Raw Data → Data Cleaning & EDA → Feature Engineering → Model Training (with MLflow tracking) → Best Model Selection → FastAPI Deployment → Dockerized Container → CI/CD via GitHub Actions → Monitoring & Drift Detection

## 🛠️ Tech Stack
| Layer | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| API Framework | FastAPI |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Custom drift detection scripts |

## 📁 Project Structure
churn-mlops/
├── data/               # Raw and processed data (not pushed to GitHub)
├── notebooks/          # EDA and experimentation notebooks
├── src/
│   ├── pipeline.py     # Data preprocessing pipeline
│   ├── train.py        # Model training script
│   └── predict.py      # Prediction logic
├── api/
│   └── main.py         # FastAPI application
├── monitoring/
│   └── monitor.py       # Model drift detection
├── tests/
│   └── test_api.py     # Automated tests
├── .github/workflows/  # CI/CD pipeline configuration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

## 🚦 Project Status
🚧 **In Progress**

- [x] Project structure and environment setup
- [ ] Phase 1: Data cleaning and exploratory data analysis
- [ ] Phase 2: Model training and evaluation
- [ ] Phase 3: Experiment tracking with MLflow
- [ ] Phase 4: API development with FastAPI
- [ ] Phase 5: Dockerization
- [ ] Phase 6: Monitoring and drift detection
- [ ] Phase 7: CI/CD pipeline
- [ ] Phase 8: Final documentation

## ⚙️ How to Run
*Instructions will be added as each phase is completed.*

```bash
# Clone the repository
git clone https://github.com/sarmelaraj/churn-mlops.git
cd churn-mlops

# Set up virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📈 Results
*Model performance metrics and comparisons will be added after Phase 2.*

## 🔍 Key Learnings
*To be added as the project progresses.*

## 📬 Contact
Dr Sarmela Raja Sekaran
www.linkedin.com/in/sarmela-raja-sekaran-980187178
sarmelaraj07@gamil.com