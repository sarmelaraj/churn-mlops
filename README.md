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
## 📊 Phase 1 Results — EDA & Data Preprocessing

### Key Findings from EDA
- **26.6%** of customers churned — dataset is imbalanced, will be addressed in modeling phase
- **Contract type** is a strong churn predictor — month-to-month customers churn at a significantly higher rate (~75%) compared to one year (~11%) and two year (~3%) contracts
- **Tenure** is inversely related to churn — customers in their first 10 months are at highest risk, churn drops significantly after 12 months
- **Monthly charges** are higher for churned customers — median of $80 vs $65 for retained customers, suggesting pricing is a key driver of attrition
- **TotalCharges** is strongly correlated with tenure (0.83) — longer customers naturally accumulate higher total charges

### Data Preprocessing Steps
- Converted `TotalCharges` from string to numeric and handled 11 null values
- Dropped `customerID` (non-predictive identifier)
- Encoded `Churn` target variable from Yes/No to 1/0
- Applied `pd.get_dummies()` encoding on 15 categorical columns — expanded from 20 to 31 columns
- Applied `StandardScaler` on `tenure`, `MonthlyCharges`, and `TotalCharges`
- Split data into 80/20 train/test sets (5,625 training / 1,407 test samples)

## 🤖 Phase 2 Results — Model Training & Evaluation

### Model Comparison
| Model | Accuracy | AUC Score |
|---|---|---|
| Logistic Regression | 78% | 0.83 |
| Random Forest | 77% | 0.82 |
| XGBoost | 79% | 0.83 |

### Hyperparameter Tuning
XGBoost was selected as the best performing model and tuned using GridSearchCV across 27 parameter combinations with 5-fold cross validation (135 fits total).

**Best Parameters Found:**
| Parameter | Value |
|---|---|
| learning_rate | 0.1 |
| max_depth | 3 |
| n_estimators | 100 |

### Final Model Performance
| Metric | No Churn | Churn |
|---|---|---|
| Precision | 0.83 | 0.65 |
| Recall | 0.90 | 0.48 |
| F1-Score | 0.86 | 0.55 |
| **Overall Accuracy** | | **79%** |
| **AUC Score** | | **0.852** |

### Known Limitations
- Model recall for churners (48%) is lower than ideal due to class imbalance (73.4% non-churn vs 26.6% churn)
- Future improvement: SMOTE oversampling or class weight adjustment to improve minority class detection

## 🚦 Project Status
🚧 **In Progress**

- [x] Project structure and environment setup
- [x] Phase 1: Data cleaning and exploratory data analysis
- [x] Phase 2: Model training and evaluation
- [ ] Phase 3: Experiment tracking with MLflow
- [ ] Phase 4: API development with FastAPI
- [ ] Phase 5: Dockerization
- [ ] Phase 6: Monitoring and drift detection
- [ ] Phase 7: CI/CD pipeline
- [ ] Phase 8: Final documentation

## ⚙️ How to Run

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
| Phase | Status | Key Metric |
|---|---|---|
| EDA & Preprocessing | ✅ Complete | 30 features, 7032 records |
| Model Training | ✅ Complete | XGBoost AUC: 0.852 |
| MLflow Tracking | 🚧 In Progress | - |
| API Deployment | ⬜ Pending | - |
| Docker | ⬜ Pending | - |
| CI/CD | ⬜ Pending | - |

## 🔍 Key Learnings
- Contract type and tenure are the strongest predictors of customer churn
- Class imbalance significantly impacts recall on minority classes and must be addressed in production models
- GridSearchCV improved AUC from 0.83 to 0.852 through systematic hyperparameter tuning

## 📬 Contact
Dr Sarmela Raja Sekaran
www.linkedin.com/in/sarmela-raja-sekaran-980187178
sarmelaraj07@gamil.com