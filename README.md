
# CardioLens AI
### AI-Powered Heart Health Assessment

CardioLens AI is an educational machine learning web application that demonstrates how cardiovascular health indicators can be used to generate heart disease prediction results using a K-Nearest Neighbors (KNN) classifier.

The project combines data exploration and model experimentation in Google Colab with a Flask-based web application, creating an end-to-end machine learning portfolio project.

> **Disclaimer:** This project is intended for educational and portfolio purposes only. It is not a medical diagnostic tool and must not be used to make healthcare decisions.

---

## Project Overview

CardioLens AI demonstrates a machine learning workflow from dataset exploration and model evaluation to web application deployment.

Data exploration, preprocessing, baseline model comparison, and hyperparameter tuning are performed in Google Colab. The selected KNN classifier is integrated into a Flask application that allows users to submit cardiovascular health indicators and view the model's prediction output.

### Key Features

- **Exploratory Data Analysis** — Investigate the dataset and target distribution in Google Colab.
- **Data Preparation** — Prepare features and target labels for model experiments.
- **Baseline Model Comparison** — Evaluate eight classification algorithms using accuracy, recall, precision, and F1-score.
- **Hyperparameter Tuning** — Compare baseline and tuned configurations for Logistic Regression, KNN, and Decision Tree.
- **KNN Integration** — Integrate the selected KNN classifier into the web application.
- **Interactive Assessment** — Submit 13 cardiovascular health indicators through a web form.
- **Visualization Reports** — Store plots from data analysis and model experiments.
- **Flask Web Application** — Serve the assessment interface and prediction functionality.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Exploration | Google Colab |
| Notebook | Jupyter Notebook |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Additional Model | XGBoost |
| Model Serialization | Joblib / Pickle |
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Styling | Tailwind CSS CDN |
| Version Control | Git, GitHub |
| Deployment Target | Render |

---

## Project Structure

```text
heart-disease/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── heart-disease-dataset.csv
│
├── notebook/
│   └── heart-disease.ipynb
│
├── models/
│   └── [trained model and preprocessing artifacts]
│
├── reports/
│   ├── target-distribution-before.png
│   ├── target-distribution-after.png
│   ├── naive-bayes-baseline.png
│   ├── logistic-regression-baseline.png
│   ├── knn-baseline.png
│   ├── svm-baseline.png
│   ├── decision-tree-baseline.png
│   ├── random-forest-baseline.png
│   ├── gradient-boosting-baseline.png
│   ├── xgboost-baseline.png
│   ├── logistic-regression-tuned.png
│   ├── knn-tuned.png
│   └── decision-tree-tuned.png
│
├── templates/
│   └── index.html
│
└── static/
    ├── css/
    │   └── style.css
    ├── js/
    │   └── script.js
    └── images/
        ├── cardiolens-social-preview.png
        ├── site.webmanifest
        ├── android-chrome-192x192.png
        ├── android-chrome-512x512.png
        ├── apple-touch-icon.png
        ├── favicon.ico
        ├── favicon-16x16.png
        └── favicon-32x32.png
```

**Structure notes:**

- `data/` stores the dataset used for the project.
- `notebook/heart-disease.ipynb` contains the notebook used for experimentation in Google Colab.
- `models/` stores the trained classifier and any preprocessing artifacts required by Flask.
- `reports/` contains visualizations generated during analysis and model evaluation.
- `templates/` contains the HTML template.
- `static/` contains CSS, JavaScript, favicon files, and social preview assets.

The dataset and model filenames above are illustrative. Replace them with the actual filenames used in the repository.

---

## Data Exploration and Experimentation

Data exploration and machine learning experiments are conducted in **Google Colab** using the notebook:

`notebook/heart-disease.ipynb`

The notebook is the primary workspace for:

1. Loading and inspecting the dataset.
2. Examining target distribution before and after data preparation.
3. Exploring feature characteristics and data quality.
4. Preparing input features and target labels.
5. Comparing baseline classification algorithms.
6. Tuning selected models.
7. Evaluating classification performance.
8. Exporting the selected model and required preprocessing artifacts.

The notebook is retained in the repository so that the experimental workflow can be reviewed alongside the deployed application.

### Experimentation Workflow

1. Load the dataset into Google Colab.
2. Perform data exploration and preparation.
3. Evaluate baseline classification models.
4. Tune selected model configurations.
5. Compare accuracy, recall, precision, and F1-score.
6. Select the model configuration for application integration.
7. Export the required model artifacts.
8. Load and test the artifacts in the Flask application.

The Flask application uses the exported model artifacts for inference. It does not need to rerun the notebook or retrain the classifier for each prediction.

---

## Dataset

The project uses a heart disease dataset stored in the `data/` directory.

The dataset is explored in Google Colab to understand its structure, target distribution, and input features before model experimentation.

The exact dataset source, number of records, preprocessing details, and class distribution should be documented from the actual dataset and notebook results.

### Analysis Reports

The `reports/` directory stores visual outputs from the experimentation workflow, including:

- Target distribution before and after data preparation.
- Baseline results for eight classification algorithms.
- Tuned Logistic Regression results.
- Tuned K-Nearest Neighbors results.
- Tuned Decision Tree results.

These visualizations support the documentation of model comparison and evaluation.

---

## Machine Learning Workflow

### 1. Exploratory Data Analysis

Inspect the dataset, understand its characteristics, and examine the target distribution in Google Colab.

### 2. Data Preparation

Prepare the features and target labels for classification experiments. Apply the preprocessing required by each model.

### 3. Baseline Model Comparison

Evaluate eight classification algorithms using four metrics:

- Accuracy
- Recall
- Precision
- F1-score

### 4. Hyperparameter Tuning

Tune selected models to assess whether their classification performance improves compared with the baseline configurations.

### 5. Model Selection

Compare the baseline and tuned results, considering both overall classification performance and the ability to identify positive cases.

KNN was selected for integration into the current CardioLens AI application. The tuned KNN configuration achieved the highest F1-score among the three tuned models evaluated in the reported comparison.

### 6. Model Export and Integration

Export the selected classifier and any required preprocessing artifacts from the experimentation environment. Load these artifacts in Flask using the same feature order and preprocessing transformations expected during training.

### 7. Web-Based Prediction

Users submit 13 cardiovascular health indicators through the assessment form. Flask processes the input and returns the classifier's prediction.

---

## Baseline Model Evaluation

The following table summarizes the baseline model evaluation results from Google Colab.

| Model | Accuracy | Recall | Precision | F1-Score |
|---|---:|---:|---:|---:|
| Naive Bayes | 0.7869 | 0.7576 | 0.8333 | 0.7937 |
| Logistic Regression | 0.8033 | 0.8485 | 0.8000 | 0.8235 |
| K-Nearest Neighbors | 0.7869 | 0.8485 | 0.7778 | 0.8116 |
| Support Vector Machine | 0.7705 | 0.8182 | 0.7714 | 0.7941 |
| Decision Tree | 0.8033 | 0.8182 | 0.8182 | 0.8182 |
| Random Forest | 0.7541 | 0.7879 | 0.7647 | 0.7761 |
| Gradient Boosting | 0.7213 | 0.7576 | 0.7353 | 0.7463 |
| XGBoost | 0.7213 | 0.7576 | 0.7353 | 0.7463 |

### Baseline Findings

- **Highest baseline accuracy:** Logistic Regression and Decision Tree, both at 0.8033.
- **Highest baseline recall:** Logistic Regression and KNN, both at 0.8485.
- **Highest baseline precision:** Naive Bayes at 0.8333.
- **Highest baseline F1-score:** Logistic Regression at 0.8235.

These results show that different algorithms perform differently across the evaluation metrics. No single metric fully captures every aspect of classification performance.

---

## Baseline vs. Tuned Model Comparison

Three models were evaluated before and after hyperparameter tuning: Logistic Regression, K-Nearest Neighbors, and Decision Tree.

| Model | Accuracy (Baseline) | Accuracy (Tuned) | Recall (Baseline) | Recall (Tuned) | Precision (Baseline) | Precision (Tuned) | F1 (Baseline) | F1 (Tuned) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8033 | 0.8033 | 0.8485 | 0.8485 | 0.8000 | 0.8000 | 0.8235 | 0.8235 |
| K-Nearest Neighbors | 0.7869 | 0.8033 | 0.8485 | 0.8788 | 0.7778 | 0.7838 | 0.8116 | 0.8286 |
| Decision Tree | 0.8033 | 0.7213 | 0.8182 | 0.7273 | 0.8182 | 0.7500 | 0.8182 | 0.7385 |

### Tuning Findings

**K-Nearest Neighbors**

- Accuracy increased from 0.7869 to 0.8033.
- Recall increased from 0.8485 to 0.8788.
- Precision increased from 0.7778 to 0.7838.
- F1-score increased from 0.8116 to 0.8286.

Among the three tuned models, KNN achieved the highest reported F1-score and recall.

**Logistic Regression**

The reported metrics remained unchanged after tuning, with an accuracy of 0.8033 and an F1-score of 0.8235.

**Decision Tree**

The tuned configuration performed worse than its baseline across all four reported metrics. Accuracy decreased from 0.8033 to 0.7213, while F1-score decreased from 0.8182 to 0.7385.

### Model Selection Rationale

The tuned KNN model was selected for integration into CardioLens AI because it achieved the highest F1-score among the three tuned models and the highest recall in the reported tuned comparison.

Its recall of 0.8788 indicates that it identified a larger proportion of positive cases in the evaluated dataset than the other tuned models. This result should be interpreted within the context of the dataset and evaluation procedure; it does not establish clinical effectiveness.

---

## Model Input Features

The assessment interface uses 13 cardiovascular health indicators.

| Feature | Description |
|---|---|
| `age` | Age |
| `sex` | Encoded sex category |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar indicator |
| `restecg` | Resting electrocardiographic result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina indicator |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia-related encoded category |

Categorical values must follow the encoding scheme used during training. The Flask application must use the same feature order and preprocessing transformations expected by the saved model.

---

## Getting Started

### Prerequisites

- Python installed locally.
- pip for Python package management.
- Git for version control.
- The trained model and required preprocessing artifacts.
- Google Colab for reviewing or rerunning experiments.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/heart-disease.git
cd heart-disease
```

Replace `YOUR-USERNAME` with the actual GitHub username.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify Model Artifacts

Confirm that all required model and preprocessing files are present at the paths referenced by `app.py`.

The artifacts must be compatible with the installed Python libraries and the preprocessing workflow used during training.

### 5. Run the Application

```bash
python app.py
```

Open the application in your browser:

http://127.0.0.1:5000

This address is accessible from the computer running the application. A public deployment is required for access by other users.

---

## Deployment

CardioLens AI is designed to be deployable as a Python web service on Render.

### General Deployment Workflow

1. Push the project to GitHub.
2. Create a new Web Service in Render.
3. Connect the GitHub repository.
4. Configure the build command.
5. Configure the production start command.
6. Ensure model artifacts are available during deployment.
7. Test the deployed website and prediction functionality.

Assuming the Flask instance is named `app` in `app.py`, a typical Render configuration is:

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
gunicorn app:app
```

Adjust these commands if the application entry point or Flask instance name differs.

---

## Responsible Use

- This project is intended for educational and portfolio purposes only.
- Predictions are model outputs, not confirmed medical diagnoses.
- Model performance may be affected by limitations or biases in the training data.
- Predictions must not replace professional medical evaluation.
- Avoid entering identifiable or sensitive patient information into a public demo.
- Do not use the application for clinical decision-making.

## Limitations

- Model performance depends on the dataset, preprocessing, validation strategy, and training configuration.
- The reported evaluation metrics do not establish clinical validity.
- The application is not a validated clinical decision-support system.
- Performance on unseen data may differ from the reported results.
- Hosting availability and response time depend on the deployment service.

---

## Future Improvements

Potential improvements include:

- Documenting the dataset source and preprocessing decisions.
- Adding automated tests for input validation and prediction routes.
- Improving error handling and monitoring.
- Optimizing frontend performance and accessibility.
- Recording model versions and dependency versions.
- Improving reproducibility between Google Colab and production inference.
- Adding deployment health checks and continuous integration.

---

## License

No license has been specified yet. Unless a license is added to this repository, reuse and redistribution are not automatically granted.

---

**CardioLens AI**  
*An educational machine learning project exploring cardiovascular health prediction.*