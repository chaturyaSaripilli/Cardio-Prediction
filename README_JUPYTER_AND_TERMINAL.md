
# Cardiovascular Disease Prediction

This project aims to predict the presence of cardiovascular (heart) disease using machine learning models based on patient medical attributes. It includes training, evaluation, and prediction components, as well as a web application interface and Jupyter notebook support for local analysis.

---

## Project Overview

Heart disease is one of the leading causes of death globally. Early detection using clinical and biometric data can assist in preventing complications. This project uses the following steps:

- Data preprocessing and scaling
- Model training using multiple ML algorithms
- Model evaluation and performance comparison
- Final model selection and saving (`.pkl`)
- Streamlit-based web application for user-friendly prediction
- CSV and SQLite database logging of prediction results
- Jupyter Notebook version for interactive analysis

---

## Dataset

- File: `heart.csv`
- Columns used:
  - `age`: Age of the patient
  - `sex`: Sex (1 = male; 0 = female)
  - `cp`: Chest pain type (0–3)
  - `trestbps`: Resting blood pressure
  - `chol`: Serum cholesterol in mg/dl
  - `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
  - `restecg`: Resting electrocardiographic results (0–2)
  - `thalach`: Maximum heart rate achieved
  - `exang`: Exercise-induced angina (1 = yes; 0 = no)
  - `oldpeak`: ST depression induced by exercise
  - `slope`: Slope of the peak exercise ST segment
  - `ca`: Number of major vessels (0–3)
  - `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect)
  - `target`: Target variable (1 = heart disease, 0 = no heart disease)

---

## Folder Structure

```
CardioPredictionProject/
├── app.py
├── heart.csv
├── heart_disease_model.pkl
├── heart_disease_prediction.py
├── heart_disease_prediction.ipynb
├── prediction_history.csv (auto-created)
├── predictions.db (auto-created)
├── README.md
```

---

## How to Run the Project in Terminal

1. **Install dependencies**
```bash
pip install -r requirements.txt
```
If `requirements.txt` is not provided, manually install:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
```

2. **Train the model and generate .pkl**
```bash
python heart_disease_prediction.py
```

3. **Run the web application**
```bash
streamlit run app.py
```

This opens a browser where users can input patient details and receive predictions. Each prediction is saved in both `.csv` and `.db` formats.

---

## How to Run in Jupyter Notebook

1. Open terminal:
```bash
jupyter notebook
```

2. Open `heart_disease_prediction.ipynb`

3. Run each cell step by step. The notebook includes:
   - Data loading and preprocessing
   - Model training and evaluation
   - Saving the model
   - Entering new patient data
   - Saving predictions to CSV and SQLite

---

## Screenshots and Output Placement

You can capture and place the following outputs in your GitHub repository:

1. **Confusion Matrices and Accuracy Reports**  
   ➤ Output from model evaluation in both `heart_disease_prediction.py` and notebook  
   ➤ Place them in a new folder:
   ```
   CardioPredictionProject/outputs/
   ```

2. **Web App Interface**  
   ➤ Screenshot of Streamlit form with predictions  
   ➤ Place in the same `/outputs/` folder  
   ➤ Refer to them in your README if needed using:
   ```markdown
   ![Model Output](outputs/confusion_matrix_rf.png)
   ```

---

## Final Notes

- Use either the terminal-based `app.py` or the Jupyter Notebook version based on your preference.
- All predictions are logged for future reference.
- You can enhance the model by exploring other algorithms or performing hyperparameter tuning.

