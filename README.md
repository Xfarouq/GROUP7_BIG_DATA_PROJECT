# 🚖 NYC Taxi Big Data Analytics Project

This project explores and analyzes **New York City Yellow Taxi trip data** using **Big Data tools and Machine Learning models**. It demonstrates a complete pipeline from **data ingestion → storage → processing → visualization → prediction**.

---

## 📌 Problem Statement

Despite the availability of rich NYC taxi data, several challenges persist:

* Passengers and drivers lack reliable information on **expected fare prices** before trips.
* **Tip behavior prediction** is difficult but crucial for driver income.
* **Spatiotemporal demand patterns** are complex and require advanced visualization.
* Handling **large-scale data** (tens of millions of rows) is challenging with limited local resources.

This project addresses these issues by:

* Building a pipeline to process and clean large taxi datasets.
* Exploring patterns and insights through interactive visualizations.
* Training ML models to predict **fare amount (regression)** and **tipping behavior (classification)**.
* Deploying an interactive **Streamlit dashboard**.

---

## 🏗 System Architecture

```text
NYC Taxi Data (Parquet Files) 
        ↓
Dask + Pandas (Data Ingestion & Processing)  
        ↓
Amazon S3 (Storage) 
        ↓
Feature Engineering + ML Models (Regression & Classification) 
        ↓
Streamlit Dashboard (Visualizations + Predictions)
```

---

## ⚙️ Tools & Technologies

* **Python 3.10+**
* **Amazon S3** → cloud storage for datasets
* **Dask** → parallel data ingestion
* **Pandas / PyArrow** → data preprocessing
* **Scikit-learn** → ML models (Regression & Classification)
* **Joblib** → model persistence
* **Streamlit** → interactive dashboard (basic & advanced visuals, geo-maps, predictions)
* **Plotly / Pydeck / Seaborn** → visualizations

---

## 📂 Project Structure

```bash
NYC-Taxi-BigData-Project/
│
├── big-data-project1.ipynb # Jupyter notebooks for EDA and model training
├── models/                # trained ML models (joblib files)
├── streamlit_app.py       # Streamlit dashboard
├── s3.py                  # script to connect and sample from S3
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── report/                # Final report and presentation
```

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Xfarouq/GROUP7_BIG_DATA_PROJECT.git
cd GROUP7_BIG_DATA_PROJECT
```

### 2️⃣ Set up virtual environment

```bash
python -m venv taxi_env
source taxi_env/bin/activate   # Linux/Mac
taxi_env\Scripts\activate      # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app will be available at:
👉 [http://localhost:8501](http://localhost:8501)

---

## 📊 Features

✅ Load data from **Amazon S3** (via Dask & Pandas)
✅ Perform **exploratory analysis** (distributions, correlations, demand patterns)
✅ **Geo-visualizations** of pickup density across NYC
✅ ML Models:

* **Fare Prediction (Regression)**
* **Tip Prediction (Classification)**
  ✅ Interactive **Streamlit dashboard** with filters & animations

---

## 📎 Resources

* Dataset: [Kaggle NYC Taxi Data](https://www.kaggle.com/datasets/farouqx/nyc-yellow-taxi-raw-parquet-20152016)
* Report: See `report/NYC_Taxi_BigData_Project.pdf`

---

## 📝 Future Improvements

* Deploy Streamlit app on **AWS/GCP/Heroku**.
* Integrate **real-time taxi trip data streams**.
* Experiment with **deep learning models** for demand prediction.
* Add **model explainability (SHAP/LIME)**.


