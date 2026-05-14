# SmartSip ☕ — Customer Analytics System

> A full-stack machine learning web application that analyzes cafe order data to predict cancellations, estimate wait times, forecast rush hours, suggest combo deals, and segment customers by taste — all served through a live Flask REST API with an interactive dashboard.


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/79d2b4a8-6745-4762-ac9d-73234cdd9699" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ed4289eb-153d-4b8d-bf4a-64b766154e5d" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2ad5613d-e2aa-4617-bcf7-bf23e975f0c3" />

---

## 📌 Overview

SmartSip processes **1,943 orders** across **21 menu items** to train 5 machine learning models that power real-time business predictions. The frontend is a single-page dashboard with live sliders, Chart.js visualizations, and instant ML inference — no page reloads.

---

## 🤖 ML Models

| Model | Task | Algorithm | Performance |
|---|---|---|---|
| 🚦 Cancel Predictor | Will this order be cancelled? | Logistic Regression | ~74% Accuracy |
| ⏱️ Wait Time Estimator | How long will the customer wait? | Linear Regression | R² = 0.94 |
| 🔥 Rush Hour Forecast | Is this time slot going to be busy? | Decision Tree | ~77% Accuracy |
| 🍭 Taste Profiler | What type of customer is this? | K-Means (k=3) | Silhouette = 0.81 |
| 🎁 Combo Deal Suggester | What items pair well together? | Apriori (Association Rules) | Conf. up to 31% |

---

## 🧠 Features

- **8 Business Insights** — peak hours, payment trends, top sellers, customer segments, and more
- **Real-time predictions** — all 5 models respond via REST API with live input from the dashboard
- **Interactive charts** — top items, revenue trend, category sales, payment split (Chart.js)
- **Customer segmentation** — Sweet Lovers 🍰, Snack Lovers 🥨, Spicy Lovers 🌶️
- **Combo recommendations** — basket analysis suggests the best item pairings for bundle deals

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Zunaisha-Javaid/SmartSip.git
cd SmartSip

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset (run once)
python generate_data.py

# 4. Train all 5 models (run once)
python train_models.py

# 5. Start the server
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
smartsip/
│
├── app.py                  ← Flask backend + 12 REST API endpoints
├── generate_data.py        ← Generates realistic dataset (1,943 orders)
├── train_models.py         ← Trains & saves all 5 ML models
│
├── requirements.txt        ← Python dependencies
│
├── data/
│   └── smartsip.csv        ← 1,943 rows × 16 features
│
├── models/
│   ├── cancel_model.pkl    ← Logistic Regression
│   ├── wait_model.pkl      ← Linear Regression
│   ├── rush_model.pkl      ← Decision Tree
│   ├── kmeans_model.pkl    ← K-Means Clustering
│   ├── combo_rules.pkl     ← Apriori Association Rules
│   └── summary.json        ← Model scores + dataset stats
│
├── templates/
│   └── index.html          ← Single-page frontend
│
└── static/
    ├── css/style.css       ← Cafe aesthetic UI design
    └── js/main.js          ← API calls, charts, interactions
```

---

## 🗂️ Dataset

| Property | Value |
|---|---|
| Total orders | 1,943 |
| Menu items | 21 |
| Unique customers | 500 |
| Features | 16 |
| Target variables | `cancel_status`, `wait_time_mins`, `rush_level` |

Key features: `hour_of_day`, `day_of_week`, `item_type`, `transaction_amount`, `transaction_type`, `orders_in_last_60_min`, `rush_level`, `is_holiday`

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML | scikit-learn, Pandas, NumPy |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Data | Custom generated CSV (1,943 rows) |

---

## 👩‍💻 Author

**Zunaisha Javaid**  
[GitHub](https://github.com/Zunaisha-Javaid) · Computer Science Student
