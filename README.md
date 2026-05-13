# SmartSip ☕ — Customer Analytics System
**University ML Project | Flask + scikit-learn | Deployable on Render**

---

## 🤖 What's inside

| Feature | What it does |
|---|---|
| 🚦 Cancel Predictor | Logistic Regression — predicts if an order will be cancelled |
| ⏱️ Wait Time | Linear Regression (R²=0.94) — estimates wait in minutes |
| 🔥 Rush Hours | Decision Tree — classifies if a time slot is rush |
| 🎁 Combo Deals | Association Rules (Apriori-style) — suggests item bundles |
| 🍭 Taste Profile | K-Means Clustering (k=3) — segments customers |

---

## 🚀 Deploy on Render (free, 5 minutes)

1. Push this folder to a **GitHub repo**
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-reads `render.yaml` — just click **Deploy**
5. Your site will be live at `https://smartsip.onrender.com` 🎉

---

## 💻 Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Generate dataset + train models (run once)
python generate_data.py
python train_models.py

# Start the server
python app.py

# Open browser
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
smartsip/
├── app.py                  ← Flask backend + all API routes
├── generate_data.py        ← Generates SmartSip dataset (1943 orders)
├── train_models.py         ← Trains all 5 ML models + saves to /models
├── requirements.txt        ← Python packages
├── Procfile                ← Gunicorn start command (for Render)
├── render.yaml             ← Render deploy config (auto build + train)
├── data/
│   └── smartsip.csv        ← Generated dataset (created on first run)
├── models/
│   ├── cancel_model.pkl    ← Logistic Regression
│   ├── wait_model.pkl      ← Linear Regression
│   ├── rush_model.pkl      ← Decision Tree
│   ├── kmeans_model.pkl    ← K-Means
│   ├── combo_rules.pkl     ← Association Rules
│   └── summary.json        ← Model scores + dataset stats
├── templates/
│   └── index.html          ← Full frontend (single page app)
└── static/
    ├── css/style.css       ← Cute cafe aesthetic design
    └── js/main.js          ← All interactions + Chart.js
```

---

## 🎓 Project Details
**Project:** Customer Analytics System — SmartSip  
**Tech:** Python, Flask, scikit-learn, Pandas, Chart.js  
**Models:** Logistic Regression, Linear Regression, Decision Tree, K-Means, Apriori  
**Dataset:** 1,943 orders · 21 menu items · 16 features
