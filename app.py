import os, json, pickle
import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

BASE   = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

# ── Load all models ──────────────────────────────────────────────
cancel_model   = pickle.load(open(os.path.join(MODELS,"cancel_model.pkl"),  "rb"))
cancel_scaler  = pickle.load(open(os.path.join(MODELS,"cancel_scaler.pkl"), "rb"))
wait_model     = pickle.load(open(os.path.join(MODELS,"wait_model.pkl"),    "rb"))
wait_scaler    = pickle.load(open(os.path.join(MODELS,"wait_scaler.pkl"),   "rb"))
rush_model     = pickle.load(open(os.path.join(MODELS,"rush_model.pkl"),    "rb"))
rush_encoder   = pickle.load(open(os.path.join(MODELS,"rush_label_encoder.pkl"), "rb"))
kmeans_model   = pickle.load(open(os.path.join(MODELS,"kmeans_model.pkl"),  "rb"))
taste_scaler   = pickle.load(open(os.path.join(MODELS,"taste_scaler.pkl"),  "rb"))
cluster_labels = pickle.load(open(os.path.join(MODELS,"cluster_labels.pkl"),"rb"))
rush_enc       = pickle.load(open(os.path.join(MODELS,"rush_encoder.pkl"),  "rb"))
pay_enc        = pickle.load(open(os.path.join(MODELS,"pay_encoder.pkl"),   "rb"))
combo_rules    = pickle.load(open(os.path.join(MODELS,"combo_rules.pkl"),   "rb"))
summary_data   = json.load(open(os.path.join(MODELS,"summary.json"),        "r"))

# ── Static data ──────────────────────────────────────────────────
MENU = [
    {"name":"Cold Coffee",       "type":"Beverages",    "price":200, "emoji":"☕"},
    {"name":"Hot Coffee",        "type":"Beverages",    "price":150, "emoji":"🍵"},
    {"name":"Black Tea",         "type":"Beverages",    "price":100, "emoji":"🫖"},
    {"name":"Fresh Juice",       "type":"Beverages",    "price":180, "emoji":"🥤"},
    {"name":"Mango Shake",       "type":"Beverages",    "price":220, "emoji":"🥭"},
    {"name":"Green Tea",         "type":"Beverages",    "price":120, "emoji":"🍃"},
    {"name":"Chocolate Brownie", "type":"Dessert",      "price":180, "emoji":"🍫"},
    {"name":"Cheesecake Slice",  "type":"Dessert",      "price":200, "emoji":"🍰"},
    {"name":"Chocolate Cake",    "type":"Dessert",      "price":250, "emoji":"🎂"},
    {"name":"Fruit Salad",       "type":"Dessert",      "price":160, "emoji":"🍓"},
    {"name":"Pasta Alfredo",     "type":"Main Course",  "price":380, "emoji":"🍝"},
    {"name":"Chicken Biryani",   "type":"Main Course",  "price":400, "emoji":"🍛"},
    {"name":"Club Sandwich",     "type":"Snack",        "price":250, "emoji":"🥪"},
    {"name":"Aloo Paratha",      "type":"Snack",        "price":120, "emoji":"🫓"},
    {"name":"Garlic Bread",      "type":"Snack",        "price":130, "emoji":"🥖"},
    {"name":"Chicken Roll",      "type":"Fastfood",     "price":220, "emoji":"🫔"},
    {"name":"Zinger Burger",     "type":"Fastfood",     "price":320, "emoji":"🍔"},
    {"name":"Chicken Shawarma",  "type":"Fastfood",     "price":280, "emoji":"🌯"},
    {"name":"Fries",             "type":"Fastfood",     "price":150, "emoji":"🍟"},
    {"name":"Veggie Wrap",       "type":"Fastfood",     "price":200, "emoji":"🥙"},
    {"name":"Beef Burger",       "type":"Fastfood",     "price":350, "emoji":"🍔"},
]

INSIGHTS = [
    {"emoji":"🕛", "value":"12pm & 5–7pm",       "title":"Peak Rush Hours",      "desc":"Lunch and evening sessions see the highest footfall. Schedule extra staff during these windows to cut wait times.", "color":"rose"},
    {"emoji":"⏱️", "value":"4 – 13 min",          "title":"Wait Time Range",      "desc":"Wait time scales with busyness. Quiet hours average 4 min; super busy periods hit 13+ min.", "color":"mint"},
    {"emoji":"⭐", "value":"Cold Coffee",          "title":"Top Selling Item",     "desc":"Cold Coffee is your bestseller — always have it stocked. Pairs beautifully with Club Sandwich.", "color":"honey"},
    {"emoji":"🍰", "value":"~60% Customers",       "title":"Sweet Lovers Win",     "desc":"Majority of customers love beverages & desserts. Target them with morning pastry + drink bundles!", "color":"lav"},
    {"emoji":"🎁", "value":"Combos = More Sales",  "title":"Bundle & Upsell",      "desc":"Customers who get a combo deal spend 15–20% more per visit. Push bundle suggestions at checkout.", "color":"peach"},
    {"emoji":"📅", "value":"Monday",               "title":"Slowest Day",          "desc":"Mondays are your quietest — perfect for staff training, deep cleaning, and running flash promotions.", "color":"sage"},
    {"emoji":"📱", "value":"42% Online Payments",  "title":"Digital is Growing",   "desc":"Online payments trend upward and correlate with higher order values. Push digital wallet incentives.", "color":"caramel"},
    {"emoji":"👥", "value":"3 Customer Types",     "title":"Know Your Customers",  "desc":"Sweet Lovers (60%), Snack Lovers (25%), Spicy Lovers (15%). Tailor your offers to each group!", "color":"mocha"},
]

# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/summary")
def api_summary():
    return jsonify(summary_data)

@app.route("/api/menu")
def api_menu():
    return jsonify(MENU)

@app.route("/api/insights")
def api_insights():
    return jsonify(INSIGHTS)

@app.route("/api/combos/<item_name>")
def api_combos(item_name):
    combos = combo_rules.get(item_name, [])
    return jsonify({"item": item_name, "combos": combos})

# ── ML Predictions ───────────────────────────────────────────────

@app.route("/api/predict/cancel", methods=["POST"])
def predict_cancel():
    d = request.json
    hour   = int(d.get("hour", 12))
    amount = float(d.get("amount", 350))
    items  = int(d.get("items", 2))
    rush   = d.get("rush", "Medium")
    payment= d.get("payment", "Cash")

    try:
        rush_e = int(rush_enc.transform([rush])[0])
        pay_e  = int(pay_enc.transform([payment])[0])
    except:
        rush_e, pay_e = 1, 0

    X = np.array([[hour, amount, rush_e, pay_e, items]], dtype=float)
    X_s = cancel_scaler.transform(X)
    prob = float(cancel_model.predict_proba(X_s)[0][1])
    pct  = round(prob * 100)

    if pct < 35:   level, label = "low",    "Low Risk ✅"
    elif pct < 60: level, label = "medium", "Medium Risk ⚠️"
    else:          level, label = "high",   "High Risk 🚨"

    tips = {
        "low":    "All good! This order is very unlikely to be cancelled.",
        "medium": "A little risky. Send a wait-time update to keep them in the loop!",
        "high":   "High risk! Move fast — notify the customer now and speed up prep!"
    }
    return jsonify({"risk_pct": pct, "level": level, "label": label, "tip": tips[level]})


@app.route("/api/predict/wait", methods=["POST"])
def predict_wait():
    d = request.json
    hour    = int(d.get("hour", 12))
    items   = int(d.get("items", 2))
    rush    = d.get("rush", "Medium")
    orders  = int(d.get("orders_last_hr", 2))
    holiday = int(d.get("is_holiday", 0))

    try:
        rush_e = int(rush_enc.transform([rush])[0])
    except:
        rush_e = 1

    X = np.array([[hour, items, rush_e, orders, holiday]], dtype=float)
    X_s = wait_scaler.transform(X)
    minutes = float(wait_model.predict(X_s)[0])
    minutes = round(max(1.0, min(25.0, minutes)), 1)

    if minutes < 7:    status = "quick"
    elif minutes < 13: status = "moderate"
    else:              status = "long"

    tips = {
        "quick":    "Super fast! Great time to upsell a dessert or drink combo 🍰",
        "moderate": "Reasonable wait. Shoot a quick ETA message to keep them happy!",
        "long":     "Long wait! Offer something complimentary — a cookie goes a long way 🍪"
    }
    return jsonify({"minutes": minutes, "status": status, "tip": tips[status]})


@app.route("/api/predict/rush", methods=["POST"])
def predict_rush():
    d = request.json
    hour    = int(d.get("hour", 12))
    day     = int(d.get("day_of_week", 4))
    holiday = int(d.get("is_holiday", 0))
    orders  = int(d.get("orders_last_hr", 2))

    X = np.array([[hour, day, holiday, orders]])
    pred_enc = rush_model.predict(X)[0]
    try:
        level = rush_encoder.inverse_transform([pred_enc])[0]
    except:
        level = "Medium"

    cls_map = {"Low":"good", "Medium":"medium", "High":"bad", "Super High":"bad"}
    tips = {
        "Low":       "Nice and quiet 😌 — 1–2 staff is fine. Great time to restock!",
        "Medium":    "It's picking up! Get 3 staff ready and prep popular items.",
        "High":      "Full team on deck! 🔥 Pre-make top 5 items and keep the line moving!",
        "Super High":"Maximum rush! 🌋 All hands on deck — no breaks right now!"
    }
    labels = {"Low":"All Chill 😌", "Medium":"Getting Busy 😤", "High":"Rush Hour 🔥", "Super High":"MAXIMUM RUSH 🌋"}
    return jsonify({"level": level, "label": labels.get(level,"Medium"), "cls": cls_map.get(level,"medium"), "tip": tips.get(level,"")})


@app.route("/api/predict/taste", methods=["POST"])
def predict_taste():
    d = request.json
    bev  = int(d.get("beverages", 0))
    des  = int(d.get("desserts", 0))
    snk  = int(d.get("snacks", 0))
    main = int(d.get("main_course", 0))
    ff   = int(d.get("fastfood", 0))

    total = max(bev+des+snk+main+ff, 1)
    X = np.array([[bev, des, snk, main, ff]], dtype=float)
    X_s = taste_scaler.transform(X)
    cluster_id = int(kmeans_model.predict(X_s)[0])
    cluster = cluster_labels.get(cluster_id, "Sweet Lover")

    recs = {
        "Sweet Lover":  "🎁 Try our Coffee + Dessert loyalty bundle — made for you!",
        "Snack Lover":  "🥨 Happy hour snack deals are calling your name!",
        "Spicy Lover":  "🌶️ Biryani combo meals with fresh juice — perfect match!"
    }
    return jsonify({
        "cluster":        cluster,
        "cluster_id":     cluster_id,
        "sweet_pct":      round((bev+des)/total*100),
        "snack_pct":      round(snk/total*100),
        "spicy_pct":      round((main+ff)/total*100),
        "recommendation": recs.get(cluster, "")
    })


@app.route("/api/chart/hourly")
def chart_hourly():
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE,"data","smartsip.csv"))
    hourly = df.groupby("hour_of_day").size().reset_index(name="count")
    return jsonify(hourly.to_dict(orient="records"))

@app.route("/api/chart/revenue")
def chart_revenue():
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE,"data","smartsip.csv"))
    # simulate month from index (no date col) — use order_id ranges
    df["month"] = pd.cut(df["order_id"], bins=12, labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    rev = df.groupby("month", observed=True)["transaction_amount"].sum().reset_index()
    return jsonify(rev.rename(columns={"month":"label","transaction_amount":"value"}).to_dict(orient="records"))

@app.route("/api/chart/categories")
def chart_categories():
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE,"data","smartsip.csv"))
    cats = df["item_type"].value_counts(normalize=True).mul(100).round(1).reset_index()
    cats.columns = ["label","value"]
    return jsonify(cats.to_dict(orient="records"))

@app.route("/api/chart/topitems")
def chart_topitems():
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE,"data","smartsip.csv"))
    top = df["item_name"].value_counts().head(8).reset_index()
    top.columns = ["label","value"]
    return jsonify(top.to_dict(orient="records"))

@app.route("/api/chart/payment")
def chart_payment():
    import pandas as pd
    df = pd.read_csv(os.path.join(BASE,"data","smartsip.csv"))
    pay = df["transaction_type"].value_counts(normalize=True).mul(100).round(1).reset_index()
    pay.columns = ["label","value"]
    return jsonify(pay.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
