import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 1943

MENU = [
    ("Cold Coffee",       "Beverages",    200),
    ("Hot Coffee",        "Beverages",    150),
    ("Black Tea",         "Beverages",    100),
    ("Fresh Juice",       "Beverages",    180),
    ("Chocolate Brownie", "Dessert",      180),
    ("Cheesecake Slice",  "Dessert",      200),
    ("Pasta Alfredo",     "Main Course",  380),
    ("Chicken Biryani",   "Main Course",  400),
    ("Club Sandwich",     "Snack",        250),
    ("Aloo Paratha",      "Snack",        120),
    ("Chicken Roll",      "Fastfood",     220),
    ("Zinger Burger",     "Fastfood",     320),
    ("Chicken Shawarma",  "Fastfood",     280),
    ("Fries",             "Fastfood",     150),
    ("Garlic Bread",      "Snack",        130),
    ("Veggie Wrap",       "Fastfood",     200),
    ("Mango Shake",       "Beverages",    220),
    ("Chocolate Cake",    "Dessert",      250),
    ("Beef Burger",       "Fastfood",     350),
    ("Fruit Salad",       "Dessert",      160),
    ("Green Tea",         "Beverages",    120),
]

ITEM_NAMES  = [m[0] for m in MENU]
ITEM_TYPES  = [m[1] for m in MENU]
ITEM_PRICES = [m[2] for m in MENU]

# Realistic weights: Cold Coffee & Beverages more popular
ITEM_WEIGHTS = [0.095, 0.080, 0.045, 0.075, 0.040, 0.035,
                0.050, 0.055, 0.065, 0.030, 0.075, 0.080,
                0.045, 0.060, 0.025, 0.025, 0.035, 0.025,
                0.030, 0.020, 0.020]
ITEM_WEIGHTS = np.array(ITEM_WEIGHTS) / sum(ITEM_WEIGHTS)

# --- Hour distribution (peaks at 12-13 and 17-19)
hour_probs = np.zeros(24)
for h in range(7, 23):
    if h in [12, 13]:   hour_probs[h] = 0.12
    elif h in [17,18,19]: hour_probs[h] = 0.10
    elif h in [8,9]:    hour_probs[h] = 0.06
    elif h in [10,11,14,15,16]: hour_probs[h] = 0.05
    else:               hour_probs[h] = 0.02
hour_probs = hour_probs / hour_probs.sum()

hours        = np.random.choice(range(24), size=N, p=hour_probs)
days_of_week = np.random.choice(range(7),  size=N, p=[0.08,0.12,0.14,0.15,0.18,0.18,0.15])
is_holiday   = np.random.choice([0,1], size=N, p=[0.88, 0.12])

items_idx  = np.random.choice(len(MENU), size=N, p=ITEM_WEIGHTS)
item_names = [ITEM_NAMES[i]  for i in items_idx]
item_types = [ITEM_TYPES[i]  for i in items_idx]
item_prices= [ITEM_PRICES[i] for i in items_idx]

quantity   = np.random.choice([1,2,3,4,5], size=N, p=[0.45,0.30,0.15,0.07,0.03])
transaction_amount = np.array(item_prices) * quantity + np.random.randint(-20, 40, N)
transaction_amount = np.clip(transaction_amount, 80, 1200).astype(int)

transaction_type = np.random.choice(["Cash","Online"], size=N, p=[0.58, 0.42])

# Orders in last 60 min (correlated with hour)
def orders_last_hr(h):
    if h in [12,13,17,18,19]: return np.random.randint(4, 9)
    elif h in [8,9,10,11,14,15,16]: return np.random.randint(1, 5)
    else: return np.random.randint(0, 3)
orders_60 = np.array([orders_last_hr(h) for h in hours])

# Rush level
def rush_level(o):
    if o <= 1: return "Low"
    elif o <= 3: return "Medium"
    elif o <= 6: return "High"
    return "Super High"
rush = np.array([rush_level(o) for o in orders_60])

# Wait time (linear relationship + noise)
rush_map = {"Low": 0, "Medium": 1, "High": 2, "Super High": 3}
wait_base = (np.array([rush_map[r] for r in rush]) * 3.5
             + quantity * 1.2
             + orders_60 * 0.85
             + np.random.normal(0, 0.8, N))
wait_base = np.clip(wait_base, 1.5, 20).round(1)

# Cancellation (logistic-like)
cancel_score = (
    (rush == "High").astype(int) * 2.5
    + (rush == "Super High").astype(int) * 4.0
    + (transaction_type == "Cash").astype(int) * 0.8
    + (quantity >= 4).astype(int) * 1.5
    + (transaction_amount < 300).astype(int) * 0.6
    + np.random.normal(0, 1, N)
)
cancel_prob = 1 / (1 + np.exp(-cancel_score + 2.5))
cancel_status = (np.random.random(N) < cancel_prob).astype(int)

# Time of sale buckets
def time_bucket(h):
    if 7 <= h < 12:  return "Morning"
    elif 12 <= h < 17: return "Afternoon"
    elif 17 <= h < 21: return "Evening"
    return "Night"
time_of_sale = [time_bucket(h) for h in hours]

# Customer IDs (500 unique customers, repeat visits)
customer_ids = np.random.randint(1001, 1501, N)

# Build DataFrame
df = pd.DataFrame({
    "order_id":            range(1, N+1),
    "customer_id":         customer_ids,
    "item_name":           item_names,
    "item_type":           item_types,
    "item_price":          item_prices,
    "quantity":            quantity,
    "transaction_amount":  transaction_amount,
    "transaction_type":    transaction_type,
    "hour_of_day":         hours,
    "day_of_week":         days_of_week,
    "time_of_sale":        time_of_sale,
    "is_holiday":          is_holiday,
    "orders_in_last_60_min": orders_60,
    "rush_level":          rush,
    "wait_time_mins":      wait_base,
    "cancel_status":       cancel_status,
})

out = os.path.join(os.path.dirname(__file__), "data", "smartsip.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"✅ Dataset saved: {out}  ({len(df)} rows)")
print(df.head())
print("\nCancel rate:", df.cancel_status.mean().round(3))
print("Rush distribution:\n", df.rush_level.value_counts())
