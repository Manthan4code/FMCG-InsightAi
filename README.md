# 📊 FMCG InsightAI
## Sales, Profit & Inventory Intelligence System

A professional data analytics and AI project built on 100,000 Indian FMCG retail transactions (2024).

---

## 🗂️ Project Structure

```
FMCG_InsightAI/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── README.md
├── data/
│   └── fmcg_data.csv               # 100,000 transaction records
├── pages/
│   ├── __init__.py
│   ├── sales_profit.py             # Part 1 – Sales & Profit Intelligence
│   ├── inventory_customer.py       # Part 2 – Inventory & Customer Intelligence
│   └── ai_insights.py              # Part 3 – AI Business Insights
└── notebooks/
    └── FMCG_InsightAI_Analysis.ipynb
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
cd FMCG_InsightAI
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📋 Dataset
| datset:'https://www.kaggle.com/datasets/arannayavadebnath/retail-fmcg-sales-dataset-2024?select=Indian+FMCG+Retail+Sales++Customer++Inventory+%282024%29.csv'
| Field | Detail |
|---|---|
| Source | Indian FMCG Retail Sales, Customer & Inventory (2024) |
| Rows | 100,000 transactions |
| Columns | 21 features |
| Date Range | 2024-01-01 → 2024-12-30 |
| Cities | Kolkata, Hyderabad, Chennai, Bengaluru, Delhi, Ahmedabad, Mumbai, Pune |
| Categories | Grocery, Snacks, Beverages, Dairy, Home Care, Personal Care, Vegetables, Fruits |
| Brands | Nestle, PepsiCo, ITC, Tata, Britannia, Amul, Parle, HUL |
| Channels | Online, Offline, Omnichannel |

---

## 🔑 Key Features

### 📈 Part 1 — Sales & Profit Intelligence
- KPI cards: Revenue, Units, Margin, Avg Margin %
- Monthly revenue & margin trend
- Revenue by category, brand, city, and channel
- Profitability comparison by brand and category
- Revenue vs Margin scatter (bubble chart)
- High-revenue & high-margin segment table
- Sidebar filters: City, Category, Brand, Channel, Store Format

### 📦 Part 2 — Inventory & Customer Intelligence
- Inventory status: Reorder Required vs Sufficient Stock
- Stock vs Reorder Level by brand
- Lead time distribution and brand comparison
- Reorder hotspots by city
- Customer loyalty, gender, and age-group analysis
- Channel preference and payment mode breakdown
- Revenue: loyal vs non-loyal customers

### 🤖 Part 3 — AI Business Insights

#### Isolation Forest (Anomaly Detection)
- Flags statistically unusual transactions (~3%)
- Features: Revenue, Units, Margin_%, Selling_Price, Cost_Price, Stock_On_Hand, Reorder_Level, Lead_Time_Days
- Visualises score distribution, revenue-margin scatter with anomaly overlay
- Anomaly breakdown by category and brand
- Top 25 most anomalous transactions table

#### K-Means Clustering (k = 4)
- Segments transactions into natural groups
- Segments named: Premium/High-Value, Mid-Tier Growth, Budget/High-Volume, Low-Activity Tail
- PCA 2-D projection with cluster labels
- Cluster profile table (avg revenue, margin, units, price, stock)
- Category mix per cluster

#### Business Findings
- Anomaly rate and business interpretation
- Margin behaviour differences between normal and anomalous transactions
- Segment-level revenue insights
- Inventory risk assessment
- Actionable next steps

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `pandas` | Data loading, cleaning, aggregation |
| `numpy` | Numerical operations, vectorised logic |
| `plotly` | Interactive charts (bar, line, scatter, pie, histogram) |
| `scikit-learn` | Isolation Forest, K-Means, PCA, StandardScaler |
| `matplotlib` | Supporting static plots (notebook) |
| `seaborn` | Supporting statistical plots (notebook) |

---

## 🧹 Data Cleaning Applied

| Issue | Action |
|---|---|
| 62 duplicate Invoice IDs | Dropped duplicates, kept first occurrence |
| 40,081 missing Customer_Age (40%) | Imputed with median age |
| 5,048 missing Customer_Gender (5%) | Filled as "Unknown" |
| Invoice_Date as string | Parsed to `datetime64` |
| Computed columns (Revenue, Cost, Margin, Margin_%) | Verified — all internally consistent (error < 1e-10) |

---

## 📌 Notes

- All financial values are in **Indian Rupees (₹)**
- Large numbers are formatted as K (thousands), L (lakhs), Cr (crores)
- Isolation Forest contamination is set to **3%** — adjust `IF_CONTAMINATION` in `pages/ai_insights.py` if needed
- K-Means uses **k = 4** clusters — adjust `N_CLUSTERS` in `pages/ai_insights.py` if needed
- Results are cached with `@st.cache_data` for fast re-renders

---

*Built as part of an IBM internship data analytics project.*
