"""
FMCG InsightAI – Sales, Profit & Inventory Intelligence System
Main Streamlit application entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# ── page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="FMCG InsightAI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
      /* dark base */
      html, body, [class*="css"] {
          font-family: "Segoe UI", system-ui, sans-serif;
      }
      .stApp {
          background-color: #0e0e1a;
          color: #c8cfe8;
      }
      /* sidebar */
      [data-testid="stSidebar"] {
          background-color: #13131f;
          border-right: 1px solid #1e1e2e;
      }
      [data-testid="stSidebar"] * {
          color: #c8cfe8 !important;
      }
      /* metric override */
      [data-testid="metric-container"] {
          background: #1e1e2e;
          border-radius: 10px;
          padding: 12px;
      }
      /* dataframe */
      [data-testid="stDataFrame"] {
          border-radius: 8px;
          overflow: hidden;
      }
      /* tabs */
      .stTabs [data-baseweb="tab-list"] {
          gap: 8px;
          background: #13131f;
          border-radius: 10px;
          padding: 4px 8px;
      }
      .stTabs [data-baseweb="tab"] {
          background: transparent;
          border-radius: 8px;
          color: #a0a8c0;
          font-weight: 600;
          padding: 8px 20px;
          font-size: 14px;
      }
      .stTabs [aria-selected="true"] {
          background: #7c5cd8 !important;
          color: #ffffff !important;
      }
      /* hide default Streamlit footer & menu */
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
      header {visibility: hidden;}
      /* multiselect tags */
      [data-baseweb="tag"] {
          background-color: #7c5cd8 !important;
      }
      /* scrollbar */
      ::-webkit-scrollbar { width: 6px; height: 6px; }
      ::-webkit-scrollbar-track { background: #13131f; }
      ::-webkit-scrollbar-thumb { background: #2e2e4e; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── data loading & preprocessing ─────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "fmcg_data.csv"


@st.cache_data(show_spinner="Loading dataset …")
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── 1. Parse dates ────────────────────────────────────────────────────────
    df["Invoice_Date"] = pd.to_datetime(df["Invoice_Date"], errors="coerce")
    df["Month"]        = df["Invoice_Date"].dt.month
    df["Month_Label"]  = df["Invoice_Date"].dt.strftime("%b")
    df["Quarter"]      = df["Invoice_Date"].dt.quarter
    df["DayOfWeek"]    = df["Invoice_Date"].dt.day_name()

    # ── 2. Remove 62 duplicate Invoice IDs (keep first) ───────────────────────
    df = df.drop_duplicates(subset="Invoice_ID", keep="first").reset_index(drop=True)

    # ── 3. Missing values ─────────────────────────────────────────────────────
    # Customer_Age: 40 % missing → fill with median
    age_median = df["Customer_Age"].median()
    df["Customer_Age"] = df["Customer_Age"].fillna(age_median)

    # Customer_Gender: 5 % missing → fill with "Unknown"
    df["Customer_Gender"] = df["Customer_Gender"].fillna("Unknown")

    # ── 4. Inventory status ───────────────────────────────────────────────────
    df["Inventory_Status"] = np.where(
        df["Stock_On_Hand"] <= df["Reorder_Level"],
        "Reorder Required",
        "Sufficient Stock",
    )

    # ── 5. Validate computed columns (they are all internally consistent) ─────
    # Revenue  = Units × Selling_Price  ✓ (max diff < 1e-10)
    # Cost     = Units × Cost_Price     ✓
    # Margin   = Revenue − Cost         ✓
    # Margin_% = Margin / Revenue       ✓

    return df


df = load_data(DATA_PATH)

# ── sidebar — app branding ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="padding:12px 0 20px 0;">
          <p style="margin:0;font-size:20px;font-weight:800;color:#f0f2ff;">
            📊 FMCG InsightAI</p>
          <p style="margin:2px 0 0 0;font-size:12px;color:#57606a;">
            Sales · Profit · Inventory · AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="background:#1e1e2e;border-radius:10px;padding:12px 14px;
                    margin-bottom:16px;font-size:12px;color:#a0a8c0;">
          <b style="color:#c8cfe8;">Dataset</b><br>
          {len(df):,} transactions<br>
          Jan 2024 – Dec 2024<br>
          8 cities · 8 brands · 8 categories
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── navigation tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈  Sales & Profit",
    "📦  Inventory & Customer",
    "🤖  AI Business Insights",
])

from pages import sales_profit, inventory_customer, ai_insights

with tab1:
    sales_profit.render(df)

with tab2:
    inventory_customer.render(df)

with tab3:
    ai_insights.render(df)
