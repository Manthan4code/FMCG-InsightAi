"""
Part 2 – Inventory & Customer Intelligence
Renders the Inventory & Customer dashboard page inside the Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── shared helpers ────────────────────────────────────────────────────────────

def fmt_inr(value: float) -> str:
    if value >= 1e7:
        return f"₹{value/1e7:,.2f} Cr"
    if value >= 1e5:
        return f"₹{value/1e5:,.2f} L"
    if value >= 1e3:
        return f"₹{value/1e3:,.1f} K"
    return f"₹{value:,.2f}"


CHART_THEME = dict(
    paper_bgcolor="#0e0e1a",
    plot_bgcolor="#0e0e1a",
    font=dict(color="#c8cfe8", family="Segoe UI, sans-serif", size=12),
    margin=dict(t=44, b=36, l=12, r=12),
    xaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
    yaxis=dict(gridcolor="#1e1e2e", zerolinecolor="#1e1e2e"),
    colorway=["#7c5cd8", "#3b82d4", "#10b981", "#f59e0b", "#ef4444",
               "#06b6d4", "#ec4899", "#84cc16"],
)


def apply_theme(fig):
    fig.update_layout(**CHART_THEME)
    return fig


def kpi_card(col, label: str, value: str, delta: str = "", icon: str = "", color: str = "#7c5cd8"):
    col.markdown(
        f"""
        <div style="background:#1e1e2e;border-radius:12px;padding:18px 20px;
                    border-left:4px solid {color};">
          <p style="margin:0;font-size:13px;color:#a0a8c0;font-weight:500;">{icon} {label}</p>
          <p style="margin:4px 0 0 0;font-size:24px;font-weight:700;color:#f0f2ff;">{value}</p>
          <p style="margin:2px 0 0 0;font-size:12px;color:{color};">{delta}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── main render function ──────────────────────────────────────────────────────

def render(df: pd.DataFrame):
    st.markdown("## 📦 Inventory & Customer Intelligence")
    st.markdown(
        "<p style='color:#a0a8c0;margin-top:-10px;'>Stock health, supplier lead times, and customer behaviour</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 Filters")
        cities = sorted(df["City"].unique().tolist())
        sel_city = st.multiselect("City", cities, placeholder="All cities", key="inv_city")

        cats = sorted(df["Category"].unique().tolist())
        sel_cat = st.multiselect("Category", cats, placeholder="All categories", key="inv_cat")

        brands = sorted(df["Brand"].unique().tolist())
        sel_brand = st.multiselect("Brand", brands, placeholder="All brands", key="inv_brand")

    fdf = df.copy()
    if sel_city:
        fdf = fdf[fdf["City"].isin(sel_city)]
    if sel_cat:
        fdf = fdf[fdf["Category"].isin(sel_cat)]
    if sel_brand:
        fdf = fdf[fdf["Brand"].isin(sel_brand)]

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — INVENTORY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🏭 Inventory Analysis")

    reorder_req = (fdf["Inventory_Status"] == "Reorder Required").sum()
    sufficient  = (fdf["Inventory_Status"] == "Sufficient Stock").sum()
    total_rows  = len(fdf)
    avg_soh     = fdf["Stock_On_Hand"].mean()
    avg_lead    = fdf["Lead_Time_Days"].mean()

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "Reorder Required",  f"{reorder_req:,}",
             f"{reorder_req/total_rows*100:.1f}% of transactions", "⚠️", "#ef4444")
    kpi_card(k2, "Sufficient Stock",  f"{sufficient:,}",
             f"{sufficient/total_rows*100:.1f}% of transactions", "✅", "#10b981")
    kpi_card(k3, "Avg Stock on Hand", f"{avg_soh:,.0f} units", "", "📊", "#3b82d4")
    kpi_card(k4, "Avg Lead Time",     f"{avg_lead:.1f} days",  "", "🚚", "#f59e0b")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row A-1: Status donut + Status by Category
    col_a1, col_a2 = st.columns([1, 2])

    with col_a1:
        st.markdown("#### Inventory Status Split")
        status_counts = fdf["Inventory_Status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_donut = px.pie(
            status_counts, names="Status", values="Count",
            color="Status",
            color_discrete_map={
                "Reorder Required": "#ef4444",
                "Sufficient Stock": "#10b981",
            },
            hole=0.52,
        )
        fig_donut.update_traces(
            textinfo="label+percent",
            textfont_size=13,
            marker=dict(line=dict(color="#0e0e1a", width=2)),
        )
        fig_donut.update_layout(**CHART_THEME, height=300, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_a2:
        st.markdown("#### Inventory Status by Category")
        cat_status = (
            fdf.groupby(["Category", "Inventory_Status"])
            .size().reset_index(name="Count")
        )
        fig_cat_status = px.bar(
            cat_status, x="Category", y="Count",
            color="Inventory_Status",
            color_discrete_map={
                "Reorder Required": "#ef4444",
                "Sufficient Stock": "#10b981",
            },
            barmode="stack",
            text_auto=True,
        )
        fig_cat_status.update_layout(**CHART_THEME, xaxis_title="Category",
                                     yaxis_title="Transactions", height=300,
                                     legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_cat_status, use_container_width=True)

    # Row A-2: Stock vs Reorder Level by Brand + Lead Time distribution
    col_a3, col_a4 = st.columns(2)

    with col_a3:
        st.markdown("#### Avg Stock vs Reorder Level by Brand")
        brand_inv = (
            fdf.groupby("Brand")
            .agg(Avg_Stock=("Stock_On_Hand", "mean"),
                 Avg_Reorder=("Reorder_Level", "mean"))
            .reset_index()
            .sort_values("Avg_Stock", ascending=False)
        )
        fig_stock = go.Figure()
        fig_stock.add_trace(go.Bar(
            x=brand_inv["Brand"], y=brand_inv["Avg_Stock"],
            name="Avg Stock on Hand", marker_color="#3b82d4",
        ))
        fig_stock.add_trace(go.Scatter(
            x=brand_inv["Brand"], y=brand_inv["Avg_Reorder"],
            name="Avg Reorder Level", mode="lines+markers",
            line=dict(color="#ef4444", width=2.5, dash="dash"),
            marker=dict(size=7),
        ))
        fig_stock.update_layout(**CHART_THEME, xaxis_title="Brand",
                                yaxis_title="Units", height=320,
                                legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_stock, use_container_width=True)

    with col_a4:
        st.markdown("#### Lead Time Distribution (Days)")
        fig_lt = px.histogram(
            fdf, x="Lead_Time_Days", nbins=12,
            color_discrete_sequence=["#7c5cd8"],
        )
        fig_lt.update_layout(**CHART_THEME, xaxis_title="Lead Time (Days)",
                             yaxis_title="Frequency", height=320)
        st.plotly_chart(fig_lt, use_container_width=True)

    # Row A-3: Inventory status by City + Reorder by Brand
    col_a5, col_a6 = st.columns(2)

    with col_a5:
        st.markdown("#### Reorder Required by City")
        city_reorder = (
            fdf[fdf["Inventory_Status"] == "Reorder Required"]
            .groupby("City").size().reset_index(name="Reorder_Count")
            .sort_values("Reorder_Count", ascending=True)
        )
        fig_cr = go.Figure(go.Bar(
            x=city_reorder["Reorder_Count"], y=city_reorder["City"],
            orientation="h", marker_color="#ef4444",
            text=city_reorder["Reorder_Count"], textposition="outside",
        ))
        fig_cr.update_layout(**CHART_THEME, xaxis_title="Count",
                             yaxis_title="", height=320)
        st.plotly_chart(fig_cr, use_container_width=True)

    with col_a6:
        st.markdown("#### Avg Lead Time by Brand")
        brand_lt = (
            fdf.groupby("Brand")["Lead_Time_Days"]
            .mean().sort_values(ascending=False).reset_index()
        )
        fig_blt = px.bar(
            brand_lt, x="Brand", y="Lead_Time_Days",
            color="Lead_Time_Days",
            color_continuous_scale=["#1e1040", "#f59e0b"],
            text=brand_lt["Lead_Time_Days"].map(lambda v: f"{v:.1f}d"),
        )
        fig_blt.update_traces(textposition="outside")
        fig_blt.update_layout(**CHART_THEME, xaxis_title="Brand",
                              yaxis_title="Avg Lead Time (Days)",
                              coloraxis_showscale=False, height=320)
        st.plotly_chart(fig_blt, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — CUSTOMER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 👥 Customer Intelligence")

    loyal_rev    = fdf[fdf["Loyalty_Flag"] == 1]["Revenue"].sum()
    nonloyal_rev = fdf[fdf["Loyalty_Flag"] == 0]["Revenue"].sum()
    loyal_pct    = fdf["Loyalty_Flag"].mean() * 100
    known_age    = fdf["Customer_Age"].notna()
    avg_age      = fdf.loc[known_age, "Customer_Age"].mean()

    k5, k6, k7, k8 = st.columns(4)
    kpi_card(k5, "Loyal Customers",     f"{loyal_pct:.1f}%",
             "of all transactions", "🌟", "#f59e0b")
    kpi_card(k6, "Loyal Revenue",       fmt_inr(loyal_rev),   "", "💎", "#7c5cd8")
    kpi_card(k7, "Non-Loyal Revenue",   fmt_inr(nonloyal_rev), "", "👤", "#3b82d4")
    kpi_card(k8, "Avg Customer Age",    f"{avg_age:.1f} yrs",
             f"{known_age.sum():,} known / {(~known_age).sum():,} missing", "🎂", "#10b981")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row B-1: Loyalty revenue pie + Customer Gender
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("#### Revenue: Loyal vs Non-Loyal")
        loyalty_df = pd.DataFrame({
            "Status": ["Loyal", "Non-Loyal"],
            "Revenue": [loyal_rev, nonloyal_rev],
        })
        fig_loy = px.pie(
            loyalty_df, names="Status", values="Revenue",
            color="Status",
            color_discrete_map={"Loyal": "#f59e0b", "Non-Loyal": "#3b82d4"},
            hole=0.48,
        )
        fig_loy.update_traces(
            textinfo="label+percent",
            textfont_size=13,
            marker=dict(line=dict(color="#0e0e1a", width=2)),
        )
        fig_loy.update_layout(**CHART_THEME, height=300, showlegend=False)
        st.plotly_chart(fig_loy, use_container_width=True)

    with col_b2:
        st.markdown("#### Transactions by Customer Gender")
        gender_map = {"M": "Male", "F": "Female", "O": "Other"}
        gender_df = fdf.copy()
        gender_df["Gender"] = gender_df["Customer_Gender"].map(gender_map).fillna("Unknown")
        gender_counts = gender_df["Gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        fig_gen = px.bar(
            gender_counts, x="Gender", y="Count",
            color="Gender",
            color_discrete_sequence=["#7c5cd8", "#ec4899", "#3b82d4", "#a0a8c0"],
            text="Count",
        )
        fig_gen.update_traces(textposition="outside")
        fig_gen.update_layout(**CHART_THEME, xaxis_title="Gender",
                              yaxis_title="Transactions", showlegend=False,
                              height=300)
        st.plotly_chart(fig_gen, use_container_width=True)

    # Row B-2: Age group analysis + Loyalty by Category
    col_b3, col_b4 = st.columns(2)

    with col_b3:
        st.markdown("#### Revenue by Customer Age Group")
        age_df = fdf[fdf["Customer_Age"].notna()].copy()
        age_df["Age_Group"] = pd.cut(
            age_df["Customer_Age"],
            bins=[17, 25, 35, 45, 55, 65],
            labels=["18-25", "26-35", "36-45", "46-55", "56-64"],
        )
        age_rev = (
            age_df.groupby("Age_Group", observed=True)["Revenue"]
            .sum().reset_index()
        )
        age_rev.columns = ["Age_Group", "Revenue"]
        fig_age = px.bar(
            age_rev, x="Age_Group", y="Revenue",
            color="Age_Group",
            color_discrete_sequence=CHART_THEME["colorway"],
            text=[fmt_inr(v) for v in age_rev["Revenue"]],
        )
        fig_age.update_traces(textposition="outside")
        fig_age.update_layout(**CHART_THEME, xaxis_title="Age Group",
                              yaxis_title="Revenue (₹)", showlegend=False,
                              height=320)
        st.plotly_chart(fig_age, use_container_width=True)

    with col_b4:
        st.markdown("#### Loyal vs Non-Loyal Units by Category")
        loy_cat = (
            fdf.groupby(["Category", "Loyalty_Flag"])["Units"]
            .sum().reset_index()
        )
        loy_cat["Loyalty"] = loy_cat["Loyalty_Flag"].map({1: "Loyal", 0: "Non-Loyal"})
        fig_lc = px.bar(
            loy_cat, x="Category", y="Units", color="Loyalty",
            color_discrete_map={"Loyal": "#f59e0b", "Non-Loyal": "#3b82d4"},
            barmode="group",
        )
        fig_lc.update_layout(**CHART_THEME, xaxis_title="Category",
                             yaxis_title="Units Sold",
                             legend=dict(orientation="h", y=1.12), height=320)
        st.plotly_chart(fig_lc, use_container_width=True)

    # Row B-3: Channel preference + Payment mode
    col_b5, col_b6 = st.columns(2)

    with col_b5:
        st.markdown("#### Customer Channel Preference")
        ch_df = fdf.groupby("Channel").agg(
            Transactions=("Invoice_ID", "count"),
            Revenue=("Revenue", "sum"),
        ).reset_index().sort_values("Transactions", ascending=False)
        fig_ch = px.bar(
            ch_df, x="Channel", y="Transactions",
            color="Channel",
            color_discrete_sequence=["#7c5cd8", "#3b82d4", "#10b981"],
            text="Transactions",
        )
        fig_ch.update_traces(textposition="outside")
        fig_ch.update_layout(**CHART_THEME, xaxis_title="Channel",
                             yaxis_title="Transactions", showlegend=False,
                             height=320)
        st.plotly_chart(fig_ch, use_container_width=True)

    with col_b6:
        st.markdown("#### Payment Mode Breakdown")
        pay_df = fdf["Payment_Mode"].value_counts().reset_index()
        pay_df.columns = ["Payment_Mode", "Count"]
        fig_pay = px.pie(
            pay_df, names="Payment_Mode", values="Count",
            color_discrete_sequence=["#7c5cd8", "#3b82d4", "#10b981", "#f59e0b"],
            hole=0.42,
        )
        fig_pay.update_traces(
            textinfo="label+percent",
            textfont_size=12,
            marker=dict(line=dict(color="#0e0e1a", width=2)),
        )
        fig_pay.update_layout(**CHART_THEME, height=320, showlegend=False)
        st.plotly_chart(fig_pay, use_container_width=True)

    # ── Customer summary table ────────────────────────────────────────────────
    st.markdown("#### 📋 Customer Revenue Summary by City & Channel")
    cust_tbl = (
        fdf.groupby(["City", "Channel"])
        .agg(
            Transactions=("Invoice_ID", "count"),
            Total_Revenue=("Revenue", "sum"),
            Loyal_Count=("Loyalty_Flag", "sum"),
        )
        .reset_index()
        .sort_values("Total_Revenue", ascending=False)
        .head(20)
    )
    cust_tbl["Loyal_%"]       = (cust_tbl["Loyal_Count"] / cust_tbl["Transactions"] * 100).map(lambda v: f"{v:.1f}%")
    cust_tbl["Total_Revenue"] = cust_tbl["Total_Revenue"].map(fmt_inr)
    cust_tbl.index = range(1, len(cust_tbl) + 1)
    st.dataframe(cust_tbl, use_container_width=True)
