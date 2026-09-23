"""
Part 1 – Sales & Profit Intelligence
Renders the full Sales & Profit dashboard page inside the Streamlit app.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── helpers ──────────────────────────────────────────────────────────────────

def fmt_inr(value: float) -> str:
    """Format a number as Indian Rupees with K/L/Cr suffix."""
    if value >= 1e7:
        return f"₹{value/1e7:,.2f} Cr"
    if value >= 1e5:
        return f"₹{value/1e5:,.2f} L"
    if value >= 1e3:
        return f"₹{value/1e3:,.1f} K"
    return f"₹{value:,.2f}"


def fmt_units(value: float) -> str:
    if value >= 1e6:
        return f"{value/1e6:,.2f} M"
    if value >= 1e3:
        return f"{value/1e3:,.1f} K"
    return f"{value:,.0f}"


def kpi_card(col, label: str, value: str, delta: str = "", icon: str = ""):
    col.markdown(
        f"""
        <div style="background:#1e1e2e;border-radius:12px;padding:18px 20px;
                    border-left:4px solid #7c5cd8;">
          <p style="margin:0;font-size:13px;color:#a0a8c0;font-weight:500;">{icon} {label}</p>
          <p style="margin:4px 0 0 0;font-size:24px;font-weight:700;color:#f0f2ff;">{value}</p>
          <p style="margin:2px 0 0 0;font-size:12px;color:#7c5cd8;">{delta}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


# ── main render function ──────────────────────────────────────────────────────

def render(df: pd.DataFrame):
    st.markdown("## 📈 Sales & Profit Intelligence")
    st.markdown(
        "<p style='color:#a0a8c0;margin-top:-10px;'>Full-year 2024 · Indian FMCG Retail</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔍 Filters")

        cities = ["All"] + sorted(df["City"].unique().tolist())
        sel_city = st.multiselect("City", cities[1:], placeholder="All cities")

        cats = ["All"] + sorted(df["Category"].unique().tolist())
        sel_cat = st.multiselect("Category", cats[1:], placeholder="All categories")

        brands = ["All"] + sorted(df["Brand"].unique().tolist())
        sel_brand = st.multiselect("Brand", brands[1:], placeholder="All brands")

        channels = ["All"] + sorted(df["Channel"].unique().tolist())
        sel_channel = st.multiselect("Channel", channels[1:], placeholder="All channels")

        formats = ["All"] + sorted(df["Store_Format"].unique().tolist())
        sel_format = st.multiselect("Store Format", formats[1:], placeholder="All formats")

    # apply filters
    fdf = df.copy()
    if sel_city:
        fdf = fdf[fdf["City"].isin(sel_city)]
    if sel_cat:
        fdf = fdf[fdf["Category"].isin(sel_cat)]
    if sel_brand:
        fdf = fdf[fdf["Brand"].isin(sel_brand)]
    if sel_channel:
        fdf = fdf[fdf["Channel"].isin(sel_channel)]
    if sel_format:
        fdf = fdf[fdf["Store_Format"].isin(sel_format)]

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_rev   = fdf["Revenue"].sum()
    total_units = fdf["Units"].sum()
    total_margin = fdf["Margin"].sum()
    avg_margin_pct = fdf["Margin_%"].mean() * 100
    total_txn   = len(fdf)

    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_card(c1, "Total Revenue",     fmt_inr(total_rev),    f"{total_txn:,} transactions", "💰")
    kpi_card(c2, "Total Units Sold",  fmt_units(total_units), f"Avg {fdf['Units'].mean():.1f} units/invoice", "📦")
    kpi_card(c3, "Total Margin",      fmt_inr(total_margin),  "", "📊")
    kpi_card(c4, "Avg Margin %",      f"{avg_margin_pct:.1f}%", "", "📐")
    kpi_card(c5, "Transactions",      f"{total_txn:,}", "", "🧾")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Monthly Revenue Trend + Revenue by Category ───────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### Monthly Revenue Trend")
        monthly = (
            fdf.groupby("Month_Label", sort=False)
            .agg(Revenue=("Revenue", "sum"), Margin=("Margin", "sum"))
            .reset_index()
        )
        # ensure correct month order
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly["Month_Label"] = pd.Categorical(
            monthly["Month_Label"], categories=month_order, ordered=True
        )
        monthly = monthly.sort_values("Month_Label")

        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=monthly["Month_Label"], y=monthly["Revenue"],
            name="Revenue", marker_color="#7c5cd8", opacity=0.85,
        ))
        fig_monthly.add_trace(go.Scatter(
            x=monthly["Month_Label"], y=monthly["Margin"],
            name="Margin", mode="lines+markers",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=6),
        ))
        fig_monthly.update_layout(
            **CHART_THEME,
            xaxis_title="Month", yaxis_title="₹",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
            height=320,
        )
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col_right:
        st.markdown("#### Revenue by Category")
        cat_rev = (
            fdf.groupby("Category")["Revenue"]
            .sum().sort_values(ascending=True).reset_index()
        )
        fig_cat = go.Figure(go.Bar(
            x=cat_rev["Revenue"], y=cat_rev["Category"],
            orientation="h",
            marker=dict(
                color=cat_rev["Revenue"],
                colorscale=[[0, "#3b3060"], [1, "#7c5cd8"]],
            ),
            text=[fmt_inr(v) for v in cat_rev["Revenue"]],
            textposition="outside",
        ))
        fig_cat.update_layout(
            **CHART_THEME,
            xaxis_title="Revenue (₹)", yaxis_title="",
            height=320,
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── Row 2: Top Brands by Revenue + Margin by Brand ───────────────────────
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown("#### Top Brands by Revenue")
        brand_rev = (
            fdf.groupby("Brand")["Revenue"]
            .sum().sort_values(ascending=False).reset_index()
        )
        fig_brand = px.bar(
            brand_rev, x="Brand", y="Revenue",
            color="Brand",
            color_discrete_sequence=CHART_THEME["colorway"],
            text=[fmt_inr(v) for v in brand_rev["Revenue"]],
        )
        fig_brand.update_traces(textposition="outside")
        fig_brand.update_layout(
            **CHART_THEME,
            xaxis_title="Brand", yaxis_title="Revenue (₹)",
            showlegend=False, height=320,
        )
        st.plotly_chart(fig_brand, use_container_width=True)

    with col_r2:
        st.markdown("#### Avg Margin % by Brand")
        brand_margin = (
            fdf.groupby("Brand")["Margin_%"]
            .mean().mul(100).sort_values(ascending=False).reset_index()
        )
        brand_margin.columns = ["Brand", "Avg_Margin_Pct"]
        fig_bm = px.bar(
            brand_margin, x="Brand", y="Avg_Margin_Pct",
            color="Avg_Margin_Pct",
            color_continuous_scale=["#3b3060", "#7c5cd8", "#10b981"],
            text=brand_margin["Avg_Margin_Pct"].map(lambda v: f"{v:.1f}%"),
        )
        fig_bm.update_traces(textposition="outside")
        fig_bm.update_layout(
            **CHART_THEME,
            xaxis_title="Brand", yaxis_title="Avg Margin %",
            showlegend=False, coloraxis_showscale=False, height=320,
        )
        st.plotly_chart(fig_bm, use_container_width=True)

    # ── Row 3: Revenue vs Margin scatter + Revenue by Channel ────────────────
    col_l3, col_r3 = st.columns([3, 2])

    with col_l3:
        st.markdown("#### Revenue vs Margin by Category")
        cat_summary = (
            fdf.groupby("Category")
            .agg(Revenue=("Revenue","sum"), Margin=("Margin","sum"),
                 Units=("Units","sum"))
            .reset_index()
        )
        fig_scatter = px.scatter(
            cat_summary, x="Revenue", y="Margin",
            size="Units", color="Category",
            color_discrete_sequence=CHART_THEME["colorway"],
            text="Category",
            size_max=55,
        )
        fig_scatter.update_traces(textposition="top center", marker_opacity=0.8)
        fig_scatter.update_layout(
            **CHART_THEME,
            xaxis_title="Total Revenue (₹)", yaxis_title="Total Margin (₹)",
            showlegend=False, height=360,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_r3:
        st.markdown("#### Revenue by Channel")
        ch_rev = fdf.groupby("Channel")["Revenue"].sum().reset_index()
        fig_ch = px.pie(
            ch_rev, names="Channel", values="Revenue",
            color_discrete_sequence=["#7c5cd8", "#3b82d4", "#10b981"],
            hole=0.45,
        )
        fig_ch.update_traces(
            textinfo="label+percent",
            textfont_size=13,
            marker=dict(line=dict(color="#0e0e1a", width=2)),
        )
        fig_ch.update_layout(
            **CHART_THEME,
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.05),
            height=360,
        )
        st.plotly_chart(fig_ch, use_container_width=True)

    # ── Row 4: Revenue by City heatmap-style + Margin by Category ────────────
    col_l4, col_r4 = st.columns(2)

    with col_l4:
        st.markdown("#### Revenue by City")
        city_rev = (
            fdf.groupby("City")["Revenue"]
            .sum().sort_values(ascending=False).reset_index()
        )
        fig_city = px.bar(
            city_rev, x="City", y="Revenue",
            color="Revenue",
            color_continuous_scale=["#1e1040", "#7c5cd8"],
            text=[fmt_inr(v) for v in city_rev["Revenue"]],
        )
        fig_city.update_traces(textposition="outside")
        fig_city.update_layout(
            **CHART_THEME,
            xaxis_title="City", yaxis_title="Revenue (₹)",
            coloraxis_showscale=False, height=320,
        )
        st.plotly_chart(fig_city, use_container_width=True)

    with col_r4:
        st.markdown("#### Profitability: Margin % by Category")
        cat_marg = (
            fdf.groupby("Category")["Margin_%"]
            .mean().mul(100).sort_values(ascending=False).reset_index()
        )
        cat_marg.columns = ["Category", "Avg_Margin_Pct"]
        fig_cm = px.bar(
            cat_marg, x="Avg_Margin_Pct", y="Category",
            orientation="h",
            color="Avg_Margin_Pct",
            color_continuous_scale=["#3b3060", "#10b981"],
            text=cat_marg["Avg_Margin_Pct"].map(lambda v: f"{v:.1f}%"),
        )
        fig_cm.update_traces(textposition="outside")
        fig_cm.update_layout(
            **CHART_THEME,
            xaxis_title="Avg Margin %", yaxis_title="",
            coloraxis_showscale=False, height=320,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Row 5: High-revenue & high-margin segment table ───────────────────────
    st.markdown("#### 🏆 High-Revenue & High-Margin Segments")
    seg = (
        fdf.groupby(["Category", "Brand"])
        .agg(
            Total_Revenue=("Revenue", "sum"),
            Total_Margin=("Margin", "sum"),
            Avg_Margin_Pct=("Margin_%", "mean"),
            Total_Units=("Units", "sum"),
        )
        .reset_index()
        .sort_values("Total_Revenue", ascending=False)
        .head(15)
    )
    seg["Total_Revenue"]    = seg["Total_Revenue"].map(fmt_inr)
    seg["Total_Margin"]     = seg["Total_Margin"].map(fmt_inr)
    seg["Avg_Margin_Pct"]   = seg["Avg_Margin_Pct"].map(lambda v: f"{v*100:.1f}%")
    seg["Total_Units"]      = seg["Total_Units"].map(lambda v: f"{v:,}")
    seg.index = range(1, len(seg) + 1)
    st.dataframe(seg, use_container_width=True)
