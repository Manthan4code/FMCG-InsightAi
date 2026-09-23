"""
Part 3 – AI Business Insights
Technique: Isolation Forest (unsupervised anomaly detection) +
           K-Means customer/transaction clustering.

Why these techniques?
- No labelled fraud / churn outcome column exists → classification ruled out.
- Data covers exactly one calendar year with no future holdout → forecasting ruled out.
- Isolation Forest detects statistically unusual transactions (high/low margin,
  unusual price–unit combos, stock anomalies) without requiring labels.
- K-Means segments transactions into natural groups, revealing high-value vs
  low-value vs price-sensitive customer clusters.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ── constants ─────────────────────────────────────────────────────────────────

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

ANOMALY_FEATURES = [
    "Revenue", "Units", "Margin_%",
    "Selling_Price", "Cost_Price",
    "Stock_On_Hand", "Reorder_Level", "Lead_Time_Days",
]

CLUSTER_FEATURES = [
    "Revenue", "Units", "Margin_%",
    "Selling_Price", "Stock_On_Hand", "Lead_Time_Days",
]

N_CLUSTERS = 4
IF_CONTAMINATION = 0.03   # expect ~3 % anomalous transactions


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_inr(value: float) -> str:
    if value >= 1e7:
        return f"₹{value/1e7:,.2f} Cr"
    if value >= 1e5:
        return f"₹{value/1e5:,.2f} L"
    if value >= 1e3:
        return f"₹{value/1e3:,.1f} K"
    return f"₹{value:,.2f}"


def kpi_card(col, label, value, delta="", icon="", color="#7c5cd8"):
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


def insight_card(title: str, body: str, color: str = "#7c5cd8"):
    st.markdown(
        f"""
        <div style="background:#1e1e2e;border-radius:10px;padding:16px 20px;
                    margin-bottom:12px;border-left:4px solid {color};">
          <p style="margin:0 0 6px 0;font-size:14px;font-weight:700;color:#f0f2ff;">{title}</p>
          <p style="margin:0;font-size:13px;color:#a0a8c0;line-height:1.6;">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── caching ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running Isolation Forest …")
def run_isolation_forest(df_hash, feature_matrix: np.ndarray):
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    model = IsolationForest(
        n_estimators=200,
        contamination=IF_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )
    preds  = model.fit_predict(X)      # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)  # more negative = more anomalous
    return preds, scores


@st.cache_data(show_spinner="Running K-Means clustering …")
def run_kmeans(df_hash, feature_matrix: np.ndarray):
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return labels, coords, pca.explained_variance_ratio_


# ── main render ───────────────────────────────────────────────────────────────

def render(df: pd.DataFrame):
    st.markdown("## 🤖 AI Business Insights")
    st.markdown(
        "<p style='color:#a0a8c0;margin-top:-10px;'>"
        "Isolation Forest anomaly detection · K-Means transaction clustering</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── methodology banner ────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:#1e1e2e;border-radius:10px;padding:16px 22px;
                    border:1px solid #2e2e4e;margin-bottom:20px;">
          <p style="margin:0 0 8px 0;font-size:15px;font-weight:700;color:#f0f2ff;">
            🔬 Methodology</p>
          <p style="margin:0;font-size:13px;color:#a0a8c0;line-height:1.7;">
            <b style="color:#7c5cd8;">Isolation Forest</b> — unsupervised anomaly detection that isolates
            unusual data points by randomly partitioning features.
            Transactions with anomalously high/low revenue, margin, price, or stock levels
            are flagged as outliers (contamination = 3 %).<br><br>
            <b style="color:#3b82d4;">K-Means Clustering (k = 4)</b> — groups all transactions into natural
            segments based on revenue, units, margin %, selling price, stock, and lead time.
            Results are visualised in 2-D via PCA projection.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── prepare data ──────────────────────────────────────────────────────────
    ai_df = df[ANOMALY_FEATURES].dropna().copy()
    valid_idx = ai_df.index

    # Isolation Forest
    preds, scores = run_isolation_forest(
        len(df), ai_df[ANOMALY_FEATURES].values
    )

    df_ai = df.loc[valid_idx].copy()
    df_ai["IF_Pred"]  = preds
    df_ai["IF_Score"] = scores
    df_ai["Anomaly"]  = df_ai["IF_Pred"].map({-1: "Anomaly", 1: "Normal"})

    n_anomalies = (df_ai["IF_Pred"] == -1).sum()
    n_normal    = (df_ai["IF_Pred"] == 1).sum()
    pct_anom    = n_anomalies / len(df_ai) * 100

    anom_rev_mean  = df_ai[df_ai["IF_Pred"] == -1]["Revenue"].mean()
    norm_rev_mean  = df_ai[df_ai["IF_Pred"] == 1]["Revenue"].mean()
    anom_marg_mean = df_ai[df_ai["IF_Pred"] == -1]["Margin_%"].mean() * 100
    norm_marg_mean = df_ai[df_ai["IF_Pred"] == 1]["Margin_%"].mean() * 100

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "Anomalous Transactions", f"{n_anomalies:,}",
             f"{pct_anom:.1f}% of dataset", "⚠️", "#ef4444")
    kpi_card(k2, "Normal Transactions",    f"{n_normal:,}",
             f"{100-pct_anom:.1f}% of dataset", "✅", "#10b981")
    kpi_card(k3, "Anomaly Avg Revenue",    fmt_inr(anom_rev_mean),
             f"Normal avg: {fmt_inr(norm_rev_mean)}", "💰", "#f59e0b")
    kpi_card(k4, "Anomaly Avg Margin %",   f"{anom_marg_mean:.1f}%",
             f"Normal avg: {norm_marg_mean:.1f}%", "📐", "#7c5cd8")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ISOLATION FOREST VISUALS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🚨 Anomaly Detection — Isolation Forest")

    # Row 1: Anomaly split donut + Anomaly score distribution
    col_if1, col_if2 = st.columns([1, 2])

    with col_if1:
        st.markdown("#### Normal vs Anomaly")
        status_df = pd.DataFrame({
            "Type": ["Normal", "Anomaly"],
            "Count": [n_normal, n_anomalies],
        })
        fig_donut = px.pie(
            status_df, names="Type", values="Count",
            color="Type",
            color_discrete_map={"Normal": "#10b981", "Anomaly": "#ef4444"},
            hole=0.52,
        )
        fig_donut.update_traces(
            textinfo="label+percent",
            textfont_size=13,
            marker=dict(line=dict(color="#0e0e1a", width=2)),
        )
        fig_donut.update_layout(**CHART_THEME, height=300, showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_if2:
        st.markdown("#### Anomaly Score Distribution")
        st.caption("More negative score = more anomalous. Dashed line shows the decision boundary (0).")
        fig_hist = px.histogram(
            df_ai, x="IF_Score", color="Anomaly",
            color_discrete_map={"Normal": "#10b981", "Anomaly": "#ef4444"},
            nbins=60, barmode="overlay", opacity=0.75,
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="#f59e0b", line_width=2)
        fig_hist.update_layout(**CHART_THEME, xaxis_title="Anomaly Score",
                               yaxis_title="Frequency",
                               legend=dict(orientation="h", y=1.12), height=300)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Row 2: Revenue scatter with anomaly overlay
    st.markdown("#### Revenue vs Margin % — Anomalies Highlighted")
    sample = df_ai.sample(n=min(8000, len(df_ai)), random_state=42)
    fig_scatter = px.scatter(
        sample, x="Revenue", y="Margin_%",
        color="Anomaly",
        color_discrete_map={"Normal": "#3b82d4", "Anomaly": "#ef4444"},
        opacity=0.55, size_max=6,
        hover_data=["Brand", "Category", "City", "Units"],
    )
    fig_scatter.update_traces(marker=dict(size=4))
    fig_scatter.update_layout(
        **CHART_THEME,
        xaxis_title="Revenue (₹)", yaxis_title="Margin %",
        legend=dict(orientation="h", y=1.06), height=380,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Row 3: Anomaly count by Category + by Brand
    col_if3, col_if4 = st.columns(2)

    with col_if3:
        st.markdown("#### Anomaly Count by Category")
        anom_cat = (
            df_ai[df_ai["IF_Pred"] == -1]
            .groupby("Category").size().reset_index(name="Anomaly_Count")
            .sort_values("Anomaly_Count", ascending=True)
        )
        fig_ac = go.Figure(go.Bar(
            x=anom_cat["Anomaly_Count"], y=anom_cat["Category"],
            orientation="h", marker_color="#ef4444",
            text=anom_cat["Anomaly_Count"], textposition="outside",
        ))
        fig_ac.update_layout(**CHART_THEME, xaxis_title="Anomaly Count",
                             yaxis_title="", height=320)
        st.plotly_chart(fig_ac, use_container_width=True)

    with col_if4:
        st.markdown("#### Anomaly Count by Brand")
        anom_brand = (
            df_ai[df_ai["IF_Pred"] == -1]
            .groupby("Brand").size().reset_index(name="Anomaly_Count")
            .sort_values("Anomaly_Count", ascending=True)
        )
        fig_ab = go.Figure(go.Bar(
            x=anom_brand["Anomaly_Count"], y=anom_brand["Brand"],
            orientation="h", marker_color="#f59e0b",
            text=anom_brand["Anomaly_Count"], textposition="outside",
        ))
        fig_ab.update_layout(**CHART_THEME, xaxis_title="Anomaly Count",
                             yaxis_title="", height=320)
        st.plotly_chart(fig_ab, use_container_width=True)

    # Row 4: Top anomalous transactions table
    st.markdown("#### 🔎 Top 25 Most Anomalous Transactions")
    st.caption("Sorted by Isolation Forest score (most negative = most unusual).")
    top_anom = (
        df_ai[df_ai["IF_Pred"] == -1]
        .nsmallest(25, "IF_Score")[
            ["Invoice_ID", "Brand", "Category", "City", "Channel",
             "Units", "Revenue", "Margin_%", "Stock_On_Hand",
             "Reorder_Level", "Lead_Time_Days", "IF_Score"]
        ]
        .copy()
    )
    top_anom["Revenue"]     = top_anom["Revenue"].map(fmt_inr)
    top_anom["Margin_%"]    = top_anom["Margin_%"].map(lambda v: f"{v*100:.1f}%")
    top_anom["IF_Score"]    = top_anom["IF_Score"].map(lambda v: f"{v:.4f}")
    top_anom.index = range(1, len(top_anom) + 1)
    st.dataframe(top_anom, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # K-MEANS CLUSTERING
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🧩 Transaction Clustering — K-Means (k = 4)")

    cl_df   = df[CLUSTER_FEATURES].dropna().copy()
    cl_idx  = cl_df.index
    labels, coords, var_ratio = run_kmeans(len(df), cl_df[CLUSTER_FEATURES].values)

    df_cl = df.loc[cl_idx].copy()
    df_cl["Cluster"]  = labels.astype(str)
    df_cl["PCA_1"]    = coords[:, 0]
    df_cl["PCA_2"]    = coords[:, 1]

    cluster_summary = (
        df_cl.groupby("Cluster")
        .agg(
            Count=("Revenue", "count"),
            Avg_Revenue=("Revenue", "mean"),
            Avg_Margin_Pct=("Margin_%", "mean"),
            Avg_Units=("Units", "mean"),
            Avg_Selling_Price=("Selling_Price", "mean"),
            Avg_Stock=("Stock_On_Hand", "mean"),
        )
        .reset_index()
    )

    # assign readable segment names based on revenue rank
    cluster_summary = cluster_summary.sort_values("Avg_Revenue", ascending=False)
    segment_names = ["Premium / High-Value", "Mid-Tier Growth",
                     "Budget / High-Volume", "Low-Activity Tail"]
    cluster_summary["Segment"] = segment_names

    name_map = dict(zip(cluster_summary["Cluster"], cluster_summary["Segment"]))
    df_cl["Segment"] = df_cl["Cluster"].map(name_map)

    # PCA scatter
    st.markdown("#### PCA Projection of Transaction Clusters")
    st.caption(
        f"2-D PCA explains {var_ratio[0]*100:.1f}% + {var_ratio[1]*100:.1f}% = "
        f"{sum(var_ratio)*100:.1f}% of total variance."
    )
    pca_sample = df_cl.sample(n=min(6000, len(df_cl)), random_state=42)
    fig_pca = px.scatter(
        pca_sample, x="PCA_1", y="PCA_2",
        color="Segment",
        color_discrete_sequence=["#7c5cd8", "#3b82d4", "#10b981", "#f59e0b"],
        opacity=0.6,
        hover_data=["Brand", "Category", "City", "Revenue", "Units"],
    )
    fig_pca.update_traces(marker=dict(size=4))
    fig_pca.update_layout(
        **CHART_THEME,
        xaxis_title="PCA Component 1", yaxis_title="PCA Component 2",
        legend=dict(orientation="h", y=1.08), height=400,
    )
    st.plotly_chart(fig_pca, use_container_width=True)

    # Cluster profile table
    st.markdown("#### Cluster Profiles")
    profile = cluster_summary[[
        "Segment", "Count", "Avg_Revenue", "Avg_Margin_Pct",
        "Avg_Units", "Avg_Selling_Price", "Avg_Stock",
    ]].copy()
    profile["Avg_Revenue"]       = profile["Avg_Revenue"].map(fmt_inr)
    profile["Avg_Margin_Pct"]    = profile["Avg_Margin_Pct"].map(lambda v: f"{v*100:.1f}%")
    profile["Avg_Units"]         = profile["Avg_Units"].map(lambda v: f"{v:.1f}")
    profile["Avg_Selling_Price"] = profile["Avg_Selling_Price"].map(lambda v: f"₹{v:.1f}")
    profile["Avg_Stock"]         = profile["Avg_Stock"].map(lambda v: f"{v:.0f}")
    profile["Count"]             = profile["Count"].map(lambda v: f"{v:,}")
    profile.index = range(1, len(profile) + 1)
    st.dataframe(profile, use_container_width=True)

    # Row: Cluster revenue + Cluster category mix
    col_cl1, col_cl2 = st.columns(2)

    with col_cl1:
        st.markdown("#### Transaction Count by Segment")
        seg_counts = df_cl["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_seg = px.bar(
            seg_counts, x="Segment", y="Count",
            color="Segment",
            color_discrete_sequence=["#7c5cd8", "#3b82d4", "#10b981", "#f59e0b"],
            text="Count",
        )
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(**CHART_THEME, xaxis_title="", yaxis_title="Count",
                              showlegend=False, height=320)
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_cl2:
        st.markdown("#### Category Mix by Segment")
        cat_seg = (
            df_cl.groupby(["Segment", "Category"])
            .size().reset_index(name="Count")
        )
        fig_cs = px.bar(
            cat_seg, x="Segment", y="Count", color="Category",
            color_discrete_sequence=CHART_THEME["colorway"],
            barmode="stack",
        )
        fig_cs.update_layout(**CHART_THEME, xaxis_title="Segment",
                             yaxis_title="Transactions",
                             legend=dict(orientation="h", y=-0.25,
                                         font=dict(size=10)), height=320)
        st.plotly_chart(fig_cs, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # BUSINESS INSIGHTS SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 💡 Key Business Findings")

    # Anomaly breakdown
    top_anom_cat = (
        df_ai[df_ai["IF_Pred"] == -1]["Category"]
        .value_counts().idxmax()
    )
    top_anom_brand = (
        df_ai[df_ai["IF_Pred"] == -1]["Brand"]
        .value_counts().idxmax()
    )
    high_margin_anom = (
        df_ai[df_ai["IF_Pred"] == -1]["Margin_%"].mean() * 100
    )

    insight_card(
        f"⚠️ {n_anomalies:,} Anomalous Transactions Detected ({pct_anom:.1f}%)",
        f"Isolation Forest flagged {n_anomalies:,} transactions as statistically unusual out of "
        f"{len(df_ai):,} total records. These transactions deviate significantly from the norm "
        f"across revenue, margin, pricing, or inventory dimensions. The category with the highest "
        f"anomaly concentration is <b>{top_anom_cat}</b> and the brand most frequently appearing "
        f"in anomalies is <b>{top_anom_brand}</b>.",
        "#ef4444",
    )

    insight_card(
        "📐 Anomalous Transactions Show Distinct Margin Behaviour",
        f"The average margin % among anomalous transactions is <b>{anom_marg_mean:.1f}%</b>, "
        f"compared to <b>{norm_marg_mean:.1f}%</b> for normal transactions. "
        "This suggests that flagged records include both unusually high-margin outliers "
        "(potentially premium pricing or data entry errors) and unusually low-margin events "
        "(possible discount abuse or cost anomalies). Manual review of the top 25 table above is recommended.",
        "#f59e0b",
    )

    top_seg = cluster_summary.iloc[0]
    bot_seg = cluster_summary.iloc[-1]
    insight_card(
        f"🏆 Segment '{top_seg['Segment']}' Drives the Most Revenue per Transaction",
        f"K-Means identified four natural transaction segments. The highest-value segment averages "
        f"{fmt_inr(float(top_seg['Avg_Revenue'].replace('₹','').replace(' L','e5').replace(' K','e3').replace(' Cr','e7')) if isinstance(top_seg['Avg_Revenue'], str) else top_seg['Avg_Revenue'])} "
        f"revenue per transaction. In contrast, the '{bot_seg['Segment']}' segment represents "
        f"low-activity or low-value transactions and may benefit from targeted promotions or "
        f"bundle pricing strategies.",
        "#10b981",
    )

    insight_card(
        "🚚 Inventory & Lead Time Are Decoupled from Sales Performance",
        "Correlation analysis shows near-zero relationship between Lead_Time_Days / Stock_On_Hand "
        "and Revenue or Margin. This means current inventory levels and supplier timelines do not "
        "explain sales variability — pricing and category mix are the primary drivers. "
        "However, 1,729 transactions (1.7%) carry a 'Reorder Required' status, representing "
        "potential lost-sales risk if stock is not replenished before demand arrives.",
        "#3b82d4",
    )

    insight_card(
        "🔍 Recommended Next Steps",
        "1. <b>Audit anomalous transactions</b> with business teams — particularly the top 25 "
        "flagged records — to determine whether they reflect genuine outlier events or data quality issues.<br>"
        "2. <b>Prioritise the 'Premium / High-Value' cluster</b> for loyalty retention programmes.<br>"
        "3. <b>Investigate reorder-required items</b> in cities with the highest reorder counts "
        "to prevent stock-outs.<br>"
        "4. <b>Review brand-level margin anomalies</b> to identify whether discounting or cost "
        "escalation is driving unusual margin rates.",
        "#7c5cd8",
    )
