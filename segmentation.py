"""
=============================================================================
 DASHBOARD — Segmentation Clients E-Commerce | RFM + ML
 Analyse avancée RFM + K-Means + Simulation
=============================================================================
 Installation :
     pip install streamlit pandas scikit-learn plotly openpyxl requests \
                 geopandas pycountry-convert matplotlib seaborn reportlab
 Lancement :
     streamlit run dashboard_segmentation.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
from datetime import datetime, timedelta
import warnings
import io
import os
import json
import base64
from io import BytesIO
import calendar

# Visualisations avancées
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

# Charts internationaux
try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEO = True
except:
    HAS_GEO = False

import requests

warnings.filterwarnings("ignore")
plt.style.use("dark_background")

# Load background image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

bg_image = get_base64_image("img.jpg")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Segmentation Clients — RFM Dashboard Pro",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "decision_log" not in st.session_state:
    st.session_state.decision_log = []

# ─────────────────────────────────────────────────────────────────────────────
# STYLE CSS
# ─────────────────────────────────────────────────────────────────────────────
css = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=Space+Mono:wght@400;700&display=swap');

  html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}

  /* Fond général */
  .stApp {{ 
    background: linear-gradient(rgba(13, 15, 20, 0.85), rgba(13, 15, 20, 0.85)), url('data:image/jpeg;base64,{bg_image}');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #e8eaf0; 
  }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
    background: #13161f;
    border-right: 1px solid #1e2230;
  }}
  [data-testid="stSidebar"] .stMarkdown h2 {{
    color: #7c6af7;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }}

  /* Titres */
  h1 {{ color: #ffffff !important; font-weight: 700 !important; }}
  h2, h3 {{ color: #c9cbdb !important; }}

  /* Cartes KPI */
  .kpi-card {{
    background: linear-gradient(135deg, #1a1d2e 0%, #1e2235 100%);
    border: 1px solid #2a2d45;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s;
  }}
  .kpi-card:hover {{ transform: translateY(-2px); border-color: #7c6af7; }}
  .kpi-value {{
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #7c6af7;
    display: block;
  }}
  .kpi-label {{
    font-size: 0.8rem;
    color: #8b8fa8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
    display: block;
  }}
  .kpi-delta {{
    font-size: 0.75rem;
    color: #50e3a4;
    margin-top: 6px;
    display: block;
  }}

  /* Badge segment */
  .segment-badge {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
  }}

  /* Section header */
  .section-header {{
    border-left: 3px solid #7c6af7;
    padding-left: 12px;
    margin: 24px 0 16px 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    color: #9b9dbf;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }}

  /* Plotly charts dark */
  .js-plotly-plot {{ border-radius: 10px; }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ 
    background: transparent; 
    border-radius: 8px;
    border-bottom: 2px solid #1e2230;
    gap: 8px;
  }}
  .stTabs [data-baseweb="tab"] {{ 
    color: #8b8fa8;
    padding: 12px 16px;
    border-radius: 6px 6px 0 0;
    border: none;
    background: transparent;
    font-weight: 500;
    transition: all 0.2s ease;
  }}
  .stTabs [data-baseweb="tab"]:hover {{
    color: #c9cbdb;
    background: #1a1d2e;
  }}
  .stTabs [aria-selected="true"] {{ 
    color: #7c6af7 !important;
    background: #1a1d2e !important;
    border-bottom: 2px solid #7c6af7 !important;
  }}

  /* Bouton */
  .stButton > button {{
    background: linear-gradient(135deg, #7c6af7, #5b4de8);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    transition: opacity 0.2s;
  }}
  .stButton > button:hover {{ opacity: 0.85; }}

  /* Info box */
  .info-box {{
    background: #1a1d2e;
    border: 1px solid #2a2d45;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
  }}
  .info-box p {{ color: #c9cbdb; margin: 4px 0; font-size: 0.9rem; }}
  .info-box strong {{ color: #7c6af7; }}

  /* Scroll top button area */
  footer {{ display: none; }}

  /* Tooltip */
  .tooltip-icon {{
    display: inline-block;
    width: 16px;
    height: 16px;
    background: #7c6af7;
    border-radius: 50%;
    color: white;
    font-size: 12px;
    font-weight: bold;
    text-align: center;
    line-height: 16px;
    cursor: help;
    margin-left: 6px;
    vertical-align: middle;
  }}

  .tooltip-icon:hover {{
    background: #5b4de8;
  }}

  /* Custom dataframe styling */
  .dataframe-container {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #2a2d45;
  }}

  /* Churn badge */
  .churn-critical {{ background: #e87c7c; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
  .churn-high {{ background: #f7a76c; color: white; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}
  .churn-medium {{ background: #f7d76c; color: #0d0f14; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }}

  /* Animated KPI */
  @keyframes slideInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  .animated-kpi {{
    animation: slideInUp 0.6s ease-out forwards;
  }}
</style>
"""

st.markdown(f"""{css}""".format(bg_image=bg_image), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
SEGMENT_COLORS = [
    "#7c6af7",   # violet
    "#50e3a4",   # vert menthe
    "#f7a76c",   # orange doux
    "#e87c7c",   # rouge rosé
    "#5bc4f7",   # bleu ciel
    "#f7d76c",   # jaune or
]

SEGMENT_INFO = {
    "Champions": {
        "color": "#50e3a4",
        "emoji": "",
        "desc": "Acheteurs récents, fréquents, gros dépensiers. Ce sont vos meilleurs clients — récompensez-les.",
        "action": "Programme de fidélité VIP, accès anticipé aux nouveautés.",
    },
    "Clients Fidèles": {
        "color": "#7c6af7",
        "emoji": "",
        "desc": "Achètent régulièrement. Bon potentiel de montée en gamme.",
        "action": "Upsell, programme de points, offres exclusives membres.",
    },
    "Clients Prometteurs": {
        "color": "#5bc4f7",
        "emoji": "",
        "desc": "Clients récents avec une fréquence modérée. À fidéliser.",
        "action": "Email onboarding, remises de bienvenue, cross-sell.",
    },
    "À Risque": {
        "color": "#f7a76c",
        "emoji": "",
        "desc": "Autrefois actifs, mais n'ont pas acheté depuis longtemps.",
        "action": "Campagne win-back, sondage satisfaction, offre choc.",
    },
    "Clients Perdus": {
        "color": "#e87c7c",
        "emoji": "",
        "desc": "Très peu actifs, faible valeur. Difficiles à réactiver.",
        "action": "Email de réactivation de dernière chance ou désengager.",
    },
    "Occasionnels": {
        "color": "#f7d76c",
        "emoji": "",
        "desc": "Faible récence et fréquence. Acheteurs ponctuels.",
        "action": "Promotions saisonnières, retargeting publicitaire.",
    },
}

PLOTLY_TEMPLATE = {
    "layout": go.Layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#13161f",
        font=dict(color="#c9cbdb", family="DM Sans"),
        xaxis=dict(gridcolor="#1e2230", linecolor="#2a2d45"),
        yaxis=dict(gridcolor="#1e2230", linecolor="#2a2d45"),
    )
}


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT & NETTOYAGE DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_clean_data(file_obj=None):
    """Charge le dataset UCI Online Retail (xlsx ou csv) et nettoie."""
    if file_obj is not None:
        try:
            df = pd.read_excel(file_obj, engine="openpyxl")
        except Exception:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, encoding="ISO-8859-1")
    else:
        # Téléchargement des données réelles depuis UCI Repository
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            df = pd.read_excel(BytesIO(response.content), engine="openpyxl")
        except Exception as e:
            st.error(f"Erreur lors du téléchargement des données : {e}. Utilisation de données synthétiques.")
            df = generate_synthetic_data()

    # Normalisation colonnes
    col_map = {c.strip().lower(): c.strip() for c in df.columns}
    rename = {}
    for key, std in [
        ("invoiceno", "InvoiceNo"), ("stockcode", "StockCode"),
        ("description", "Description"), ("quantity", "Quantity"),
        ("invoicedate", "InvoiceDate"), ("unitprice", "UnitPrice"),
        ("customerid", "CustomerID"), ("country", "Country"),
    ]:
        for orig in col_map:
            if orig == key:
                rename[col_map[orig]] = std
    df.rename(columns=rename, inplace=True)

    # Nettoyage
    df.dropna(subset=["CustomerID", "InvoiceDate"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def generate_synthetic_data():
    """Génère un jeu de données synthétique représentatif."""
    np.random.seed(42)
    n = 25000
    countries = ["United Kingdom"] * 14 + ["Germany", "France", "Spain",
                 "Netherlands", "Belgium", "Switzerland", "Portugal",
                 "Australia", "USA", "Japan"]
    date_start = pd.Timestamp("2010-12-01")
    date_end = pd.Timestamp("2011-12-09")
    date_range = (date_end - date_start).days

    customer_ids = np.random.randint(10000, 18500, n)
    invoices = np.random.randint(500000, 600000, n)
    quantities = np.random.randint(1, 80, n)
    prices = np.round(np.random.exponential(scale=3.5, size=n) + 0.5, 2)
    days_offset = np.random.randint(0, date_range, n)
    dates = [date_start + pd.Timedelta(days=int(d)) for d in days_offset]
    country_list = np.random.choice(countries, n)

    df = pd.DataFrame({
        "InvoiceNo": invoices,
        "StockCode": np.random.randint(10000, 99999, n),
        "Description": "Product " + pd.Series(np.random.randint(1, 500, n)).astype(str),
        "Quantity": quantities,
        "InvoiceDate": dates,
        "UnitPrice": prices,
        "CustomerID": customer_ids,
        "Country": country_list,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL RFM
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()
    rfm = rfm[rfm["Monetary"] > 0]
    return rfm


# ─────────────────────────────────────────────────────────────────────────────
# K-MEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_kmeans(rfm: pd.DataFrame, n_clusters: int):
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    # Log-transform pour réduire la skewness
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = km.fit_predict(X_scaled)

    rfm = rfm.copy()
    rfm["Cluster"] = labels

    sil = silhouette_score(X_scaled, labels) if n_clusters > 1 else None
    inertia = km.inertia_
    return rfm, sil, inertia


@st.cache_data(show_spinner=False)
def elbow_data(rfm: pd.DataFrame, k_max: int = 10):
    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    X["Frequency"] = np.log1p(X["Frequency"])
    X["Monetary"] = np.log1p(X["Monetary"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertias, sils = [], []
    ks = list(range(2, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X_scaled, km.labels_))
    return ks, inertias, sils


# ─────────────────────────────────────────────────────────────────────────────
# NOMMAGE AUTOMATIQUE DES SEGMENTS
# ─────────────────────────────────────────────────────────────────────────────
def name_segments(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    stats = rfm_clustered.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    # Score composite : bas Recency = bon; haut Frequency & Monetary = bon
    stats["Score"] = (
        -stats["Recency"] / stats["Recency"].max()
        + stats["Frequency"] / stats["Frequency"].max()
        + stats["Monetary"] / stats["Monetary"].max()
    )
    stats_sorted = stats.sort_values("Score", ascending=False)

    all_labels = list(SEGMENT_INFO.keys())
    n = len(stats_sorted)
    label_map = {}
    for i, cluster_id in enumerate(stats_sorted.index):
        label_map[cluster_id] = all_labels[i % len(all_labels)]

    rfm_clustered = rfm_clustered.copy()
    rfm_clustered["Segment"] = rfm_clustered["Cluster"].map(label_map)
    return rfm_clustered


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS GRAPHIQUES
# ─────────────────────────────────────────────────────────────────────────────
def style_fig(fig):
    fig.update_layout(
        paper_bgcolor="#0d0f14",
        plot_bgcolor="#13161f",
        font=dict(color="#c9cbdb", family="DM Sans, sans-serif", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2d45"),
        margin=dict(t=40, l=10, r=10, b=10),
    )
    fig.update_xaxes(gridcolor="#1e2230", linecolor="#2a2d45", zeroline=False)
    fig.update_yaxes(gridcolor="#1e2230", linecolor="#2a2d45", zeroline=False)
    return fig


def tooltip_help(text: str):
    """Affiche une icône ? avec le texte d'aide au survol."""
    return f"""<span class="tooltip-icon" title="{text}">?</span>"""


@st.cache_data(show_spinner=False)
def compute_calendar_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Prépare les données pour une heatmap calendrier jour par jour."""
    df = df.copy()
    df["Date"] = df["InvoiceDate"].dt.date
    daily_sales = df.groupby("Date")["TotalPrice"].sum().reset_index()
    daily_sales.columns = ["Date", "Sales"]
    daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
    daily_sales["Week"] = daily_sales["Date"].dt.isocalendar().week
    daily_sales["DayOfWeek"] = daily_sales["Date"].dt.dayofweek
    daily_sales["Month"] = daily_sales["Date"].dt.month
    daily_sales["Year"] = daily_sales["Date"].dt.year
    return daily_sales


def plot_calendar_heatmap(daily_sales: pd.DataFrame):
    """Crée une heatmap style GitHub pour l'activité quotidienne."""
    year = daily_sales["Year"].min()
    months_data = []
    
    for month in range(1, 13):
        month_data = daily_sales[
            (daily_sales["Month"] == month) & (daily_sales["Year"] == year)
        ]
        if len(month_data) == 0:
            continue
            
        dates = pd.date_range(start=f"{year}-{month:02d}-01",
                              end=f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}")
        month_sales = pd.DataFrame({"Date": dates})
        month_sales = month_sales.merge(month_data, on="Date", how="left")
        month_sales["Sales"] = month_sales["Sales"].fillna(0)
        month_sales["Week"] = month_sales["Date"].dt.isocalendar().week
        month_sales["DayOfWeek"] = month_sales["Date"].dt.dayofweek
        
        months_data.append((calendar.month_abbr[month], month_sales))
    
    n_months = len(months_data)
    fig, axes = plt.subplots(n_months, 1, figsize=(16, 2.5 * n_months))
    
    if n_months == 1:
        axes = [axes]
    
    max_sales = daily_sales["Sales"].max()
    
    for ax, (month_name, month_data) in zip(axes, months_data):
        pivot = month_data.pivot_table(
            index="DayOfWeek", columns="Week", values="Sales", aggfunc="max"
        )
        
        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd", cbar_kws={"label": "CA (£)"},
            vmin=0, vmax=max_sales * 0.8, linewidths=0.5, linecolor="#1e2230"
        )
        ax.set_title(f"Activité d'achat — {month_name} {year}", fontsize=12, fontweight="bold", color="#c9cbdb")
        ax.set_ylabel("Jour de la semaine", fontsize=10, color="#c9cbdb")
        ax.set_xlabel("")
        days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        ax.set_yticklabels([days[i] for i in range(7)], rotation=0, fontsize=9)
    
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def compute_churn_risk(rfm_clustered: pd.DataFrame) -> pd.DataFrame:
    """Calcule un score de churn (urgence de réactivation) par client."""
    churn = rfm_clustered.copy()
    
    # Score de risque basé sur Recency, inversement sur Frequency/Monetary
    max_r = churn["Recency"].max()
    max_f = churn["Frequency"].max()
    max_m = churn["Monetary"].max()
    
    # Churn = high recency (inactif), low frequency & monetary (faible valeur)
    churn["Churn_Score"] = (
        (churn["Recency"] / max_r) * 0.5  # Inactivité = risque élevé
        - (churn["Frequency"] / max_f) * 0.25  # Fréquence réduit risque
        - (churn["Monetary"] / max_m) * 0.25   # Valeur réduit risque
    )
    churn["Churn_Score"] = ((churn["Churn_Score"] + 1) / 2 * 100).clip(0, 100)  # Normalize à 0-100
    
    # Catégories de risque
    churn["Churn_Category"] = pd.cut(
        churn["Churn_Score"],
        bins=[0, 30, 60, 100],
        labels=["Faible risque", "Risque moyen", "CRITIQUE"],
        include_lowest=True
    )
    
    return churn.sort_values("Churn_Score", ascending=False)


def predict_segment_from_rfm(rfm_input: dict, rfm_clustered: pd.DataFrame, scaler, km):
    """Prédit le segment pour des valeurs RFM données."""
    X_input = np.array([[
        rfm_input["Recency"],
        np.log1p(rfm_input["Frequency"]),
        np.log1p(rfm_input["Monetary"])
    ]])
    X_scaled = scaler.fit_transform(rfm_clustered[["Recency", "Frequency", "Monetary"]].copy())
    scaler_actual = StandardScaler()
    scaler_actual.fit(rfm_clustered[["Recency", "Frequency", "Monetary"]].copy())
    
    # Transform input avec le bon scaler
    X_input_log = np.array([[
        rfm_input["Recency"],
        np.log1p(rfm_input["Frequency"]),
        np.log1p(rfm_input["Monetary"])
    ]])
    X_input_scaled = scaler_actual.transform(X_input_log)
    pred_cluster = km.predict(X_input_scaled)[0]
    
    # Récupérer label du cluster
    cluster_map = rfm_clustered.groupby("Cluster")["Segment"].first().to_dict()
    return cluster_map.get(pred_cluster, "Clusters Inconnu")


def kpi_card(label: str, value: str, delta: str = ""):
    delta_html = f'<span class="kpi-delta">{delta}</span>' if delta else ""
    return f"""
    <div class="kpi-card">
      <span class="kpi-value">{value}</span>
      <span class="kpi-label">{label}</span>
      {delta_html}
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## RFM Dashboard — Segmentation Clients")
    st.markdown("---")
    
    # Navigation dropdown
    st.markdown("### Navigation")
    current_page = st.selectbox(
        "Sélectionner une page",
        options=[
            "Vue Globale",
            "Analyse Descriptive",
            "Segmentation RFM",
            "Interprétation Métier",
            "Fiche Client",
            "Comparateur",
            "Simulateur What-If",
            "Clients à Risque",
            "Export & Décisions",
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Source de données")

    uploaded_file = st.file_uploader(
        "Importer Online Retail (.xlsx / .csv)",
        type=["xlsx", "xls", "csv"],
        help="Dataset UCI : https://archive.ics.uci.edu/ml/datasets/online+retail",
    )

    use_demo = st.checkbox("Utiliser les données de démonstration", value=True)

    st.markdown("---")
    st.markdown("### Paramètres de segmentation")
    n_clusters = st.slider("Nombre de segments (k)", min_value=2, max_value=8, value=4, step=1)
    show_elbow = st.checkbox("Afficher la méthode du coude", value=False)

    st.markdown("---")
    st.markdown("### Filtres")

    # Ces filtres seront remplis après chargement des données
    filter_placeholder = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Chargement et nettoyage des données…"):
    if uploaded_file is not None and not use_demo:
        df_raw = load_and_clean_data(uploaded_file)
    else:
        df_raw = load_and_clean_data(None)

# ─────────────────────────────────────────────────────────────────────────────
# FILTRES DYNAMIQUES (sidebar)
# ─────────────────────────────────────────────────────────────────────────────
with filter_placeholder.container():
    countries = sorted(df_raw["Country"].dropna().unique())
    selected_countries = st.multiselect(
        "Pays", countries, default=["United Kingdom"],
        help="Sélectionner un ou plusieurs pays"
    )

    min_date = df_raw["InvoiceDate"].min().date()
    max_date = df_raw["InvoiceDate"].max().date()
    date_range_sel = st.date_input(
        "Période", value=(min_date, max_date),
        min_value=min_date, max_value=max_date
    )

# Appliquer filtres
df = df_raw.copy()
if selected_countries:
    df = df[df["Country"].isin(selected_countries)]
if len(date_range_sel) == 2:
    d0, d1 = date_range_sel
    df = df[(df["InvoiceDate"].dt.date >= d0) & (df["InvoiceDate"].dt.date <= d1)]

# Vérification
if df.empty:
    st.error("Aucune donnée pour les filtres sélectionnés. Élargissez la sélection.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CALCUL RFM + CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
rfm = compute_rfm(df)
rfm_clustered, sil_score, inertia = run_kmeans(rfm, n_clusters)
rfm_clustered = name_segments(rfm_clustered)

seg_colors_map = {
    seg: SEGMENT_INFO.get(seg, {}).get("color", SEGMENT_COLORS[i % len(SEGMENT_COLORS)])
    for i, seg in enumerate(rfm_clustered["Segment"].unique())
}

# ─────────────────────────────────────────────────────────────────────────────
# EN-TÊTE PROFESSIONNEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background: linear-gradient(135deg, #0d0f14 0%, #13161f 100%); padding: 32px 24px; border-radius: 12px; border-left: 4px solid #7c6af7; margin-bottom: 24px;">
  <div style="display:flex; align-items:flex-start; gap:20px;">
    <div style="font-size:3.5rem;">�</div>
    <div style="flex: 1;">
      <h1 style="margin:0 0 8px 0; font-size:2.2rem; font-weight:800; color:#ffffff; letter-spacing:-0.5px;">
        Segmentation Clients E-Commerce
      </h1>
      <p style="margin:0 0 12px 0; color:#8b8fa8; font-size:0.95rem; line-height:1.5;">
        Analyse avancée RFM + Machine Learning &nbsp;·&nbsp; Intelligence client en temps réel &nbsp;·&nbsp; Dashboard décisionnel
      </p>
      <div style="display:flex; gap:20px; margin-top:12px;">
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="color:#50e3a4; font-weight:600;">✓</span>
          <span style="color:#c9cbdb; font-size:0.85rem;">Segmentation automatique</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="color:#50e3a4; font-weight:600;">✓</span>
          <span style="color:#c9cbdb; font-size:0.85rem;">Prédiction comportement</span>
        </div>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="color:#50e3a4; font-weight:600;">✓</span>
          <span style="color:#c9cbdb; font-size:0.85rem;">Export & orchestration</span>
        </div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ONGLETS PRINCIPAUX — NAVIGATION STRUCTURÉE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 16px;">
  <div style="display:flex; gap:4px; justify-content:space-between; margin-bottom:8px;">
    <div style="font-size:0.8rem; color:#8b8fa8; text-transform:uppercase; letter-spacing:0.1em; font-weight:600;">Navigation</div>
  </div>
</div>
""", unsafe_allow_html=True)

# La variable current_page est déjà définie dans la sidebar

# ══════════════════════════════════════════════════════════════════════════════
# CONTENU DES PAGES
# ══════════════════════════════════════════════════════════════════════════════
if current_page == "Vue Globale":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
        <div>
            <h2 style="margin:0 0 8px 0; color:#ffffff;">Vue Globale du Portefeuille</h2>
            <p style="margin:0; color:#8b8fa8; font-size:0.9rem;">
                Aperçu financier et géographique · Indicateurs clés · Tendances d'activité
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <div style="font-size:1.2rem;">01 — Indicateurs Clés de Performance</div>
        <span class="tooltip-icon" title="Vue d'ensemble des métriques principales du portefeuille clients sur la période sélectionnée">?</span>
    </div>
    """, unsafe_allow_html=True)

    total_ca = df["TotalPrice"].sum()
    nb_clients = df["CustomerID"].nunique()
    nb_commandes = df["InvoiceNo"].nunique()
    panier_moyen = total_ca / nb_commandes if nb_commandes > 0 else 0
    ca_client_moyen = total_ca / nb_clients if nb_clients > 0 else 0

    # KPI avec animations
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_items = [
        (c1, "Clients", f"{nb_clients:,}", "parmi vos contacts"),
        (c2, "CA Total", f"£{total_ca:,.0f}", "sur la période"),
        (c3, "Commandes", f"{nb_commandes:,}", "factures nettes"),
        (c4, "Panier", f"£{panier_moyen:.1f}", "montant moyen"),
        (c5, "CA/Client", f"£{ca_client_moyen:.0f}", "valeur moyenne"),
    ]
    
    for col, title, value, hint in kpi_items:
        with col:
            st.markdown(f"""
            <div class="kpi-card animated-kpi">
              <span class="kpi-value">{value}</span>
              <span class="kpi-label">{title}</span>
              <span class="kpi-delta">{hint}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Heatmap calendrier
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <div style="font-size:1.2rem;">Heatmap d'Activité (Achat Jour par Jour)</div>
        <span class="tooltip-icon" title="Visualisation type GitHub : montre l'intensité des achats pour chaque jour. Plus la couleur est rouge/verte, plus l'activité était intense.">?</span>
    </div>
    """, unsafe_allow_html=True)
    
    daily_sales = compute_calendar_heatmap(df)
    cal_fig = plot_calendar_heatmap(daily_sales)
    st.pyplot(cal_fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <div style="font-size:1.2rem;">02 — Évolution du Chiffre d'Affaires</div>
        <span class="tooltip-icon" title="Tendance mensuelle en gris, lissée sur 3 mois en orange pour identifier les saisons et cycles d'achat">?</span>
    </div>
    """, unsafe_allow_html=True)

    df_monthly = df.copy()
    df_monthly["Month"] = df_monthly["InvoiceDate"].dt.to_period("M").astype(str)
    ca_monthly = df_monthly.groupby("Month")["TotalPrice"].sum().reset_index()
    ca_monthly.columns = ["Mois", "CA"]

    fig_ca = go.Figure()
    fig_ca.add_trace(go.Bar(
        x=ca_monthly["Mois"], y=ca_monthly["CA"],
        marker=dict(
            color=ca_monthly["CA"],
            colorscale=[[0, "#2a1f6e"], [0.5, "#7c6af7"], [1, "#50e3a4"]],
            showscale=False,
        ),
        hovertemplate="<b>%{x}</b><br>CA: £%{y:,.0f}<extra></extra>",
    ))
    fig_ca.add_trace(go.Scatter(
        x=ca_monthly["Mois"], y=ca_monthly["CA"].rolling(3, min_periods=1).mean(),
        mode="lines", name="Tendance (moy. 3 mois)",
        line=dict(color="#f7a76c", width=2, dash="dash"),
    ))
    fig_ca.update_layout(
        showlegend=True, height=340, xaxis_tickangle=-30,
    )
    style_fig(fig_ca)
    st.plotly_chart(fig_ca, use_container_width=True)

    # Carte choroplèthe mondiale
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <div style="font-size:1.2rem;">Carte Mondiale : CA par Pays</div>
        <span class="tooltip-icon" title="Visualisation géographique du chiffre d'affaires. Les pays plus foncés ont un CA plus élevé.">?</span>
    </div>
    """, unsafe_allow_html=True)
    
    ca_country = df.groupby("Country")["TotalPrice"].sum().reset_index()
    ca_country.columns = ["Country", "CA"]
    ca_country = ca_country.sort_values("CA", ascending=False)
    
    # Mapping ISO pour la choroplèthe
    country_iso_map = {
        "United Kingdom": "GBR",
        "Netherlands": "NLD",
        "Germany": "DEU",
        "France": "FRA",
        "Spain": "ESP",
        "Belgium": "BEL",
        "Switzerland": "CHE",
        "Australia": "AUS",
        "USA": "USA",
        "Japan": "JPN",
        "Portugal": "PRT",
        "Sweden": "SWE",
        "Italy": "ITA",
        "Poland": "POL",
        "Norway": "NOR",
    }
    
    ca_country["ISO"] = ca_country["Country"].map(country_iso_map)
    ca_country = ca_country.dropna(subset=["ISO"])
    
    if len(ca_country) > 0:
        fig_choropleth = px.choropleth(
            ca_country,
            locations="ISO",
            color="CA",
            color_continuous_scale="YlGnBu",
            hover_name="Country",
            hover_data={"CA": ":.0f", "ISO": False},
            labels={"CA": "CA (£)"},
            projection="natural earth"
        )
        fig_choropleth.update_layout(
            geo=dict(bgcolor="#0d0f14", showland=True, landcolor="#13161f"),
            height=400
        )
        style_fig(fig_choropleth)
        st.plotly_chart(fig_choropleth, use_container_width=True)
    
    # Top pays en tableau compact
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <div style="font-size:1.1rem;">Top 10 Pays</div>
    </div>
    """, unsafe_allow_html=True)
    
    top_countries = ca_country.nlargest(10, "CA")[["Country", "CA"]].copy()
    top_countries["CA"] = top_countries["CA"].apply(lambda x: f"£{x:,.0f}")
    top_countries.columns = ["Pays", "CA"]
    
    col_t1, col_t2, col_t3 = st.columns(3)
    for i in range(3):
        if i < len(top_countries):
            with [col_t1, col_t2, col_t3][i]:
                row = top_countries.iloc[i]
                st.metric(row["Pays"], row["CA"])

    st.markdown("<br>", unsafe_allow_html=True)
    col_map, col_top = st.columns([1.5, 1.5])

    with col_map:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
            <div style="font-size:1.2rem;">03 — Articles Populaires</div>
            <span class="tooltip-icon" title="8 produits les plus vendus en valeur (CA)">?</span>
        </div>
        """, unsafe_allow_html=True)
        top_products = (df.groupby("Description")["TotalPrice"]
                        .sum().reset_index()
                        .sort_values("TotalPrice", ascending=False)
                        .head(8))
        top_products["Description"] = top_products["Description"].str[:35]
        fig_bar_prod = px.bar(
            top_products, x="Description", y="TotalPrice",
            color="TotalPrice",
            color_continuous_scale=["#2a1f6e", "#7c6af7", "#50e3a4"],
            labels={"TotalPrice": "CA (£)", "Description": ""},
        )
        fig_bar_prod.update_layout(height=340, showlegend=False, xaxis_tickangle=-45)
        style_fig(fig_bar_prod)
        st.plotly_chart(fig_bar_prod, use_container_width=True)

    with col_top:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
            <div style="font-size:1.1rem;">04 — Mix Produits</div>
            <span class="tooltip-icon" title="Distribution de la valeur : parts de CA par catégorie de produit">?</span>
        </div>
        """, unsafe_allow_html=True)
        top_products = (df.groupby("Description")["TotalPrice"]
                        .sum().reset_index()
                        .sort_values("TotalPrice", ascending=False)
                        .head(8))
        top_products["Description"] = top_products["Description"].str[:30]
        fig_pie = px.pie(
            top_products, values="TotalPrice", names="Description",
            hole=0.45,
            color_discrete_sequence=SEGMENT_COLORS,
        )
        fig_pie.update_traces(textinfo="percent", hovertemplate="%{label}<br>£%{value:,.0f}")
        fig_pie.update_layout(height=380, showlegend=True,
                               legend=dict(font=dict(size=9)))
        style_fig(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSE DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Analyse Descriptive":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
        <div>
            <h2 style="margin:0 0 8px 0; color:#ffffff;">Analyse Descriptive</h2>
            <p style="margin:0; color:#8b8fa8; font-size:0.9rem;">
                Distribution RFM · Comportements d'achat · Patterns saisonniers
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">05 — Distribution des Variables RFM</div>',
                unsafe_allow_html=True)

    c_r, c_f, c_m = st.columns(3)

    for col_st, var, title, color in zip(
        [c_r, c_f, c_m],
        ["Recency", "Frequency", "Monetary"],
        ["Récence (jours)", "Fréquence (commandes)", "Montant total (£)"],
        ["#7c6af7", "#50e3a4", "#f7a76c"],
    ):
        data = rfm[var]
        if var == "Monetary":
            data = data[data < data.quantile(0.99)]  # retirer outliers extrêmes
        fig = go.Figure(go.Histogram(
            x=data, nbinsx=40,
            marker_color=color,
            opacity=0.85,
        ))
        fig.update_layout(title=title, height=250, margin=dict(t=35, l=5, r=5, b=5))
        style_fig(fig)
        col_st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">06 — Fréquence d\'Achat</div>',
                unsafe_allow_html=True)

    col_freq, col_dow = st.columns(2)

    with col_freq:
        freq_bins = pd.cut(rfm["Frequency"], bins=[0, 1, 2, 5, 10, 20, 9999],
                           labels=["1", "2", "3-5", "6-10", "11-20", "20+"])
        freq_dist = freq_bins.value_counts().sort_index().reset_index()
        freq_dist.columns = ["Nb_Commandes", "Clients"]
        fig_freq = px.bar(
            freq_dist, x="Nb_Commandes", y="Clients",
            color_discrete_sequence=["#7c6af7"],
            labels={"Nb_Commandes": "Nombre de commandes", "Clients": "Nb clients"},
        )
        fig_freq.update_layout(title="Distribution de la fréquence d'achat", height=300)
        style_fig(fig_freq)
        st.plotly_chart(fig_freq, use_container_width=True)

    with col_dow:
        df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        fr_labels = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        dow = df.groupby("DayOfWeek")["TotalPrice"].sum().reindex(order).fillna(0)
        fig_dow = go.Figure(go.Bar(
            x=fr_labels, y=dow.values,
            marker=dict(color=dow.values,
                        colorscale=["#2a1f6e", "#7c6af7", "#50e3a4"],
                        showscale=False),
        ))
        fig_dow.update_layout(title="CA par jour de la semaine", height=300)
        style_fig(fig_dow)
        st.plotly_chart(fig_dow, use_container_width=True)

    st.markdown('<div class="section-header">07 — Distribution des Montants</div>',
                unsafe_allow_html=True)

    col_box, col_scatter = st.columns(2)

    with col_box:
        rfm_plot = rfm.copy()
        rfm_plot["Monetary_clip"] = rfm_plot["Monetary"].clip(upper=rfm_plot["Monetary"].quantile(0.95))
        bins_m = pd.qcut(rfm_plot["Monetary_clip"], q=5,
                         labels=["Très faible", "Faible", "Moyen", "Élevé", "Très élevé"],
                         duplicates="drop")
        spend_dist = bins_m.value_counts().reset_index()
        spend_dist.columns = ["Tranche", "Clients"]
        fig_spend = px.bar(
            spend_dist, x="Tranche", y="Clients",
            color="Tranche",
            color_discrete_sequence=SEGMENT_COLORS,
        )
        fig_spend.update_layout(title="Distribution des montants dépensés (quintiles)",
                                 height=300, showlegend=False)
        style_fig(fig_spend)
        st.plotly_chart(fig_spend, use_container_width=True)

    with col_scatter:
        rfm_sample = rfm.sample(min(1500, len(rfm)), random_state=1)
        fig_scat = px.scatter(
            rfm_sample, x="Frequency", y="Monetary",
            color="Recency", size_max=10,
            color_continuous_scale=["#50e3a4", "#7c6af7", "#e87c7c"],
            opacity=0.6,
            labels={"Recency": "Récence", "Frequency": "Fréquence", "Monetary": "Montant"},
        )
        fig_scat.update_layout(title="Fréquence vs Montant (coloré par Récence)",
                                height=300)
        style_fig(fig_scat)
        st.plotly_chart(fig_scat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Segmentation RFM":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
        <div>
            <h2 style="margin:0 0 8px 0; color:#ffffff;">Segmentation RFM</h2>
            <p style="margin:0; color:#8b8fa8; font-size:0.9rem;">
                Clustering K-Means · Résultats RFM · Statistiques par segment
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Méthode du coude (optionnel)
    if show_elbow:
        st.markdown('<div class="section-header">08 — Méthode du Coude (choix du k optimal)</div>',
                    unsafe_allow_html=True)
        with st.spinner("Calcul des courbes elbow & silhouette…"):
            ks, inertias, sils = elbow_data(rfm)

        fig_elbow = make_subplots(specs=[[{"secondary_y": True}]])
        fig_elbow.add_trace(
            go.Scatter(x=ks, y=inertias, mode="lines+markers",
                       name="Inertie", line=dict(color="#7c6af7", width=2),
                       marker=dict(size=7)),
            secondary_y=False,
        )
        fig_elbow.add_trace(
            go.Scatter(x=ks, y=sils, mode="lines+markers",
                       name="Silhouette", line=dict(color="#50e3a4", width=2, dash="dot"),
                       marker=dict(size=7)),
            secondary_y=True,
        )
        fig_elbow.update_layout(title="Inertie & score Silhouette selon k", height=320)
        fig_elbow.update_yaxes(title_text="Inertie", secondary_y=False,
                                gridcolor="#1e2230", color="#c9cbdb")
        fig_elbow.update_yaxes(title_text="Score Silhouette", secondary_y=True,
                                color="#c9cbdb")
        style_fig(fig_elbow)
        st.plotly_chart(fig_elbow, use_container_width=True)

    # Métriques clustering
    st.markdown('<div class="section-header">09 — Résultats du Clustering K-Means</div>',
                unsafe_allow_html=True)

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.markdown(kpi_card("Segments créés", str(n_clusters), f"k = {n_clusters}"), unsafe_allow_html=True)
    mc2.markdown(kpi_card("Clients segmentés", f"{len(rfm_clustered):,}", "100 % RFM"), unsafe_allow_html=True)
    mc3.markdown(kpi_card("Score Silhouette", f"{sil_score:.3f}" if sil_score else "—",
                           "0 → 1 (plus c'est élevé, mieux c'est)"), unsafe_allow_html=True)
    mc4.markdown(kpi_card("Inertie", f"{inertia:,.0f}", "Somme des dist² intra-cluster"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_pie_seg, col_bubble = st.columns([1, 1.4])

    with col_pie_seg:
        seg_count = rfm_clustered["Segment"].value_counts().reset_index()
        seg_count.columns = ["Segment", "Clients"]
        fig_seg_pie = px.pie(
            seg_count, values="Clients", names="Segment",
            hole=0.5,
            color="Segment",
            color_discrete_map=seg_colors_map,
        )
        fig_seg_pie.update_traces(textinfo="percent+label",
                                   hovertemplate="%{label}<br>%{value:,} clients (%{percent})")
        fig_seg_pie.update_layout(title="Répartition des clients par segment",
                                   height=380, showlegend=False)
        style_fig(fig_seg_pie)
        st.plotly_chart(fig_seg_pie, use_container_width=True)

    with col_bubble:
        seg_stats = rfm_clustered.groupby("Segment").agg(
            Recency_moy=("Recency", "mean"),
            Frequency_moy=("Frequency", "mean"),
            Monetary_moy=("Monetary", "mean"),
            Count=("CustomerID", "count"),
        ).reset_index()

        fig_bubble = px.scatter(
            seg_stats,
            x="Recency_moy", y="Frequency_moy",
            size="Monetary_moy", color="Segment",
            text="Segment",
            size_max=60,
            color_discrete_map=seg_colors_map,
            labels={
                "Recency_moy": "Récence moyenne (jours)",
                "Frequency_moy": "Fréquence moyenne",
                "Monetary_moy": "Montant moyen (£)",
            },
        )
        fig_bubble.update_traces(textposition="top center", marker=dict(opacity=0.8))
        fig_bubble.update_layout(title="Segments — Récence vs Fréquence (taille = Montant)",
                                  height=380, showlegend=False)
        style_fig(fig_bubble)
        st.plotly_chart(fig_bubble, use_container_width=True)

    # Scatter 3D
    st.markdown('<div class="section-header">10 — Vue 3D des Segments RFM</div>',
                unsafe_allow_html=True)

    rfm_sample3d = rfm_clustered.sample(min(3000, len(rfm_clustered)), random_state=42)
    fig_3d = px.scatter_3d(
        rfm_sample3d,
        x="Recency", y="Frequency", z="Monetary",
        color="Segment",
        color_discrete_map=seg_colors_map,
        opacity=0.65,
        labels={"Recency": "Récence", "Frequency": "Fréquence", "Monetary": "Montant"},
        hover_data=["CustomerID"],
    )
    fig_3d.update_traces(marker=dict(size=3))
    fig_3d.update_layout(
        height=520,
        scene=dict(
            bgcolor="#13161f",
            xaxis=dict(backgroundcolor="#13161f", gridcolor="#1e2230", color="#c9cbdb"),
            yaxis=dict(backgroundcolor="#13161f", gridcolor="#1e2230", color="#c9cbdb"),
            zaxis=dict(backgroundcolor="#13161f", gridcolor="#1e2230", color="#c9cbdb"),
        ),
    )
    style_fig(fig_3d)
    st.plotly_chart(fig_3d, use_container_width=True)

    # Tableau stats par segment
    st.markdown('<div class="section-header">11 — Statistiques par Segment</div>',
                unsafe_allow_html=True)

    stats_display = rfm_clustered.groupby("Segment").agg(
        Clients=("CustomerID", "count"),
        Récence_moy=("Recency", "mean"),
        Fréquence_moy=("Frequency", "mean"),
        CA_moyen=("Monetary", "mean"),
        CA_total=("Monetary", "sum"),
    ).reset_index()
    stats_display["% Clients"] = (stats_display["Clients"] / stats_display["Clients"].sum() * 100).round(1)
    stats_display["% CA"] = (stats_display["CA_total"] / stats_display["CA_total"].sum() * 100).round(1)
    stats_display["Récence_moy"] = stats_display["Récence_moy"].round(0).astype(int)
    stats_display["Fréquence_moy"] = stats_display["Fréquence_moy"].round(1)
    stats_display["CA_moyen"] = stats_display["CA_moyen"].apply(lambda x: f"£{x:,.0f}")
    stats_display["CA_total"] = stats_display["CA_total"].apply(lambda x: f"£{x:,.0f}")

    st.dataframe(
        stats_display[["Segment", "Clients", "% Clients", "Récence_moy",
                        "Fréquence_moy", "CA_moyen", "CA_total", "% CA"]],
        use_container_width=True,
        hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — INTERPRÉTATION MÉTIER
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Interprétation Métier":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
        <div>
            <h2 style="margin:0 0 8px 0; color:#ffffff;">Interprétation Métier</h2>
            <p style="margin:0; color:#8b8fa8; font-size:0.9rem;">
                Profils clients · Recommandations · Actions marketing prioritaires
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('### 12 — Profils Clients & Recommandations')

    present_segs = rfm_clustered["Segment"].unique()

    for seg in present_segs:
        info = SEGMENT_INFO.get(seg, {})
        color = info.get("color", "#7c6af7")
        emoji = ""  # emojis removed
        desc = info.get("desc", "")
        action = info.get("action", "")

        seg_data = rfm_clustered[rfm_clustered["Segment"] == seg]
        n_cli = len(seg_data)
        r_avg = seg_data["Recency"].mean()
        f_avg = seg_data["Frequency"].mean()
        m_avg = seg_data["Monetary"].mean()
        m_total = seg_data["Monetary"].sum()
        pct = n_cli / len(rfm_clustered) * 100

        # profil client sans HTML compliqué
        st.markdown(f"**{seg}**  \
                     \n{n_cli:,} clients ({pct:.1f}%)")
        st.markdown(f"- **Profil :** {desc}")
        st.markdown(f"- **RFM :** Récence moy. {r_avg:.0f}j · Fréquence {f_avg:.1f} · CA moyen £{m_avg:,.0f} · CA total £{m_total:,.0f}")
        st.markdown(f"- **Action recommandée :** {action}")
        st.markdown("---")

    # Matrice valeur / engagement
    st.markdown('<div class="section-header">13 — Matrice Valeur / Engagement</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#8b8fa8; font-size:0.85rem; margin-bottom:12px;">
    Cette matrice positionne chaque segment selon sa valeur monétaire (CA moyen) et son engagement 
    (fréquence d'achat). Elle guide la priorisation des actions marketing.
    </div>
    """, unsafe_allow_html=True)

    seg_matrix = rfm_clustered.groupby("Segment").agg(
        Frequence=("Frequency", "mean"),
        CA=("Monetary", "mean"),
        Recence=("Recency", "mean"),
        Count=("CustomerID", "count"),
    ).reset_index()

    fig_matrix = px.scatter(
        seg_matrix,
        x="Frequence", y="CA",
        size="Count", color="Segment",
        text="Segment",
        size_max=70,
        color_discrete_map=seg_colors_map,
        labels={
            "Frequence": "← Peu engagé   |   Engagement (fréquence)   |   Très engagé →",
            "CA": "← Faible valeur   |   Valeur (CA moyen £)   |   Haute valeur →",
        },
    )
    fig_matrix.add_hline(y=seg_matrix["CA"].median(), line_dash="dot",
                          line_color="#2a2d45", annotation_text="Médiane CA",
                          annotation_font_color="#8b8fa8")
    fig_matrix.add_vline(x=seg_matrix["Frequence"].median(), line_dash="dot",
                          line_color="#2a2d45", annotation_text="Médiane freq.",
                          annotation_font_color="#8b8fa8")
    fig_matrix.update_traces(textposition="top center", marker=dict(opacity=0.85))
    fig_matrix.update_layout(height=480, showlegend=False)
    style_fig(fig_matrix)
    st.plotly_chart(fig_matrix, use_container_width=True)

    # Export
    st.markdown('<div class="section-header">14 — Export des données segmentées</div>',
                unsafe_allow_html=True)
    export_df = rfm_clustered[["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]].copy()
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Télécharger la segmentation (CSV)",
        data=csv_buffer.getvalue(),
        file_name="segmentation_rfm_clients.csv",
        mime="text/csv",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — FICHE CLIENT
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Fiche Client":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
        <div>
            <h2 style="margin:0 0 8px 0; color:#ffffff;">Fiche Client Détaillée</h2>
            <p style="margin:0; color:#8b8fa8; font-size:0.9rem;">
                Recherche par ID · Historique · Comportement d'achat · Profil RFM
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Consultation Fiche Client Détaillée</div>',
                unsafe_allow_html=True)

    fcol1, fcol2 = st.columns([2, 1])
    
    with fcol1:
        cust_id = st.text_input(
            "Saisir un ID Client",
            placeholder="Ex: 12345",
            help="Entrez l'identifiant unique du client"
        )
    
    with fcol2:
        search_btn = st.button("Rechercher", use_container_width=True)

    if search_btn and cust_id:
        # Chercher client dans les données
        cust_in_rfm = rfm_clustered[rfm_clustered["CustomerID"].astype(str) == str(cust_id)]
        cust_in_df = df[df["CustomerID"].astype(str) == str(cust_id)]

        if len(cust_in_rfm) == 0:
            st.error(f"Aucun client avec l'ID **{cust_id}** trouvé dans les données.")
        else:
            cust = cust_in_rfm.iloc[0]
            segment = cust["Segment"]
            info = SEGMENT_INFO.get(segment, {})
            
            # Fiche en haut
            fcol_header1, fcol_header2, fcol_header3 = st.columns([1, 2, 1.5])
            
            with fcol_header1:
                st.markdown(f"""
                <div class="info-box" style="border-color:{info.get('color', '#7c6af7')}40; border-left: 4px solid {info.get('color', '#7c6af7')}; text-align:center;">
                    
                    <div style="font-size:0.8rem; color:#8b8fa8;">Client ID</div>
                    <div style="font-size:1.3rem; font-weight:bold; color:#c9cbdb;">{cust_id}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with fcol_header2:
                st.markdown(f"""
                <div class="info-box" style="border-color:{info.get('color', '#7c6af7')}40; border-left: 4px solid {info.get('color', '#7c6af7')};">
                    <div style="font-size:1.1rem; font-weight:bold; color:{info.get('color', '#7c6af7')};">{segment}</div>
                    <p>{info.get('desc', '')}</p>
                    <p style="color:#8b8fa8; font-size:0.85rem;"><strong>Action recommandée :</strong> {info.get('action', '')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with fcol_header3:
                mf, rc, mo = st.columns(3)
                mf.metric("R (jours)", int(cust["Recency"]))
                rc.metric("F (commandes)", int(cust["Frequency"]))
                mo.metric("M (£)", f"£{cust['Monetary']:,.0f}")
            
            st.markdown("---")
            
            # Historique d'achats
            st.markdown('<div class="section-header">Historique d\'Achats</div>',
                        unsafe_allow_html=True)
            
            his_agg = cust_in_df.groupby("InvoiceNo").agg({
                "InvoiceDate": "first",
                "TotalPrice": "sum",
                "Quantity": "sum",
                "Country": "first"
            }).reset_index().sort_values("InvoiceDate", ascending=False)
            
            his_display = his_agg.copy()
            his_display["InvoiceDate"] = his_display["InvoiceDate"].dt.strftime("%d/%m/%Y")
            his_display["Total"] = his_display["TotalPrice"].apply(lambda x: f"£{x:,.2f}")
            his_display = his_display[["InvoiceNo", "InvoiceDate", "Quantity", "Total", "Country"]].head(20)
            his_display.columns = ["Facture", "Date", "Articles", "Montant", "Pays"]
            
            st.dataframe(his_display, use_container_width=True, hide_index=True)
            
            # Statistiques
            st.markdown('<div class="section-header">Comportement d\'Achat</div>',
                        unsafe_allow_html=True)
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            total_spent = cust_in_df["TotalPrice"].sum()
            total_qty = cust_in_df["Quantity"].sum()
            total_invoices = len(his_agg)
            avg_order = total_spent / total_invoices if total_invoices > 0 else 0
            
            stat_col1.metric("Total dépensé", f"£{total_spent:,.0f}")
            stat_col2.metric("Articles achetés", int(total_qty))
            stat_col3.metric("Nombre factures", int(total_invoices))
            stat_col4.metric("Panier moyen", f"£{avg_order:.2f}")
            
            # Chronologie
            his_agg["InvoiceDate"] = his_agg["InvoiceDate"].dt.to_period("M").astype(str)
            monthly_spend = his_agg.groupby("InvoiceDate")["TotalPrice"].sum()
            
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=monthly_spend.index, y=monthly_spend.values,
                marker=dict(color=info.get('color', '#7c6af7')),
                name="Dépenses mensuelles"
            ))
            fig_hist.update_layout(title="Évolution des dépenses", height=320)
            style_fig(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("Entrez un Customer ID pour consulter sa fiche détaillée.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — COMPARATEUR DE SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Comparateur":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
        <div>
            <h2 style="margin:0 0 8px 0; color:#ffffff;">Comparateur de Segments</h2>
            <p style="margin:0; color:#8b8fa8; font-size:0.9rem;">
                Analyse côte à côte · RFM comparatif · Diagramme radar
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Comparateur de Segments : Analyse Côte à Côte</div>',
                unsafe_allow_html=True)

    segments_available = sorted(rfm_clustered["Segment"].unique())
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        seg1 = st.selectbox("Segment 1", segments_available, key="seg1")
    
    with comp_col2:
        seg2 = st.selectbox("Segment 2", segments_available, 
                             index=min(1, len(segments_available) - 1) if len(segments_available) > 1 else 0,
                             key="seg2")

    if seg1 == seg2:
        st.warning("Sélectionnez deux segments différents pour une comparaison.")
    else:
        seg1_data = rfm_clustered[rfm_clustered["Segment"] == seg1]
        seg2_data = rfm_clustered[rfm_clustered["Segment"] == seg2]

        # Stats comparatives
        comp_df = pd.DataFrame({
            "Métrique": [
                "Clients",
                "% du portefeuille",
                "Récence (jours)",
                "Fréquence (cmd)",
                "Montant moyen (£)",
                "CA total (£)",
                "CA par client (£)"
            ],
            seg1: [
                len(seg1_data),
                f"{len(seg1_data) / len(rfm_clustered) * 100:.1f}%",
                f"{seg1_data['Recency'].mean():.0f}",
                f"{seg1_data['Frequency'].mean():.1f}",
                f"£{seg1_data['Monetary'].mean():.0f}",
                f"£{seg1_data['Monetary'].sum():,.0f}",
                f"£{seg1_data['Monetary'].sum() / len(seg1_data):,.0f}",
            ],
            seg2: [
                len(seg2_data),
                f"{len(seg2_data) / len(rfm_clustered) * 100:.1f}%",
                f"{seg2_data['Recency'].mean():.0f}",
                f"{seg2_data['Frequency'].mean():.1f}",
                f"£{seg2_data['Monetary'].mean():.0f}",
                f"£{seg2_data['Monetary'].sum():,.0f}",
                f"£{seg2_data['Monetary'].sum() / len(seg2_data):,.0f}",
            ]
        })

        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Radargramme
        st.markdown('<div class="section-header">Profil Radar</div>', unsafe_allow_html=True)
        
        # Normalisation pour radar
        r_norm_1 = (rfm_clustered["Recency"].max() - seg1_data["Recency"].mean()) / rfm_clustered["Recency"].max()
        f_norm_1 = seg1_data["Frequency"].mean() / rfm_clustered["Frequency"].max()
        m_norm_1 = seg1_data["Monetary"].mean() / rfm_clustered["Monetary"].max()
        
        r_norm_2 = (rfm_clustered["Recency"].max() - seg2_data["Recency"].mean()) / rfm_clustered["Recency"].max()
        f_norm_2 = seg2_data["Frequency"].mean() / rfm_clustered["Frequency"].max()
        m_norm_2 = seg2_data["Monetary"].mean() / rfm_clustered["Monetary"].max()

        fig_radar = go.Figure()
        
        info1 = SEGMENT_INFO.get(seg1, {})
        info2 = SEGMENT_INFO.get(seg2, {})
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[r_norm_1, f_norm_1, m_norm_1, r_norm_1],
            theta=["Récence", "Fréquence", "Montant", "Récence"],
            name=seg1,
            fill="toself",
            line=dict(color=info1.get("color", "#7c6af7")),
            opacity=0.6
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[r_norm_2, f_norm_2, m_norm_2, r_norm_2],
            theta=["Récence", "Fréquence", "Montant", "Récence"],
            name=seg2,
            fill="toself",
            line=dict(color=info2.get("color", "#50e3a4")),
            opacity=0.6
        ))
        
        fig_radar.update_layout(height=500, polar=dict(bgcolor="#13161f"))
        style_fig(fig_radar)
        st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — SIMULATEUR WHAT-IF
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Simulateur What-If":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
        <div>
            <h2 style="margin:0 0 8px 0; color:#ffffff;">Simulateur What-If</h2>
            <p style="margin:0; color:#8b8fa8; font-size:0.9rem;">
                Curseurs RFM · Prédiction segment · Scénarios hypothétiques
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('### Simulateur What-If : Quel segment tomberait un client ?')
    st.markdown("""
    Ajustez les trois dimensions du modèle RFM ci-dessous pour simuler un client hypothétique 
    et découvrir dans quel segment il serait classé.
    """)

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        sim_recency = st.slider(
            "Récence (jours depuis dernier achat)",
            min_value=0,
            max_value=int(rfm["Recency"].max()) + 30,
            value=int(rfm["Recency"].median()),
            step=1,
            help="Plus faible = plus actif récemment"
        )
    
    with sim_col2:
        sim_frequency = st.slider(
            "Fréquence (nombre de commandes)",
            min_value=1,
            max_value=int(rfm["Frequency"].max()) + 5,
            value=int(rfm["Frequency"].median()),
            step=1,
            help="Plus élevé = client plus régulier"
        )
    
    with sim_col3:
        sim_monetary = st.slider(
            "Montant total dépensé (£)",
            min_value=0.0,
            max_value=float(rfm["Monetary"].quantile(0.95)) * 1.5,
            value=float(rfm["Monetary"].median()),
            step=10.0,
            help="Plus élevé = client de meilleure valeur"
        )

    # Préparation du scaler et du model pour la prédiction
    X_train = rfm_clustered[["Recency", "Frequency", "Monetary"]].copy()
    X_train["Frequency"] = np.log1p(X_train["Frequency"])
    X_train["Monetary"] = np.log1p(X_train["Monetary"])
    
    scaler_sim = StandardScaler()
    X_train_scaled = scaler_sim.fit_transform(X_train)
    
    km_sim = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km_sim.fit(X_train_scaled)

    # Prédiction
    X_sim = np.array([[sim_recency, np.log1p(sim_frequency), np.log1p(sim_monetary)]])
    X_sim_scaled = scaler_sim.transform(X_sim)
    pred_cluster = km_sim.predict(X_sim_scaled)[0]
    
    # Récupérer segment
    cluster_to_segment = rfm_clustered.groupby("Cluster")["Segment"].first().to_dict()
    pred_segment = cluster_to_segment.get(pred_cluster, "Inconnu")
    pred_info = SEGMENT_INFO.get(pred_segment, {})

    st.markdown("---")
    st.markdown('<div class="section-header">Résultat de la Simulation</div>',
                unsafe_allow_html=True)

    result_col1, result_col2 = st.columns([1.5, 2])

    with result_col1:
        # affichage simple sans HTML
        st.markdown("**Segment prédit**")
        st.markdown(f"### {pred_segment}")

    with result_col2:
        st.markdown(f"""
        <div class="info-box">
            <p><strong>Profil du client simulé :</strong></p>
            <p>• <strong>Récence :</strong> {sim_recency} jours (inactivité: {"actif" if sim_recency < 60 else "moyen" if sim_recency < 180 else "inactif"})</p>
            <p>• <strong>Fréquence :</strong> {sim_frequency} commandes (engag.: {"élevé" if sim_frequency > rfm["Frequency"].quantile(0.75) else "moyen" if sim_frequency > rfm["Frequency"].quantile(0.25) else "faible"})</p>
            <p>• <strong>Montant :</strong> £{sim_monetary:,.0f} (valeur: {"haut" if sim_monetary > rfm["Monetary"].quantile(0.75) else "moyen" if sim_monetary > rfm["Monetary"].quantile(0.25) else "bas"})</p>
            <p style="color:{pred_info.get('color', '#7c6af7')}; margin-top:10px;"><strong>Recommandation :</strong> {pred_info.get('action', '')}</p>
        </div>
        """, unsafe_allow_html=True)

    # Comparaison avec la distribution réelle
    st.markdown('<div class="section-header">Comparaison avec Segments Réels</div>',
                unsafe_allow_html=True)

    seg_comp_data = rfm_clustered.groupby("Segment").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "CustomerID": "count"
    }).reset_index()
    seg_comp_data.columns = ["Segment", "Récence", "Fréquence", "Montant", "Clients"]

    # Ajouter le client virtuel
    virtuel_row = pd.DataFrame({
        "Segment": ["Simulé"],
        "Récence": [sim_recency],
        "Fréquence": [sim_frequency],
        "Montant": [sim_monetary],
        "Clients": [1]
    })

    seg_comp_data = pd.concat([seg_comp_data, virtuel_row], ignore_index=True)

    fig_sim_scatter = px.scatter_3d(
        seg_comp_data,
        x="Récence", y="Fréquence", z="Montant",
        color="Segment",
        size="Clients",
        text="Segment",
        size_max=50,
        labels={"Récence": "Récence (j)", "Fréquence": "Fréquence", "Montant": "Montant (£)"}
    )
    fig_sim_scatter.update_traces(textposition="top center", marker=dict(opacity=0.8))
    fig_sim_scatter.update_layout(height=600)
    style_fig(fig_sim_scatter)
    st.plotly_chart(fig_sim_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — CLIENTS À RISQUE DE CHURN
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Clients à Risque":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
        <div>
            <h2 style="margin:0 0 8px 0; color:#ffffff;">Clients à Risque</h2>
            <p style="margin:0; color:#8b8fa8; font-size:0.9rem;">
                Détection churn · Score de priorité · Actions prédictives
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <div style="font-size:1.2rem;">Tableau d'Alerte : Clients à Risque de Churn</div>
        <span class="tooltip-icon" title="Clients susceptibles d'être perdus prochainement, classés par urgence d'action. Score de 0 à 100.">?</span>
    </div>
    """, unsafe_allow_html=True)

    churn_data = compute_churn_risk(rfm_clustered)
    
    # Filtres
    churn_col1, churn_col2 = st.columns([1, 1])
    
    with churn_col1:
        min_churn_score = st.slider("Score de churn minimum", 0, 100, 60, step=5)
    
    with churn_col2:
        risk_cat = st.multiselect(
            "Filtrer par catégorie de risque",
            options=["Faible risque", "Risque moyen", "CRITIQUE"],
            default=["Faible risque", "Risque moyen", "CRITIQUE"],
            help="Sélectionner une ou plusieurs catégories"
        )

    # Appliquer filtres
    churn_filtered = churn_data[
        (churn_data["Churn_Score"] >= min_churn_score) &
        (churn_data["Churn_Category"].isin(risk_cat))
    ].head(100)

    if len(churn_filtered) == 0:
        st.info("Aucun client ne correspond à ces critères de churn.")
    else:
        # Afficher en tableau
        display_churn = churn_filtered.copy()
        display_churn["Churn_Score"] = display_churn["Churn_Score"].round(1)
        display_churn["Recency"] = display_churn["Recency"].astype(int)
        display_churn["Frequency"] = display_churn["Frequency"].astype(int)
        display_churn["Monetary"] = display_churn["Monetary"].apply(lambda x: f"£{x:,.0f}")
        
        # Badge couleur pour la catégorie
        def churn_badge(cat):
            # retourner un texte simple (sans HTML)
            if cat == "CRITIQUE":
                return "CRITIQUE"
            elif cat == "Risque moyen":
                return "Risque moyen"
            else:
                return "Faible risque"
        
        display_churn["Risque"] = display_churn["Churn_Category"].apply(churn_badge)
        display_churn = display_churn[
            ["CustomerID", "Recency", "Frequency", "Monetary", "Segment", "Churn_Score", "Risque"]
        ]
        display_churn.columns = ["ID Client", "Jour. Inactif", "Cmd.", "Dépensé", "Segment", "Score", "Catégorie"]
        
        st.markdown(f"**Résultat :** {len(churn_filtered)} clients à risque identifiés")
        st.dataframe(display_churn, use_container_width=True, hide_index=True)
        
        # Stats summary
        st.markdown("---")
        st.markdown("""
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
            <div style="font-size:1.1rem;">Analyse de Churn</div>
        </div>
        """, unsafe_allow_html=True)
        
        churn_stat_col1, churn_stat_col2, churn_stat_col3, churn_stat_col4 = st.columns(4)
        
        churn_stat_col1.metric(
            "CA total à risque",
            f"£{churn_filtered['Monetary'].sum():,.0f}",
            f"{churn_filtered['Monetary'].sum() / rfm_clustered['Monetary'].sum() * 100:.1f}% du portefeuille"
        )
        churn_stat_col2.metric(
            "Inactivité moyenne",
            f"{churn_filtered['Recency'].mean():.0f} jours",
            "depuis dernier achat"
        )
        churn_stat_col3.metric(
            "Fréquence moyenne",
            f"{churn_filtered['Frequency'].mean():.1f} cmd.",
            "par client"
        )
        churn_stat_col4.metric(
            "Score churn moyen",
            f"{churn_filtered['Churn_Score'].mean():.0f}/100",
            "urgence d'action"
        )
        
        # Graphique churn par segment
        churn_by_seg = churn_filtered.groupby("Segment").agg({
            "Churn_Score": "mean",
            "CustomerID": "count"
        }).reset_index()
        churn_by_seg.columns = ["Segment", "Score Moyen", "Nb Clients"]
        
        fig_churn_seg = px.bar(
            churn_by_seg,
            x="Segment",
            y="Score Moyen",
            color="Score Moyen",
            color_continuous_scale=[[0, "#50e3a4"], [0.5, "#f7a76c"], [1, "#e87c7c"]],
            labels={"Score Moyen": "Score de Churn", "Segment": ""},
            text="Nb Clients"
        )
        fig_churn_seg.update_traces(textposition="outside")
        fig_churn_seg.update_layout(height=380)
        style_fig(fig_churn_seg)
        st.plotly_chart(fig_churn_seg, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — EXPORT ET DÉCISIONS
# ══════════════════════════════════════════════════════════════════════════════
elif current_page == "Export & Décisions":
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:20px;">
        <div>
            <h2 style="margin:0 0 8px 0; color:#ffffff;">Export & Décisions</h2>
            <p style="margin:0; color:#8b8fa8; font-size:0.9rem;">
                Téléchargements · Email lists · Audit trail marketing
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px;">
        <div style="font-size:1.2rem;">Export & Documentation des Décisions</div>
        <span class="tooltip-icon" title="Exportez vos données et documentez les actions marketing prises par segment.">?</span>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION EXPORT
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('### Téléchargements')
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        st.subheader("Export Emails par Segment")
        st.markdown("""
        Générez une liste d'emails pour vos campagnes Mailchimp, MailerLite ou tout autre CRM.
        """)
        
        seg_for_email = st.selectbox(
            "Sélectionner le segment",
            options=rfm_clustered["Segment"].unique(),
            key="seg_email"
        )
        
        if st.button("Générer liste d'emails", use_container_width=True):
            # Récupérer les clients du segment
            seg_customers = rfm_clustered[rfm_clustered["Segment"] == seg_for_email]["CustomerID"].unique()
            seg_email_data = df[df["CustomerID"].isin(seg_customers)][["CustomerID", "InvoiceNo"]].drop_duplicates()
            
            # Mock emails (en vraie, il faudrait les vraies adresses emails)
            email_list = pd.DataFrame({
                "CustomerID": seg_email_data["CustomerID"].unique(),
                "Email": [f"customer+{cid}@example.com" for cid in seg_email_data["CustomerID"].unique()],
                "Segment": seg_for_email,
            })
            
            csv_emails = email_list.to_csv(index=False)
            st.download_button(
                label=f"Télécharger {len(email_list)} emails ({seg_for_email})",
                data=csv_emails,
                file_name=f"emails_{seg_for_email.lower().replace(' ', '_')}.csv",
                mime="text/csv",
            )
            st.success(f"{len(email_list)} adresses e-mails prêtes pour campagne Mailchimp")
    
    with exp_col2:
        st.subheader("Export Segmentation Complète")
        st.markdown("""
        Téléchargez la segmentation RFM complète pour vos analyses externes.
        """)
        
        export_full = rfm_clustered[["CustomerID", "Recency", "Frequency", "Monetary", "Segment"]].copy()
        export_full["Recency"] = export_full["Recency"].astype(int)
        export_full["Frequency"] = export_full["Frequency"].astype(int)
        export_full["Monetary"] = export_full["Monetary"].round(2)
        
        csv_full = export_full.to_csv(index=False)
        st.download_button(
            label=f"Télécharger segmentation ({len(export_full)} clients)",
            data=csv_full,
            file_name="segmentation_rfm_complete.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # ─────────────────────────────────────────────────────────────────────────
    # LOG DES DÉCISIONS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Log des Décisions Marketing</div>', 
                unsafe_allow_html=True)
    st.markdown("""
    Documentez ici vos décisions et actions marketing par segment. 
    Cela crée une trace d'audit de votre stratégie et facilite le suivi des résultats.
    """)
    
    # Créer les colonnes pour chaque segment
    seg_list = sorted(rfm_clustered["Segment"].unique())
    
    st.subheader("Actions décidées par segment")
    
    for i, segment in enumerate(seg_list):
        info = SEGMENT_INFO.get(segment, {})
        
        with st.expander(f"{segment} — Notes et Actions", expanded=(i == 0)):
            # Zone de texte pour les notes
            decision_text = st.text_area(
                f"Notes pour {segment}",
                placeholder=f"Ex: Campagne email lancée le XX/XX, remise de 10%, segmentation A/B test...",
                height=100,
                label_visibility="collapsed",
                key=f"decision_{segment}"
            )
            
            if decision_text:
                # Bouton pour sauvegarder
                col_decide1, col_decide2 = st.columns([3, 1])
                with col_decide2:
                    if st.button("Enregistrer", key=f"btn_{segment}"):
                        st.session_state.decision_log.append({
                            "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M"),
                            "segment": segment,
                            "decision": decision_text
                        })
                        st.success("Décision enregistrée")
    
    # Afficher historique
    if st.session_state.decision_log:
        st.markdown("---")
        st.markdown('<div class="section-header">Historique des Décisions</div>', 
                    unsafe_allow_html=True)
        
        hist_display = []
        for log_entry in reversed(st.session_state.decision_log):
            hist_display.append({
                "Heure": log_entry["timestamp"],
                "Segment": log_entry["segment"],
                "Décision": log_entry["decision"][:60] + "..." if len(log_entry["decision"]) > 60 else log_entry["decision"]
            })
        
        hist_df = pd.DataFrame(hist_display)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        
        # Export du log
        log_csv = hist_df.to_csv(index=False)
        st.download_button(
            label="Télécharger historique des décisions",
            data=log_csv,
            file_name="decision_log.csv",
            mime="text/csv",
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#3a3d55; font-size:0.75rem; padding:8px 0;">
  Dashboard Segmentation Clients E-Commerce &nbsp;·&nbsp; RFM + K-Means &nbsp;·&nbsp;
  UCI Online Retail Dataset &nbsp;·&nbsp; Streamlit
</div>
""", unsafe_allow_html=True)
