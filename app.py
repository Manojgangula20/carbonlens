import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import numpy as np
from io import StringIO
import re


# NEW:
from openai import OpenAI
import google.generativeai as genai


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Emissions Intelligence Dashboard",
    layout="wide",
    
)
# ---------- LLM CONFIG ----------
# Read keys from [api_keys] section in secrets.toml
api_keys = st.secrets.get("api_keys", {})

OPENAI_ENABLED = bool(api_keys.get("openai_api_key"))
GEMINI_ENABLED = bool(api_keys.get("gemini_api_key"))

# Configure Gemini ONLY (OpenAI is configured when creating the client)
if GEMINI_ENABLED:
    genai.configure(api_key=api_keys["gemini_api_key"])



# ---------- DATA LOADING ----------

@st.cache_data
def load_emissions_data():
    """
    Sector-wise GHG emissions by country & year.
    Source: Our World in Data – 'ghg-emissions-by-sector'
    """
    url = "https://ourworldindata.org/grapher/ghg-emissions-by-sector.csv"
    df = pd.read_csv(url)
    df = df.rename(columns={"Entity": "country", "Year": "year"})
    return df

@st.cache_data
def load_recent_co2_data():
    """
    Latest total CO2 & GHG metrics by country & year.
    Source: OWID 'owid-co2-data.csv'.
    """
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    df = pd.read_csv(url)
    cols_of_interest = {"country", "year", "co2", "co2_including_luc", "ghg_per_capita"}
    existing_cols = [c for c in df.columns if c in cols_of_interest]
    return df[existing_cols]

df_raw = load_emissions_data()
df_co2 = load_recent_co2_data()

# ---------- BASIC CHECKS & GLOBAL STRUCTURE ----------

required_cols = {"country", "year"}
if not required_cols.issubset(df_raw.columns):
    st.error("Sector dataset schema has changed. Please update column mappings.")
    st.stop()

max_year_sectors = int(df_raw["year"].max())

# Identify sector columns globally (used in many places)
non_sector_cols_global = ["country", "year"]
if "Code" in df_raw.columns:
    non_sector_cols_global.append("Code")
sector_cols_global = [c for c in df_raw.columns if c not in non_sector_cols_global]

# ---------- SIDEBAR CONTROLS ----------

st.sidebar.title("Filters")

all_years = sorted(df_raw["year"].unique())
default_year = max_year_sectors

year = st.sidebar.slider(
    "Year (sector-wise dataset)",
    int(all_years[0]),
    int(all_years[-1]),
    int(default_year),
)

countries_all = sorted(df_raw["country"].unique())
countries_with_world = ["World"] + countries_all

country = st.sidebar.selectbox("Primary region / country", countries_with_world, index=0)

compare_country = st.sidebar.selectbox(
    "Compare with another country (optional)",
    ["None"] + countries_all,
    index=0,
    help="Side-by-side comparison of sector emissions.",
)

benchmark_countries = st.sidebar.multiselect(
    "Benchmark multiple countries (total emissions in selected year)",
    options=countries_all,
    default=[c for c in ["China", "United States", "India"] if c in countries_all],
)

# ----- LAYOUT + DARK THEME -----

# Use a single Plotly template everywhere
plotly_template = "plotly_dark"

st.markdown(
    """
    <style>
    /* Full-width main container */
    [data-testid="block-container"] {
        max-width: 100% !important;
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        animation: fadeIn 0.6s ease-in-out;
    }

    /* Keep sidebar neat */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    /* Dark background + light text */
    .stApp {
        background-color: #0e1117 !important;
        color: #f5f5f5 !important;
    }

    /* Make common text elements light */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stCaption, label, .stRadio, .stSelectbox, .stMultiSelect,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
    .stDownloadButton > button, .stButton > button {
        color: #f5f5f5 !important;
    }

    /* Fade-in animation for main content */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f"_Note: Sector-wise breakdown is currently available up to **{max_year_sectors}** "
    "in this public dataset due to official reporting lags._"
)




# ---------- FILTER PRIMARY COUNTRY DATA ----------

if country == "World":
    df_filtered = df_raw[(df_raw["country"] == "World") & (df_raw["year"] == year)]
else:
    df_filtered = df_raw[(df_raw["country"] == country) & (df_raw["year"] == year)]

if df_filtered.empty:
    st.warning("No sector-wise data available for this selection.")
    st.stop()

non_sector_cols = ["country", "year", "Code"] if "Code" in df_filtered.columns else ["country", "year"]
sector_cols = [c for c in df_filtered.columns if c not in non_sector_cols]

df_sectors = df_filtered.melt(
    id_vars=["country", "year"],
    value_vars=sector_cols,
    var_name="sector",
    value_name="emissions_mtco2e",
)
df_sectors = df_sectors[df_sectors["emissions_mtco2e"].notna()]
total_emissions = df_sectors["emissions_mtco2e"].sum()

# ---------- KPI DELTAS (YOY) ----------

prev_year_for_kpi = year - 1
df_prev_kpi = df_raw[(df_raw["country"] == country) & (df_raw["year"] == prev_year_for_kpi)]

if not df_prev_kpi.empty:
    prev_sectors_kpi = df_prev_kpi.melt(
        id_vars=["country", "year"],
        value_vars=sector_cols,
        var_name="sector",
        value_name="emissions_mtco2e",
    )
    prev_total_kpi = prev_sectors_kpi["emissions_mtco2e"].sum()
    if prev_total_kpi != 0:
        total_delta_abs = total_emissions - prev_total_kpi
        total_delta_pct = total_delta_abs / prev_total_kpi * 100
        total_delta_str = f"{total_delta_pct:+.1f}% vs {prev_year_for_kpi}"
    else:
        total_delta_str = None
else:
    total_delta_str = None

# ---------- TITLE & TOP KPIs ----------

st.title("🌍 Emissions Intelligence Dashboard")
st.caption(
    "Explore how emissions evolve across sectors, countries, and time — and stress-test scenarios with an interactive copilot."
)
with st.expander("ℹ️ How to use this dashboard & data notes", expanded=False):
    st.markdown(
        f"""
- **Time coverage (sector-wise):** Greenhouse gas emissions by sector are available up to **{max_year_sectors}**.
- **Latest CO₂ snapshot:** Uses OWID’s separate CO₂ dataset, which may extend beyond sector-wise data.
- **Units:** All emissions values are shown in **tonnes of CO₂-equivalent (tCO₂e)**, unless otherwise noted.
- **Forecasts:** The time-series projection is based on a simple linear trend — illustrative, not a scientific climate model.
- **Best use:** Compare sectors, countries, and long-term trends (not exact regulatory reporting).
        """
    )


kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        "Total emissions (selected year, all sectors)",
        f"{total_emissions:,.0f}",
        delta=total_delta_str,
        help="Metric tonnes CO₂e (approx.) from sector-wise dataset.",
    )

top_sector_row = df_sectors.sort_values("emissions_mtco2e", ascending=False).iloc[0]

with kpi2:
    st.metric(
        "Top emitting sector",
        top_sector_row["sector"],
        f"{top_sector_row['emissions_mtco2e']:,.0f} tCO₂e",
    )

share_top3 = (
    df_sectors.sort_values("emissions_mtco2e", ascending=False)["emissions_mtco2e"]
    .head(3)
    .sum()
    / total_emissions
    * 100
) if total_emissions else 0

with kpi3:
    st.metric("Share of Top 3 sectors", f"{share_top3:.1f}%")

st.markdown(
    "_Sector-wise data uses a dedicated GHG-by-sector dataset, while the latest CO₂ "
    "snapshot below uses OWID's consolidated CO₂ dataset. Values are not expected to "
    "match exactly but should be directionally consistent._"
)

# ---------- ALERTS PANEL ----------

st.markdown("### 🔔 Emissions Alerts")

alerts = []

prev_year = year - 1
df_prev = df_raw[(df_raw["country"] == country) & (df_raw["year"] == prev_year)]
if not df_prev.empty:
    prev_sectors = df_prev.melt(
        id_vars=["country", "year"],
        value_vars=sector_cols,
        var_name="sector",
        value_name="emissions_mtco2e",
    )
    prev_total = prev_sectors["emissions_mtco2e"].sum()
    if prev_total != 0:
        delta = total_emissions - prev_total
        pct_change = delta / prev_total * 100
        if abs(pct_change) > 5:
            direction = "increased" if pct_change > 0 else "decreased"
            alerts.append(
                f"Total emissions for **{country}** have **{direction} by {pct_change:.1f}%** "
                f"vs **{prev_year}**."
            )

    # Sector-level alert
    df_curr_sectors = df_sectors.set_index("sector")["emissions_mtco2e"]
    prev_sectors_indexed = prev_sectors.set_index("sector")["emissions_mtco2e"]
    common_sector_index = df_curr_sectors.index.intersection(prev_sectors_indexed.index)
    if len(common_sector_index) > 0:
        delta_sector = (df_curr_sectors[common_sector_index] - prev_sectors_indexed[common_sector_index]).dropna()
        if not delta_sector.empty:
            max_sector = delta_sector.abs().idxmax()
            change = delta_sector[max_sector]
            direction = "up" if change > 0 else "down"
            alerts.append(
                f"The sector with the largest year-over-year change is **{max_sector}** "
                f"({direction} by {abs(change):,.0f} tCO₂e vs {prev_year})."
            )

if alerts:
    for a in alerts:
        st.info(a)
else:
    st.caption("No significant year-on-year changes detected based on simple thresholds.")


st.markdown("---")

# ---------- LATEST CO2 SNAPSHOT ----------

st.subheader("📌 Where are we today? (Latest CO₂ snapshot)")

df_co2_country = df_co2[df_co2["country"] == country]

if df_co2_country.empty:
    st.caption("Recent CO₂ data is not available for this selection in the OWID dataset.")
else:
    df_co2_country = df_co2_country.dropna(subset=["co2"])
    if df_co2_country.empty:
        st.caption("CO₂ values are missing for this region in the OWID dataset.")
    else:
        latest_year_recent = int(df_co2_country["year"].max())
        latest_row = df_co2_country[df_co2_country["year"] == latest_year_recent].iloc[0]
        latest_co2 = float(latest_row["co2"])

        col_latest1, col_latest2 = st.columns([1, 2])

        with col_latest1:
            st.metric(
                label=f"Latest fossil CO₂ emissions ({latest_year_recent})",
                value=f"{latest_co2:,.0f} tCO₂",
            )

        with col_latest2:
            st.caption(
                "This metric uses OWID's consolidated CO₂ dataset, which is updated "
                "annually using the latest Global Carbon Budget and other sources. "
                f"As of now, the latest available year for **{country}** is **{latest_year_recent}**. "
                "Sector-wise detail typically lags behind total CO₂ estimates."
            )

st.markdown("---")

# ---------- GLOBAL VIEW (MAP + TOP 5) ----------

st.subheader(f"🌍 Who is emitting the most in {year}? (Global view)")

df_year_global = df_raw[df_raw["year"] == year].copy()

if df_year_global.empty:
    st.caption(f"No data available for the year **{year}** to show a global view.")
    common_sector_cols_year = []
else:
    common_sector_cols_year = [c for c in sector_cols_global if c in df_year_global.columns]
    if common_sector_cols_year:
        df_year_global["total_emissions"] = df_year_global[common_sector_cols_year].sum(axis=1, skipna=True)

if df_year_global.empty or not common_sector_cols_year:
    st.caption("Could not compute total emissions across countries for this year.")
else:
    df_map = df_year_global[
        (df_year_global["country"] != "World") & df_year_global["total_emissions"].notna()
    ].copy()

    if df_map.empty:
        st.caption("No per-country totals available to draw the world map.")
    else:
        top5_global = df_map.sort_values("total_emissions", ascending=False).head(5)

        map_col, top5_col = st.columns([3, 1.4])

        with map_col:
            if "Code" in df_map.columns and df_map["Code"].notna().any():
                fig_map = px.choropleth(
                    df_map,
                    locations="Code",
                    color="total_emissions",
                    hover_name="country",
                    color_continuous_scale="Viridis",
                    labels={"total_emissions": "tCO₂e"},
                    title=f"Total emissions by country – {year}",
                    template=plotly_template,
                )
            else:
                fig_map = px.choropleth(
                    df_map,
                    locations="country",
                    locationmode="country names",
                    color="total_emissions",
                    hover_name="country",
                    color_continuous_scale="Viridis",
                    labels={"total_emissions": "tCO₂e"},
                    title=f"Total emissions by country – {year}",
                    template=plotly_template,
                )

            fig_map.update_layout(
                margin=dict(l=0, r=0, t=40, b=0),
                coloraxis_colorbar=dict(title="tCO₂e"),
            )
            st.plotly_chart(fig_map, width="stretch")

        with top5_col:
            st.markdown("**Top 5 emitters**")
            for i, row in enumerate(top5_global.itertuples(), start=1):
                st.write(f"{i}. **{row.country}** – {row.total_emissions:,.0f} tCO₂e")

st.markdown("---")

# ---------- COUNTRY COMPARISON (A vs B) ----------

st.subheader("🇨🇳 Compare economies – who emits more, and where?")

if compare_country == "None":
    st.caption("Select a comparison country in the sidebar to see side-by-side sector breakdown.")
else:
    df_a = df_raw[(df_raw["country"] == country) & (df_raw["year"] == year)]
    df_b = df_raw[(df_raw["country"] == compare_country) & (df_raw["year"] == year)]
    if df_a.empty or df_b.empty:
        st.caption("Not enough data to compare the selected pair for this year.")
    else:
        df_a_m = df_a.melt(
            id_vars=["country", "year"],
            value_vars=sector_cols,
            var_name="sector",
            value_name="emissions_mtco2e",
        )
        df_b_m = df_b.melt(
            id_vars=["country", "year"],
            value_vars=sector_cols,
            var_name="sector",
            value_name="emissions_mtco2e",
        )
        df_cmp = pd.concat([df_a_m, df_b_m], ignore_index=True)
        df_cmp = df_cmp[df_cmp["emissions_mtco2e"].notna()]

        fig_cmp = px.bar(
            df_cmp,
            x="sector",
            y="emissions_mtco2e",
            color="country",
            barmode="group",
            labels={"emissions_mtco2e": "tCO₂e"},
            template=plotly_template,
        )
        fig_cmp.update_layout(
            xaxis={"tickangle": -45},
            margin=dict(l=40, r=10, t=20, b=80),
        )
        st.plotly_chart(fig_cmp, width="stretch")

st.markdown("---")

# ---------- BENCHMARKING MULTIPLE COUNTRIES ----------

st.subheader(f"📊 Benchmark {year} – how do selected countries stack up?")

if not benchmark_countries:
    st.caption("Select at least one country in the sidebar to see benchmark chart.")
else:
    if df_year_global.empty or not common_sector_cols_year:
        st.caption("No data for selected benchmark countries in this year.")
    else:
        df_bench = df_year_global[df_year_global["country"].isin(benchmark_countries)].copy()
        if df_bench.empty:
            st.caption("No data for selected benchmark countries in this year.")
        else:
            df_bench["total_emissions"] = df_bench[common_sector_cols_year].sum(axis=1, skipna=True)
            fig_bench = px.bar(
                df_bench,
                x="country",
                y="total_emissions",
                labels={"total_emissions": "tCO₂e", "country": "Country"},
                template=plotly_template,
            )
            fig_bench.update_layout(
                margin=dict(l=40, r=10, t=40, b=80),
            )
            st.plotly_chart(fig_bench, width="stretch")

st.markdown("---")

# ---------- WITHIN-COUNTRY SECTOR BREAKDOWN ----------

st.subheader(f"🔍 Within {country} – which sectors drive emissions in {year}?")

sector_col1, sector_col2 = st.columns(2)

# ---------- SECTOR PIE CHART (IMPROVED) ----------
st.markdown("**Emissions breakdown by sector (pie)**")

df_pie = df_sectors.sort_values("emissions_mtco2e", ascending=False).copy()

# Wrap long sector names into 2 lines (replace first space with <br>)
df_pie["sector_wrapped"] = df_pie["sector"].str.replace(" ", "<br>", 1)

fig_pie = px.pie(
    df_pie,
    names="sector_wrapped",
    values="emissions_mtco2e",
    template=plotly_template,
    height=520,  # bigger pie
)

fig_pie.update_traces(
    textposition="inside",
    textinfo="label+value+percent",
    hovertemplate="<b>%{label}</b><br>%{value:,.0f} tCO₂e<extra></extra>",
)

fig_pie.update_layout(
    margin=dict(l=40, r=40, t=20, b=20),
)

st.plotly_chart(fig_pie, use_container_width=True)



# ---------- SECTOR BAR CHART (IMPROVED) ----------
st.markdown("**Emissions by sector (bar)**")

df_bar = df_sectors.sort_values("emissions_mtco2e", ascending=False).copy()

# Wrap labels to two lines
df_bar["sector_wrapped"] = df_bar["sector"].str.replace(" ", "<br>", 1)

fig_bar = px.bar(
    df_bar,
    x="sector_wrapped",
    y="emissions_mtco2e",
    labels={"emissions_mtco2e": "tCO₂e", "sector_wrapped": "Sector"},
    template=plotly_template,
    height=520,  # bigger chart
)

fig_bar.update_layout(
    margin=dict(l=40, r=20, t=30, b=80),
)

st.plotly_chart(fig_bar, use_container_width=True)


st.markdown("---")

# ---------- TIME-SERIES TRENDS + FORECAST ----------

st.subheader(f"⏱️ How have emissions evolved over time for {country}?")

df_country_hist = df_raw[df_raw["country"] == country].copy()

if df_country_hist.empty:
    st.caption(f"No historical data available for {country}.")
else:
    non_sector_cols_hist = ["country", "year", "Code"] if "Code" in df_country_hist.columns else ["country", "year"]
    sector_cols_hist = [c for c in df_country_hist.columns if c not in non_sector_cols_hist]

    df_country_hist["total_emissions"] = df_country_hist[sector_cols_hist].sum(axis=1, skipna=True)
    df_country_hist = df_country_hist.dropna(subset=["total_emissions"])
    df_country_hist = df_country_hist.sort_values("year")

    # Forecast next 5 years using simple linear fit
    if len(df_country_hist) >= 3:
        x = df_country_hist["year"].values
        y = df_country_hist["total_emissions"].values
        coeffs = np.polyfit(x, y, deg=1)
        poly = np.poly1d(coeffs)

        future_years = np.arange(df_country_hist["year"].max() + 1, df_country_hist["year"].max() + 6)
        future_emissions = poly(future_years)

        df_future = pd.DataFrame(
            {"year": future_years, "total_emissions": future_emissions, "type": "Forecast"}
        )
        df_hist_for_chart = df_country_hist[["year", "total_emissions"]].copy()
        df_hist_for_chart["type"] = "Historical"

        df_ts = pd.concat([df_hist_for_chart, df_future], ignore_index=True)
    else:
        df_ts = df_country_hist[["year", "total_emissions"]].copy()
        df_ts["type"] = "Historical"

    fig_ts = px.line(
        df_ts,
        x="year",
        y="total_emissions",
        color="type",
        labels={"total_emissions": "tCO₂e", "year": "Year"},
        template=plotly_template,
    )
    fig_ts.update_layout(margin=dict(l=40, r=10, t=20, b=40))
    st.plotly_chart(fig_ts, width="stretch")

    if "Forecast" in df_ts["type"].unique():
        st.caption(
            "Forecast is a simple trend-based projection (linear fit on historical totals) "
            "to illustrate how emissions may evolve if current patterns persist."
        )

st.markdown("---")

# ---------- EMISSIONS REDUCTION SIMULATOR ----------

st.subheader("🎛️ Emissions reduction simulator")

sim_col1, sim_col2 = st.columns(2)

with sim_col1:
    sim_sector = st.selectbox("Choose a sector to adjust", df_sectors["sector"].unique())
    sim_reduction_pct = st.slider(
        "Reduction in that sector (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
    )

with sim_col2:
    curr_val = float(df_sectors.set_index("sector").loc[sim_sector, "emissions_mtco2e"])
    new_val = curr_val * (1 - sim_reduction_pct / 100.0)
    new_total = total_emissions - curr_val + new_val
    absolute_drop = total_emissions - new_total
    pct_drop_total = absolute_drop / total_emissions * 100 if total_emissions else 0

    st.metric(
        f"New total emissions if {sim_sector} ↓ {sim_reduction_pct}%",
        f"{new_total:,.0f} tCO₂e",
        f"-{absolute_drop:,.0f} tCO₂e (−{pct_drop_total:.1f}%)",
    )
    st.caption(
        "This simple simulator shows how reducing a single sector affects the overall emissions "
        f"for **{country} in {year}**."
    )

st.markdown("---")

# ---------- DOWNLOAD / EXPORT ----------

st.subheader("📥 Download data & insights")

insights_buffer = StringIO()
insights_buffer.write(f"Emissions insights for {country} in {year}\n\n")
insights_buffer.write(f"Total emissions: {total_emissions:,.0f} tCO2e\n")
insights_buffer.write(f"Top sector: {top_sector_row['sector']} ({top_sector_row['emissions_mtco2e']:,.0f} tCO2e)\n")
insights_buffer.write(f"Share of top 3 sectors: {share_top3:.1f}%\n\n")

if alerts:
    insights_buffer.write("Alerts:\n")
    for a in alerts:
        insights_buffer.write("- " + a.replace("**", "") + "\n")
    insights_buffer.write("\n")

insights_text = insights_buffer.getvalue()

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    st.download_button(
        label="Download sector data (CSV)",
        data=df_sectors.to_csv(index=False),
        file_name=f"emissions_sectors_{country}_{year}.csv",
        mime="text/csv",
    )

with col_dl2:
    st.download_button(
        label="Download insights summary (TXT)",
        data=insights_text,
        file_name=f"emissions_insights_{country}_{year}.txt",
        mime="text/plain",
    )

st.markdown("---")

# ---------- CHAT PANEL ----------

st.subheader("🤖 Emissions Copilot – Ask Questions")

st.caption(
    "Ask about **this dashboard's data** (e.g. _“which country has the most emissions in "
    f"{year}?”_) or broader topics (e.g. _“what causes industrial emissions?”_). "
    "The app auto-routes your question to either the dataset or an ESG-tuned LLM with web context."
)

explain_mode = st.checkbox(
    "Enable detailed / explainable answers",
    value=False,
    help="When enabled, answers include more context and explanation.",
)

# LLM provider selection
llm_options = []
if OPENAI_ENABLED:
    llm_options.append("OpenAI")
if GEMINI_ENABLED:
    llm_options.append("Gemini")
if OPENAI_ENABLED and GEMINI_ENABLED:
    llm_options.append("Both")

if llm_options:
    llm_provider = st.radio(
        "LLM backend for ESG / web questions",
        options=llm_options,
        horizontal=True,
        help="Used when the question is not directly answerable from the dashboard data.",
    )
else:
    llm_provider = None
    st.warning(
        "LLM API keys are not configured. Web/ESG questions will fall back to a basic Wikipedia summary."
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Suggested questions
st.markdown("**Try one of these questions:**")
suggestions = [
    f"Which country has the most emissions recorded in {year}?",
    f"What are the top 3 emitting sectors in {country} in {year}?",
    f"How have total emissions for {country} changed over time?",
    "What causes industrial greenhouse gas emissions?",
    "Which sectors contribute most to global emissions?",
]

rows = [suggestions[:3], suggestions[3:]]
chosen_suggestion = None
for row in rows:
    cols = st.columns(len(row))
    for i, s in enumerate(row):
        if cols[i].button(s):
            chosen_suggestion = s


def detect_mode(question: str) -> str:
    """
    Auto-detect whether this is a data question or a web/ESG question.
    Returns: "data" or "web".
    """
    q = question.lower()

    data_keywords = [
        "which country",
        "top country",
        "most emissions",
        "highest emissions",
        "compare",
        "trend",
        "over time",
        "increase",
        "decrease",
        "sector",
        "share",
        "percentage",
        "top 3",
        "top three",
        "ranking",
        "rank",
        "in year",
        "latest",
        "current",
        "up to date",
        str(year),
    ]

    if any(k in q for k in data_keywords):
        return "data"
    return "web"


def extract_year_from_question(question: str) -> int | None:
    """
    Look for a 4-digit year in the question (e.g., 1990–2099).
    Returns an int year if found, else None.
    """
    q = question.lower()
    matches = re.findall(r"\b(19\d{{2}}|20\d{{2}})\b", q)
    if not matches:
        return None
    # Take the last mentioned year if multiple
    return int(matches[-1])

def answer_about_data(question: str, verbose: bool = False) -> str:
    """
    Rule-based answer generator about current data and simple global rankings.
    Tries to respect a year mentioned in the question; if that year is not
    available in the dataset, it falls back to the latest available year.
    """
    q = question.lower()
    lines = []

    # Figure out which year to use
    requested_year = extract_year_from_question(question)
    if requested_year is None:
        used_year = year  # current slider year
        year_note = ""
    else:
        if requested_year > max_year_sectors:
            used_year = max_year_sectors
            year_note = (
                f"Note: the dataset currently goes up to **{max_year_sectors}** only, "
                f"so I'm answering for **{max_year_sectors}** instead of {requested_year}."
            )
        else:
            used_year = requested_year
            year_note = "" if used_year == year else (
                f"(Using **{used_year}** based on your question, "
                f"not the dashboard year {year}.)"
            )

    # 1) Country ranking globally in selected year
    if ("which country" in q or "most emissions" in q or "highest emissions" in q) and "sector" not in q:
        df_year = df_raw[df_raw["year"] == used_year].copy()

        if df_year.empty:
            return f"I don't have sector-wise data for the year **{used_year}**."

        common_sector_cols = [c for c in sector_cols if c in df_year.columns]
        if not common_sector_cols:
            return "I couldn't identify sector columns to compute totals across countries."

        df_year["total_emissions"] = df_year[common_sector_cols].sum(axis=1, skipna=True)
        df_year_nonnull = df_year.dropna(subset=["total_emissions"])
        if df_year_nonnull.empty:
            return f"I don't have complete sector-wise totals for **{used_year}**."

        df_ranked = df_year_nonnull.sort_values("total_emissions", ascending=False)
        top_country_row = df_ranked.iloc[0]
        top_country = top_country_row["country"]
        top_value = top_country_row["total_emissions"]

        top5 = df_ranked.head(5)
        leaderboard_lines = []
        for i, (_, r) in enumerate(top5.iterrows(), start=1):
            leaderboard_lines.append(
                f"{i}. **{r['country']}** – {r['total_emissions']:,.0f} tCO₂e"
            )

        if year_note:
            lines.append(year_note)
            lines.append("")

        lines.append(
            f"In **{used_year}**, the country with the highest total sector-wise emissions is "
            f"**{top_country}**, with approximately **{top_value:,.0f} tCO₂e**."
        )

        if verbose:
            lines.append(
                "This is computed by summing emissions across all available sectors in the "
                "dataset for each country and then ranking by total."
            )

        lines.append("")
        lines.append("Top 5 emitters by total sector-wise emissions:")
        lines.append("\n".join(leaderboard_lines))

        return "\n\n".join(lines)

    # 2) Default: talk about current country/year (or requested year for this country)
    df_country_used = df_raw[(df_raw["country"] == country) & (df_raw["year"] == used_year)]
    if df_country_used.empty:
        return f"I don't have sector-wise data for **{country}** in **{used_year}**."

    non_sector_cols_local = ["country", "year", "Code"] if "Code" in df_country_used.columns else ["country", "year"]
    sector_cols_local = [c for c in df_country_used.columns if c not in non_sector_cols_local]

    df_sectors_used = df_country_used.melt(
        id_vars=["country", "year"],
        value_vars=sector_cols_local,
        var_name="sector",
        value_name="emissions_mtco2e",
    )
    df_sectors_used = df_sectors_used[df_sectors_used["emissions_mtco2e"].notna()]
    total_used = df_sectors_used["emissions_mtco2e"].sum()

    top3 = df_sectors_used.sort_values("emissions_mtco2e", ascending=False).head(3)

    if year_note:
        lines.append(year_note)
        lines.append("")

    lines.append(
        f"For **{country} in {used_year}**, total emissions across all tracked sectors "
        f"are approximately **{total_used:,.0f} tCO₂e**."
    )

    lines.append("The top three emitting sectors are:")
    for _, row in top3.iterrows():
        share = row["emissions_mtco2e"] / total_used * 100 if total_used else 0
        lines.append(
            f"- **{row['sector']}**: {row['emissions_mtco2e']:,.0f} tCO₂e "
            f"(~{share:.1f}% of the total)"
        )

    if "trend" in q or "increase" in q or "decrease" in q or "over time" in q:
        prev_year_local = used_year - 1
        df_prev_local = df_raw[(df_raw["country"] == country) & (df_raw["year"] == prev_year_local)]
        if not df_prev_local.empty:
            prev_sectors_local = df_prev_local.melt(
                id_vars=["country", "year"],
                value_vars=sector_cols_local,
                var_name="sector",
                value_name="emissions_mtco2e",
            )
            prev_total_local = prev_sectors_local["emissions_mtco2e"].sum()
            delta_local = total_used - prev_total_local
            if prev_total_local != 0:
                pct_change_local = delta_local / prev_total_local * 100
            else:
                pct_change_local = 0
            direction_local = "higher" if delta_local > 0 else "lower"

            lines.append(
                f"Compared to **{prev_year_local}**, total emissions are "
                f"**{abs(delta_local):,.0f} tCO₂e {direction_local}** "
                f"({pct_change_local:+.1f}% change)."
            )
        else:
            lines.append(
                f"I don't have sector-wise data for **{prev_year_local}** to compute a year-on-year trend."
            )

    if verbose:
        lines.append(
            "\n_These figures are based purely on the public OWID sector dataset and are "
            "intended to be directionally accurate rather than exact regulatory values._"
        )

    return "\n\n".join(lines)

def answer_latest_data_overview(verbose: bool = False) -> str:
    """
    Explain what the latest available emissions data is in this dashboard:
    - latest sector-wise year (from ghg-emissions-by-sector)
    - latest CO2 year (from owid-co2-data)
    """
    lines = []

    # Sector-wise dataset (E2E dashboard core)
    lines.append(
        f"The sector-wise greenhouse gas dataset used in this dashboard "
        f"currently goes up to **{max_year_sectors}**. "
        "This reflects the latest year for which sectoral GHG breakdowns are available "
        "from public sources such as Our World in Data and underlying UN/IEA inventories."
    )

    # Latest CO2 snapshot from owid-co2-data
    if not df_co2.empty:
        latest_co2_year = int(df_co2["year"].max())
        df_co2_world = df_co2[df_co2["country"] == "World"]
        if not df_co2_world.empty:
            df_co2_world = df_co2_world.dropna(subset=["co2"])
            if not df_co2_world.empty:
                latest_world_year = int(df_co2_world["year"].max())
                latest_world_val = float(
                    df_co2_world[df_co2_world["year"] == latest_world_year].iloc[0]["co2"]
                )
                lines.append(
                    f"For **total fossil CO₂ emissions**, a separate consolidated dataset "
                    f"is used, which currently has data up to **{latest_co2_year}**. "
                    f"For the world as a whole, the latest available year in that dataset is "
                    f"**{latest_world_year}**, with total fossil CO₂ emissions of "
                    f"about **{latest_world_val:,.0f} tonnes**."
                )
        else:
            lines.append(
                f"The consolidated CO₂ dataset has data up to **{latest_co2_year}**, "
                "but the global 'World' series is not available in this deployment."
            )

    lines.append(
        "Because official reporting and harmonisation of inventories take time, "
        "sector-wise breakdowns typically lag behind total CO₂ estimates by 1–2 years. "
        "This dashboard reflects that: sector charts stop at the latest year with "
        "reliable sector splits, while the CO₂ snapshot panel uses the most recent "
        "total emissions data available."
    )

    if verbose:
        lines.append(
            "\n_If you need more up-to-date or granular numbers (e.g., company-level), "
            "you would normally integrate paid or proprietary ESG / climate data providers. "
            "For this hackathon implementation, we intentionally stick to open datasets._"
        )

    return "\n\n".join(lines)



def search_wikipedia(query: str) -> str:
    """
    Minimal Wikipedia-based web info fetcher.
    """
    topic = query.strip()
    if not topic:
        return "Please provide a non-empty query."

    topic_with_context = topic + " greenhouse gas emissions"
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + topic_with_context.replace(" ", "_")

    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            extract = data.get("extract")
            if extract:
                return extract
            else:
                return "I couldn't find a concise summary for that query."
        else:
            return "No relevant article found for that query."
    except Exception as e:
        return f"Error while reaching the web API: {e}"


def build_context_summary() -> str:
    """
    Build a short summary of the current dashboard state
    to give the LLM some grounding in what the user is seeing.
    """
    top3 = df_sectors.sort_values("emissions_mtco2e", ascending=False).head(3)

    lines = [
        f"Selected country/region: {country}",
        f"Selected year: {year}",
        f"Total emissions (all sectors): {total_emissions:,.0f} tCO2e",
        "Top sectors by emissions:",
    ]
    for _, row in top3.iterrows():
        share = row["emissions_mtco2e"] / total_emissions * 100 if total_emissions else 0
        lines.append(
            f"- {row['sector']}: {row['emissions_mtco2e']:,.0f} tCO2e (~{share:.1f}% of total)"
        )

    return "\n".join(lines)


def ask_openai_esg(question: str, context: str, wiki_snippet: str, verbose: bool) -> str:
    if not OPENAI_ENABLED:
        return "OpenAI API key is not configured in this deployment."

    try:
        client = OpenAI(
            api_key=api_keys.get("openai_api_key") or st.secrets.get("OPENAI_API_KEY")
        )

        system_msg = (
            "You are an ESG and climate emissions expert assistant. "
            "Explain emissions, sectors, sustainability drivers, and climate impact clearly. "
            "Stay neutral, do not give investment advice."
        )

        user_msg = (
            f"Dashboard context:\n{context}\n\n"
            f"Web snippet:\n{wiki_snippet}\n\n"
            f"User question:\n{question}\n\n"
            + ("Provide reasoning and detailed ESG insights."
               if verbose else "Answer concisely with key ESG insights.")
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",  # <===== FIXED MODEL
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=700,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"OpenAI error: {e}"


def ask_gemini_esg(question: str, context: str, wiki_snippet: str, verbose: bool) -> str:
    if not GEMINI_ENABLED:
        return "Gemini API key is not configured in this deployment."

    model = genai.GenerativeModel("gemini-2.5-flash")

    context_block = f"Dashboard context:\n{context}\n\n"
    wiki_block = f"Web snippet (may be partial or noisy):\n{wiki_snippet}\n\n"

    prompt = (
        "You are an ESG and climate emissions specialist.\n"
        "Use the provided dashboard context and web snippet to answer the question.\n"
        "Stay factual, explain emissions drivers, sector roles, and sustainability angles.\n\n"
        f"{context_block}{wiki_block}"
        f"User question: {question}\n\n"
        + ("Provide a richer explanation with ESG framing, but avoid unnecessary jargon."
           if verbose else "Answer in 3–5 short paragraphs with clear ESG context and one or two concrete examples.")
    )

    try:
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Gemini error: {e}"


def is_llm_error(ans: str) -> bool:
    """
    Detect whether the model returned an API error instead of a real answer.
    We simply check for the formatted prefixes we use when wrapping LLM exceptions.
    """
    if not isinstance(ans, str):
        return False
    return ans.startswith("OpenAI error:") or ans.startswith("Gemini error:")
    


def ask_esg_llm(question: str, context: str, wiki_snippet: str, verbose: bool, provider) -> str:
    """
    Route ESG/web questions to OpenAI, Gemini, or both, but degrade gracefully
    to the web summary (Wikipedia) if model calls fail due to quota / invalid key.
    """
    # No provider selected → pure web search
    if not provider:
        return wiki_snippet

    # ----- OpenAI only -----
    if provider == "OpenAI":
        ans = ask_openai_esg(question, context, wiki_snippet, verbose)
        if is_llm_error(ans):
            return (
                wiki_snippet
                + "\n\n_The OpenAI ESG assistant is temporarily unavailable (quota / access). "
                "Showing a basic web summary instead._"
            )
        return ans

    # ----- Gemini only -----
    if provider == "Gemini":
        ans = ask_gemini_esg(question, context, wiki_snippet, verbose)
        if is_llm_error(ans):
            return (
                wiki_snippet
                + "\n\n_The Gemini ESG assistant is temporarily unavailable (API key / quota). "
                "Showing a basic web summary instead._"
            )
        return ans


    # ----- BOTH → combine non-error responses only -----
    openai_ans = ask_openai_esg(question, context, wiki_snippet, verbose)
    gemini_ans = ask_gemini_esg(question, context, wiki_snippet, verbose)

    parts = []
    if not is_llm_error(openai_ans):
        parts.append("**OpenAI perspective:**\n\n" + openai_ans)
    if not is_llm_error(gemini_ans):
        parts.append("**Gemini perspective:**\n\n" + gemini_ans)

    # If at least one succeeded → return the working one(s)
    if parts:
        return "\n\n---\n\n".join(parts)

    # If both failed → fallback
    return (
        wiki_snippet
        + "\n\n_Both ESG assistants are temporarily unavailable (keys / quota). "
        "Showing a basic web summary instead._"
    )

# ---- Chat input (no auto-scroll) ----
input_col1, input_col2 = st.columns([4, 1])

with input_col1:
    raw_user_input = st.text_input(
        "Ask a question about emissions...",
        key="chat_text",
        placeholder="e.g. Which country had the highest emissions in 2019?",
    )
with input_col2:
    send_clicked = st.button("Send", key="chat_send", width="stretch")

if chosen_suggestion:
    question = chosen_suggestion
elif send_clicked and raw_user_input:
    question = raw_user_input
else:
    question = None

if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

        mode = detect_mode(question)
    q_lower = question.lower()

    if mode == "data":
        # Special handling for "latest data" questions
        reply = answer_about_data(question, verbose=explain_mode)
        mode_text = "Answering from the dashboard data."
    else:
        context_summary = build_context_summary()
        wiki_snippet = search_wikipedia(question)

        reply = ask_esg_llm(
            question=question,
            context=context_summary,
            wiki_snippet=wiki_snippet,
            verbose=explain_mode,
            provider=llm_provider,
        )

        if llm_provider is None:
            mode_text = "Answering using a basic web summary (Wikipedia)."
        elif llm_provider == "Both":
            mode_text = "Answering using both OpenAI and Gemini ESG models plus web context."
        else:
            mode_text = f"Answering using {llm_provider} ESG model plus web context."


    reply = f"*{mode_text}*\n\n" + reply

    st.session_state.chat_history.append(("assistant", reply))
    with st.chat_message("assistant"):
        st.markdown(reply)