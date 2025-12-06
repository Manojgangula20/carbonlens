# 🌍 CarbonLens — Emissions Intelligence Dashboard with AI Copilot
### Built for **Stride Labs: HackForward 2025 — Round 2**

**Developer:** Manoj Gangula  
**Tech Stack:** Python • Streamlit • Plotly • Pandas • OpenAI • Gemini • OWID

---

## 🚀 Overview
CarbonLens is an end-to-end emissions analytics platform that transforms global greenhouse gas data into **interactive visual insights** and **AI-assisted ESG explanations**.

The system provides:
- Sector-wise emissions breakdown by country and year
- Global ranking and country-to-country comparison
- Historical trend analysis with forecasting
- Emissions reduction simulator
- AI chat copilot (OpenAI / Gemini / both) for ESG reasoning + web context

The goal is to make emissions intelligence **simple, explainable, and decision-focused**.

---

## 🗂 Data Sources
| Dataset | Purpose | Latest Year |
|--------|---------|-------------|
| OWID — GHG Emissions by Sector | Sector-wise breakdown across countries | **2022** |
| OWID — CO₂ Global Dataset | Latest consolidated CO₂ totals | **2024** |
| Wikipedia REST API | Web context for ESG queries | Real-time |

> Sector-wise reporting lags behind consolidated CO₂ totals; the dashboard accounts for this naturally.

---

## 🔑 Core Features
### 1️⃣ Sector-Wise Emissions Dashboard
- KPIs: total emissions, top sector, share of top 3 sectors  
- Pie chart + bar chart for sector breakdown

### 2️⃣ Global View
- Choropleth world map of emissions by country  
- Top 5 emitters list

### 3️⃣ Country-to-Country Comparison
- Side-by-side sector analysis

### 4️⃣ Benchmark Multiple Economies
- Compare emissions of several countries simultaneously

### 5️⃣ Time-Series Trends + Forecast
- Historical trends
- 5-year linear projection

### 6️⃣ Emissions Reduction Simulator
- Adjust emission share of any sector  
- Live impact on total emissions

### 7️⃣ AI Copilot — ESG + Web Intelligence
| Mode | Function |
|------|----------|
| Dashboard Data | Answers from dataset |
| OpenAI | ESG reasoning + sustainability insights |
| Gemini | ESG + contextual emissions framing |
| Both | Merged dual answer for richer perspective |

Auto-routing determines whether the question refers to dataset values or ESG/web knowledge.

---

## 🔧 Tech Stack
| Component | Technology |
|----------|-------------|
| App Framework | Streamlit |
| Charts | Plotly |
| Data Handling | Pandas / NumPy |
| LLMs | OpenAI `gpt-4o-mini`, Gemini |
| Web Context | Wikipedia REST API |
| Deployment | Streamlit Cloud / AWS / GCP / Render |

---

## 🔐 API Keys
Create a file:
.streamlit/secrets.toml
with:

```toml
[api_keys]
openai_api_key = "YOUR_OPENAI_KEY"
gemini_api_key = "YOUR_GEMINI_KEY"
```
LLM features activate automatically depending on which keys are present.

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### 🚢 Deployment Notes

The app supports deployment on:

* Streamlit Community Cloud

* Render

* HuggingFace Spaces

* AWS / Azure / GCP

Ensure secrets.toml or hosting-equivalent environment variables are configured.

### 📬 Submission Checklist
Requirement	|Status |
|-----------|-------|
|Dashboard with emissions data	|✔️|
|Chat panel for data queries	|✔️|
|LLM for ESG + web insights	|✔️|
|Intuitive user experience	|✔️|
|Full deployment	|🔜 (link to be added)|
|Documentation	|✔️|


### 🧭 Future Enhancements

* Company-level ESG scoring and climate risk indicators

* Policy impact + carbon pricing sensitivity modeling

* RAG pipeline using climate research datasets

* Alerts for NDC / SDG alignment and net-zero trajectories

### 👤 Contact

📧 Manoj Gangula
(available for follow-ups and discussion)

### 🏁 Final Note

This project shows how open climate datasets + interactive visual analytics + AI-assisted ESG intelligence can accelerate sustainability insight and decision-making — in line with Stride Ventures’ vision of technology-driven private market infrastructure.