# Quant Pipeline

XGBoost-powered BUY / SELL signal generator for NSE stocks.
Runs as a FastAPI web app — accessible from any device including iPad.

## Project structure

```
quant_pipeline/
├── ml/
│   ├── train.py        # Downloads data, trains model, saves artefacts
│   └── predict.py      # CLI inference tool
├── server/
│   └── app.py          # FastAPI server (replaces the old Node.js server)
├── public/
│   └── index.html      # Frontend UI (served as static files)
├── Dockerfile
├── railway.toml
├── render.yaml
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
python ml/train.py                                    # first time only (~2 min)
uvicorn server.app:app --reload --port 3000
# open http://localhost:3000
```

---

## Deploy to Railway (recommended — fastest)

Railway gives you a live HTTPS URL in about 5 minutes.

### Step 1 — Push your code to GitHub

```bash
git init
git add .
git commit -m "initial commit"
# create a repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/quant-pipeline.git
git push -u origin main
```

### Step 2 — Create a Railway project

1. Go to [railway.app](https://railway.app) and sign up (free, GitHub login works)
2. Click **New Project → Deploy from GitHub repo**
3. Select your `quant-pipeline` repo
4. Railway detects the `Dockerfile` automatically — click **Deploy**

### Step 3 — Get your URL

1. In the Railway dashboard, open your service → **Settings → Networking**
2. Click **Generate Domain** — you get a URL like `quant-pipeline-production.up.railway.app`
3. Open that URL on your iPad's Safari — done ✅

> **First boot note:** On the very first deploy, the server auto-trains the model
> (downloads 5 years of data for 4 stocks and fits XGBoost).
> This takes ~2–3 minutes. The `/health` endpoint will return `model_ready: false`
> until training completes. Refresh the page after a couple of minutes.

---

## Deploy to Render (alternative)

### Step 1 — Push to GitHub (same as above)

### Step 2 — Create a Render Web Service

1. Go to [render.com](https://render.com) and sign up (free)
2. Click **New → Web Service**
3. Connect your GitHub repo
4. Render detects the `Dockerfile` — leave all defaults
5. Click **Create Web Service**

### Step 3 — Open your URL

Render gives you a URL like `quant-pipeline.onrender.com`.
Open it on your iPad.

> **Free tier note:** Render free services spin down after 15 min of inactivity.
> The first request after a sleep takes ~30–60 sec to wake up.
> Railway's free tier does not spin down.
> 

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + model ready flag |
| `GET` | `/live?ticker=RELIANCE.NS` | Fetch live data & predict |
| `POST` | `/predict` | Manual feature input → signal |
| `GET` | `/metadata` | Model info (accuracy, features, date) |
| `POST` | `/retrain` | Trigger background retraining |
| `GET` | `/retrain/status` | Is retraining in progress? |

## Supported tickers

| Ticker | Stock |
|--------|-------|
| `RELIANCE.NS` | Reliance Industries |
| `TCS.NS` | Tata Consultancy Services |
| `INFY.NS` | Infosys |
| `HDFCBANK.NS` | HDFC Bank |

To add more stocks, edit the `STOCKS` list in `ml/train.py` and retrain.
