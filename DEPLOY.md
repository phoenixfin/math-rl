# Math-RL: Deployment Guide

## Project Structure

```
math-rl/
├── app.py                  ← Streamlit app (main interface)
├── requirements.txt        ← Python dependencies
├── .streamlit/
│   └── config.toml         ← Theme & server config
├── data/                   ← Auto-created: ratings.json, pending_paths.json
└── math_rl/                ← RL backend (from previous step)
    ├── __init__.py
    ├── env.py
    ├── agent.py
    └── train.py
```

---

## Local Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run locally
streamlit run app.py
# → Opens at http://localhost:8501
```

---

## Deploy to Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Math-RL theorem discovery interface"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/math-rl.git
git push -u origin main
```

### Step 2 — Deploy
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repo → Branch: `main` → Main file: `app.py`
5. Click **Deploy**

Your app will be live at:
`https://YOUR_USERNAME-math-rl-app-XXXX.streamlit.app`

---

## Deploy to Hugging Face Spaces (Alternative)

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create a new Space (Streamlit SDK)
# Go to: https://huggingface.co/new-space
# SDK: Streamlit | Visibility: Public or Private
# Then push:

git remote add space https://huggingface.co/spaces/YOUR_USERNAME/math-rl
git push space main
```

---

## Claude Code Workflow (Recommended)

After installing Claude Code (`npm install -g @anthropic-ai/claude-code`),
navigate to this folder and run:

```bash
cd math-rl/
claude
```

Then tell Claude Code:
- "Set up the git repo and push to GitHub"
- "Deploy this to Streamlit Cloud"
- "Help me connect the ratings output back to the RL training loop"

---

## Connecting Feedback to RL Training

Once ratings are collected in `data/ratings.json`, feed them back into training:

```python
import json
from math_rl.train import train

# Load human ratings
ratings = json.loads(open("data/ratings.json").read())

# Use composite score as reward signal shaping
# (this is where your reward model goes next)
for r in ratings:
    path      = r['path']
    human_score = r['composite']   # 1.0–5.0
    print(f"{path[-1]}: human={human_score:.1f}")
```

---

## Next Steps (Post-Deployment)

1. Collect ~50 ratings from mathematicians
2. Train a **reward model** on `(path, composite_score)` pairs
3. Replace checkpoint-based reward in `env.py` with reward model predictions
4. The agent now learns **human mathematical intuition** as its reward signal
