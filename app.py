"""
Math-RL v3: Human Feedback Interface
Generative propositional logic — agent builds theorems from axioms.
UI-only (no TensorFlow). Training runs locally via train_local.py.
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="Math-RL · Theorem Discovery",
    page_icon="∴", layout="wide",
    initial_sidebar_state="expanded",
)

ROOT         = Path(__file__).parent
DATA_DIR     = ROOT / "data";   DATA_DIR.mkdir(exist_ok=True)
RATINGS_FILE = DATA_DIR / "ratings.json"
QUEUE_FILE   = DATA_DIR / "pending_paths.json"

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,400&display=swap');
html, body, .stApp { background: #0A0E17 !important; color: #E2E8F0; }
h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; }
code, .mono { font-family: 'DM Mono', monospace !important; }
[data-testid="stSidebar"] { background: #111827 !important; border-right: 1px solid #1F2D3D; }
.metric-card { background: #111827; border: 1px solid #1F2D3D; border-radius: 10px; padding: 14px 18px; text-align: center; margin-bottom: 10px; }
.metric-value { font-family: 'Fraunces', serif; font-size: 2.2rem; color: #E8A838; line-height: 1; }
.metric-label { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #64748B; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 4px; }

/* Proof step row */
.step-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-family: 'DM Mono', monospace; font-size: 0.83rem; }
.step-num  { color: #4A5568; min-width: 28px; text-align: right; }
.rule-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; }
.rule-A1 { background: #1E3A5F; color: #93C5FD; }
.rule-A2 { background: #1E3A5F; color: #93C5FD; }
.rule-A3 { background: #1E3A5F; color: #93C5FD; }
.rule-MP { background: #0F2E1E; color: #86EFAC; }
.formula  { color: #CBD5E1; flex: 1; }
.cp-badge { display: inline-block; padding: 2px 10px; border-radius: 12px; background: #3D1F0A; color: #FED7AA; border: 1px solid #C2540A; font-size: 0.72rem; }

/* Proof container */
.proof-box { background: #0A0F1A; border: 1px solid #1F2D3D; border-radius: 10px; padding: 16px 20px; max-height: 420px; overflow-y: auto; }
.formula-box { background: #0D1520; border-left: 3px solid #E8A838; border-radius: 0 8px 8px 0; padding: 8px 14px; font-family: 'DM Mono', monospace; font-size: 0.85rem; color: #CBD5E1; margin: 5px 0; }
.instruction-box { background: #111827; border: 1px solid #1F2D3D; border-radius: 10px; padding: 18px 22px; font-family: 'DM Mono', monospace; font-size: 0.84rem; color: #94A3B8; }
.stButton > button { font-family: 'DM Mono', monospace !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Checkpoint reference (no TF dependency) ───────────────────────────────────
CHECKPOINTS = {
    'T_identity':   {'name': 'Identity',               'formula': '(p→p)',                       'reward': 8.0},
    'T_double_neg': {'name': 'Double Negation Elim',   'formula': '(¬¬p→p)',                     'reward': 12.0},
    'T_contrapos':  {'name': 'Contrapositive',         'formula': '((p→q)→(¬q→¬p))',             'reward': 10.0},
    'T_ex_falso':   {'name': 'Ex Falso Quodlibet',     'formula': '(¬p→(p→q))',                  'reward': 8.0},
    'T_hyp_syll':   {'name': 'Hypothetical Syllogism', 'formula': '((p→q)→((q→r)→(p→r)))',       'reward': 9.0},
    'T_peirce':     {'name': "Peirce's Law",            'formula': '(((p→q)→p)→p)',               'reward': 11.0},
    'T_weakening':  {'name': 'Weakening',              'formula': '(p→(q→p))',                   'reward': 3.0},
}

RULE_DESC = {
    'A1': 'Axiom A1 (Weakening)',
    'A2': 'Axiom A2 (Frege)',
    'A3': 'Axiom A3 (Contraposition)',
    'MP': 'Modus Ponens',
}

SAMPLE_PATHS = [{
    'id': 'sample_001',
    'graph': 'Propositional Logic (Generative)',
    'type': 'generative',
    'steps': [
        {'rule': 'A1', 'args': ['p', 'q'],         'formula': '(p→(q→p))',                         'cp': 'T_weakening'},
        {'rule': 'A1', 'args': ['p', '(q→p)'],     'formula': '(p→((q→p)→p))',                     'cp': None},
        {'rule': 'A2', 'args': ['p', '(q→p)', 'p'],'formula': '((p→((q→p)→p))→((p→(q→p))→(p→p)))','cp': None},
        {'rule': 'MP', 'args': ['(p→((q→p)→p))', '((p→((q→p)→p))→((p→(q→p))→(p→p)))'],
                                                    'formula': '((p→(q→p))→(p→p))',                 'cp': None},
        {'rule': 'MP', 'args': ['(p→(q→p))', '((p→(q→p))→(p→p))'],
                                                    'formula': '(p→p)',                             'cp': 'T_identity'},
    ],
    'checkpoints_hit': ['T_weakening', 'T_identity'],
    'n_checkpoints': 2,
    'total_reward': 22.4,
    'episode': 7,
    'source': 'sample',
}]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_ratings():  return json.loads(RATINGS_FILE.read_text()) if RATINGS_FILE.exists() else []
def save_rating(r):
    ratings = load_ratings(); ratings.append(r)
    RATINGS_FILE.write_text(json.dumps(ratings, indent=2))
def load_queue():
    if not QUEUE_FILE.exists(): return SAMPLE_PATHS
    q = json.loads(QUEUE_FILE.read_text())
    return q if q else SAMPLE_PATHS

def rule_badge(rule):
    cls = f'rule-{rule}' if rule in ('A1','A2','A3','MP') else 'rule-A1'
    return f'<span class="rule-badge {cls}">{rule}</span>'

def cp_badge(cp_id):
    name = CHECKPOINTS.get(cp_id, {}).get('name', cp_id)
    return f'<span class="cp-badge">⭐ {name}</span>'

def render_proof(steps):
    """Render a proof as HTML step-by-step."""
    rows = []
    for i, s in enumerate(steps):
        rule   = s.get('rule', '?')
        formula= s.get('formula', '')
        cp     = s.get('cp')
        cp_html= f' {cp_badge(cp)}' if cp else ''
        rows.append(
            f'<div class="step-row">'
            f'<span class="step-num">{i+1}.</span>'
            f'{rule_badge(rule)}'
            f'<span class="formula">{formula}</span>'
            f'{cp_html}'
            f'</div>'
        )
    return '<div class="proof-box">' + ''.join(rows) + '</div>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ∴ Math-RL")
    st.markdown('<p style="color:#64748B;font-family:\'DM Mono\',monospace;font-size:0.76rem;">Generative Theorem Discovery<br>Human Feedback Interface v3</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", [
        "📋 Rate Proofs", "📊 Dashboard",
        "🗺️ Checkpoint Map", "⚙️ Settings",
    ], label_visibility="collapsed")
    st.markdown("---")

    ratings   = load_ratings()
    queue     = load_queue()
    rated_ids = {r['path_id'] for r in ratings}
    pending   = [p for p in queue if p['id'] not in rated_ids]

    c1, c2 = st.columns(2)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(pending)}</div><div class="metric-label">Pending</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{len(ratings)}</div><div class="metric-label">Rated</div></div>', unsafe_allow_html=True)

    n_left = max(0, 10 - len(ratings))
    hint   = f"{n_left} more ratings until reward model trains" if n_left > 0 else "Reward model: ready"
    st.markdown(f'<p style="color:#64748B;font-size:0.7rem;font-family:\'DM Mono\',monospace;">{hint}</p>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<p style="color:#4A5568;font-size:0.7rem;font-family:\'DM Mono\',monospace;">Training runs locally.<br>Push data/ to sync.</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Rate Proofs
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📋 Rate Proofs":
    st.markdown("# Rate Discovery Proofs")
    st.markdown('<p style="color:#64748B;">The agent built these proofs from scratch using A1/A2/A3 + Modus Ponens.<br>Rate each proof to train the reward model.</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not pending:
        st.success("✓ All proofs rated. Run `train_local.py` locally and push `data/` to generate new discoveries.")
    else:
        if 'path_idx' not in st.session_state:
            st.session_state.path_idx = 0
        st.session_state.path_idx = min(st.session_state.path_idx, len(pending)-1)
        pd = pending[st.session_state.path_idx]

        st.progress(len(rated_ids)/max(len(queue),1),
                    text=f"Rated {len(rated_ids)} of {len(queue)} proofs")
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Proof display ──────────────────────────────────────────────────────
        col_proof, col_stats = st.columns([3, 1])
        with col_proof:
            st.markdown(f"**Proof `{pd['id']}`** · Episode {pd.get('episode', 0)}")

            steps    = pd.get('steps', [])
            cps_hit  = pd.get('checkpoints_hit', [])

            st.markdown(render_proof(steps), unsafe_allow_html=True)

            if cps_hit:
                st.markdown("<br>**Checkpoints discovered:**", unsafe_allow_html=True)
                cp_html = " ".join(cp_badge(c) for c in cps_hit)
                st.markdown(cp_html, unsafe_allow_html=True)

        with col_stats:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{pd.get("total_reward",0):.0f}</div><div class="metric-label">Agent Reward</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(steps)}</div><div class="metric-label">Proof Steps</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(cps_hit)}/{len(CHECKPOINTS)}</div><div class="metric-label">Checkpoints</div></div>', unsafe_allow_html=True)

            # Rule breakdown
            rules = [s.get('rule','?') for s in steps]
            for rule in ['A1','A2','A3','MP']:
                n = rules.count(rule)
                if n: st.markdown(f'<span class="rule-badge rule-{rule}">{rule}</span> ×{n}', unsafe_allow_html=True)

        st.markdown("---")

        # ── Checkpoint details ─────────────────────────────────────────────────
        if cps_hit:
            with st.expander("📖 Checkpoint details", expanded=False):
                for cp_id in cps_hit:
                    cp_def = CHECKPOINTS.get(cp_id, {})
                    st.markdown(f"**{cp_def.get('name', cp_id)}**")
                    st.markdown(f'<div class="formula-box">{cp_def.get("formula", "")}</div>', unsafe_allow_html=True)

        # ── Rating ─────────────────────────────────────────────────────────────
        st.markdown("### Your Assessment")
        c1, c2, c3 = st.columns(3)
        with c1:
            elegance = st.slider("**Elegance**", 1, 5, 3,
                help="Is the proof concise? Does it avoid unnecessary steps?")
            st.caption("1=convoluted · 5=beautifully direct")
        with c2:
            depth = st.slider("**Depth**", 1, 5, 3,
                help="Does the proof reach significant results?")
            st.caption("1=trivial · 5=profound")
        with c3:
            novelty = st.slider("**Novelty**", 1, 5, 3,
                help="Does the agent find a non-obvious proof strategy?")
            st.caption("1=obvious · 5=surprising")
        notes = st.text_area("Notes (optional)",
            placeholder="e.g. 'Reaches Identity via an unusual A2 instantiation'",
            height=80)

        col_skip, col_sub = st.columns([1,3])
        with col_skip:
            if st.button("⏭ Skip", use_container_width=True):
                st.session_state.path_idx = (st.session_state.path_idx+1) % max(len(pending),1)
                st.rerun()
        with col_sub:
            if st.button("✓ Submit Rating", type="primary", use_container_width=True):
                r = {
                    'path_id':   pd['id'],
                    'graph':     pd.get('graph', 'Propositional Logic (Generative)'),
                    'steps':     steps,
                    'checkpoints_hit': cps_hit,
                    'elegance':  elegance,
                    'depth':     depth,
                    'novelty':   novelty,
                    'composite': round((elegance+depth+novelty)/3, 2),
                    'notes':     notes,
                    'timestamp': datetime.utcnow().isoformat(),
                }
                save_rating(r)
                st.success(f"✓ Rated! Composite: **{r['composite']:.1f}/5.0**")
                st.session_state.path_idx = min(st.session_state.path_idx+1, len(pending)-1)
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("# Ratings Dashboard")
    st.markdown("---")
    ratings = load_ratings()

    if not ratings:
        st.info("No ratings yet.")
    else:
        import pandas as pd
        df = pd.DataFrame(ratings)

        c1, c2, c3, c4 = st.columns(4)
        for col, val, label in zip([c1,c2,c3,c4],
            [len(df), df['composite'].mean(), df['elegance'].mean(), df['depth'].mean()],
            ['Total Ratings','Avg Composite','Avg Elegance','Avg Depth']):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-value">{val:.1f}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### All Ratings")
        disp = [c for c in ['path_id','elegance','depth','novelty','composite','notes','timestamp'] if c in df.columns]
        st.dataframe(df[disp].sort_values('composite', ascending=False),
                     use_container_width=True, hide_index=True)

        st.markdown("### Checkpoint Discovery Rate")
        if 'checkpoints_hit' in df.columns:
            from collections import Counter
            all_cps = [cp for row in df['checkpoints_hit'].dropna() for cp in (row if isinstance(row, list) else [])]
            if all_cps:
                cp_counts = Counter(all_cps)
                cp_df = pd.DataFrame([
                    {'Checkpoint': CHECKPOINTS.get(k,{}).get('name',k), 'Times Discovered': v}
                    for k,v in cp_counts.items()
                ])
                st.bar_chart(cp_df.set_index('Checkpoint'), color='#E05C2A')

        # Training plot
        st.markdown("### Training Curves")
        plt_path = DATA_DIR / "training_generative.png"
        if plt_path.exists():
            st.image(str(plt_path), use_container_width=True)
        else:
            st.info("No training plot yet. Run `train_local.py` and commit `data/`.")

        st.markdown("---")
        st.download_button("⬇ Export Ratings JSON",
                           data=RATINGS_FILE.read_text(),
                           file_name="math_rl_ratings.json",
                           mime="application/json")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Checkpoint Map
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Checkpoint Map":
    st.markdown("# Checkpoint Map — Propositional Logic (Generative)")
    st.markdown('<p style="color:#64748B;">Target theorems the agent can discover by constructing proofs from axiom schemas.</p>', unsafe_allow_html=True)
    st.markdown("---")

    img_path = DATA_DIR / "graph_propositional_logic_generative.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
    else:
        st.info("Graph image not found. Run `train_local.py` and commit `data/`.")

    st.markdown("---")
    st.markdown("### Checkpoint Reference")
    for cp_id, cp_def in CHECKPOINTS.items():
        col_id, col_name, col_formula, col_reward = st.columns([1.5, 2, 4, 1])
        with col_id:
            st.markdown(f'<span class="cp-badge">{cp_id}</span>', unsafe_allow_html=True)
        with col_name:
            st.markdown(f"**{cp_def['name']}**")
        with col_formula:
            st.markdown(f'<div class="formula-box">{cp_def["formula"]}</div>', unsafe_allow_html=True)
        with col_reward:
            st.markdown(f"⭐ `+{cp_def['reward']:.0f}`")

    st.markdown("---")
    st.markdown("### Inference Rules")
    for rule, desc in {
        'A1': 'p → (q → p)  —  Weakening: if p is true, it stays true given any q',
        'A2': '(p → (q → r)) → ((p → q) → (p → r))  —  Distribution of →',
        'A3': '(¬q → ¬p) → (p → q)  —  Contraposition / classical bridge',
        'MP': 'If p and (p → q) are proven, derive q  —  Modus Ponens',
    }.items():
        st.markdown(f'<span class="rule-badge rule-{rule}">{rule}</span> {desc}', unsafe_allow_html=True)
        st.markdown("")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Settings
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown("# Settings")
    st.markdown("---")

    st.markdown("### Workflow")
    st.markdown("""
<div class="instruction-box">
1. <b>Train locally:</b><br>
&nbsp;&nbsp;&nbsp;<code>python train_local.py --episodes 800</code><br><br>
2. <b>Commit outputs:</b><br>
&nbsp;&nbsp;&nbsp;<code>git add data/ models/ && git commit -m "run" && git push</code><br><br>
3. <b>Rate proofs here</b> → ratings.json grows<br><br>
4. <b>Retrain with human reward (--blend 0.3):</b><br>
&nbsp;&nbsp;&nbsp;<code>python train_local.py --blend 0.3</code><br><br>
5. Repeat — agent gradually learns human intuition.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Data Management")
    c1, c2 = st.columns(2)
    with c1:
        if RATINGS_FILE.exists():
            st.download_button("⬇ Download Ratings", data=RATINGS_FILE.read_text(),
                               file_name="ratings.json", mime="application/json",
                               use_container_width=True)
    with c2:
        if st.button("🗑 Clear Ratings", use_container_width=True):
            if RATINGS_FILE.exists(): RATINGS_FILE.unlink()
            st.warning("Ratings cleared."); st.rerun()
