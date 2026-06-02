"""
Math-RL: Human Feedback Interface
Streamlit app for mathematicians to rate theorem discovery paths.
"""

import streamlit as st
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="Math-RL · Theorem Discovery",
    page_icon="∴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
RATINGS_FILE = DATA_DIR / "ratings.json"
QUEUE_FILE   = DATA_DIR / "pending_paths.json"

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,400&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,400&display=swap');

:root {
    --bg:        #0A0E17;
    --surface:   #111827;
    --border:    #1F2D3D;
    --accent:    #E8A838;
    --accent2:   #3B82F6;
    --cp:        #F97316;
    --text:      #E2E8F0;
    --muted:     #64748B;
    --green:     #22C55E;
    --red:       #EF4444;
}

html, body, .stApp { background: var(--bg) !important; color: var(--text); }

h1, h2, h3 { font-family: 'Fraunces', Georgia, serif !important; }
code, .mono { font-family: 'DM Mono', monospace !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Metric boxes */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-value {
    font-family: 'Fraunces', serif;
    font-size: 2.4rem;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 6px;
}

/* Theorem pill */
.theorem-pill {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    padding: 4px 10px;
    border-radius: 20px;
    margin: 3px;
    border: 1px solid;
}
.pill-axiom      { background: #1E3A5F; border-color: #2E5D9F; color: #93C5FD; }
.pill-checkpoint { background: #3D1F0A; border-color: #C2540A; color: #FED7AA; }
.pill-theorem    { background: #0F2E1E; border-color: #166534; color: #86EFAC; }

/* Path display */
.path-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
    margin: 12px 0;
    font-family: 'DM Mono', monospace;
}
.path-arrow { color: var(--muted); margin: 0 4px; }

/* Formula box */
.formula-box {
    background: #0D1520;
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    color: #CBD5E1;
    margin: 6px 0;
}

/* Rating stars */
.stars { font-size: 1.6rem; letter-spacing: 4px; }

/* Divider */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* Badge */
.badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 500;
}
.badge-cp   { background: #431407; color: #FED7AA; }
.badge-new  { background: #1E3A5F; color: #BAE6FD; }

.stButton > button {
    font-family: 'DM Mono', monospace !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────────────────────────────────
THEOREM_META = {
    'A1':  {'name': 'Weakening',              'formula': 'p → (q → p)',                                 'type': 'axiom'},
    'A2':  {'name': 'Frege / Distribution',   'formula': '(p → (q → r)) → ((p → q) → (p → r))',       'type': 'axiom'},
    'A3':  {'name': 'Contraposition',         'formula': '(¬q → ¬p) → (p → q)',                       'type': 'axiom'},
    'T1':  {'name': 'Identity',               'formula': 'p → p',                                      'type': 'checkpoint'},
    'T2':  {'name': 'Argument Permutation',   'formula': '(p → q) → ((q → r) → (p → r))',             'type': 'theorem'},
    'T3':  {'name': 'Hypothetical Syllogism', 'formula': '(q → r) → ((p → q) → (p → r))',             'type': 'checkpoint'},
    'T4':  {'name': 'Double Negation Elim',   'formula': '¬¬p → p',                                    'type': 'checkpoint'},
    'T5':  {'name': 'Double Negation Intro',  'formula': 'p → ¬¬p',                                    'type': 'theorem'},
    'T6':  {'name': 'Contrapositive',         'formula': '(p → q) → (¬q → ¬p)',                       'type': 'checkpoint'},
    'T7':  {'name': 'Ex Falso Quodlibet',     'formula': '¬p → (p → q)',                               'type': 'checkpoint'},
    'T8':  {'name': 'Excluded Middle',        'formula': 'p ∨ ¬p',                                     'type': 'checkpoint'},
    'T9':  {'name': 'Non-Contradiction',      'formula': '¬(p ∧ ¬p)',                                  'type': 'checkpoint'},
    'T10': {'name': 'Reductio ad Absurdum',   'formula': '((p → q) ∧ (p → ¬q)) → ¬p',                'type': 'checkpoint'},
    'T11': {'name': 'Conjunction Elim',       'formula': '(p ∧ q) → p',                               'type': 'theorem'},
    'T12': {'name': 'Currying / Exportation', 'formula': '(p → (q → r)) ↔ ((p ∧ q) → r)',             'type': 'checkpoint'},
    'T13': {'name': "De Morgan's Law I",      'formula': '¬(p ∨ q) ↔ (¬p ∧ ¬q)',                     'type': 'checkpoint'},
    'T14': {'name': "De Morgan's Law II",     'formula': '¬(p ∧ q) ↔ (¬p ∨ ¬q)',                     'type': 'checkpoint'},
}
CHECKPOINTS = {t for t, m in THEOREM_META.items() if m['type'] == 'checkpoint'}

SAMPLE_PATHS = [
    {
        'id': 'path_001',
        'path': ['A1', 'A2', 'T1', 'T2', 'A3', 'T3', 'T4', 'T6', 'T7', 'T8', 'T9', 'T5', 'T10', 'T11', 'T12', 'T13', 'T14'],
        'episode': 42,
        'total_reward': 95.4,
        'source': 'agent_v1',
    },
    {
        'id': 'path_002',
        'path': ['A1', 'A2', 'A3', 'T1', 'T4', 'T2', 'T3', 'T7', 'T6', 'T5', 'T8', 'T9', 'T10', 'T11', 'T12', 'T13', 'T14'],
        'episode': 87,
        'total_reward': 91.2,
        'source': 'agent_v1',
    },
    {
        'id': 'path_003',
        'path': ['A2', 'A1', 'T2', 'A3', 'T1', 'T3', 'T4', 'T6', 'T8', 'T7', 'T9', 'T10', 'T5', 'T11', 'T13', 'T14', 'T12'],
        'episode': 153,
        'total_reward': 88.7,
        'source': 'agent_v1',
    },
]

def load_ratings() -> list:
    if RATINGS_FILE.exists():
        return json.loads(RATINGS_FILE.read_text())
    return []

def save_rating(rating: dict):
    ratings = load_ratings()
    ratings.append(rating)
    RATINGS_FILE.write_text(json.dumps(ratings, indent=2))

def load_queue() -> list:
    if QUEUE_FILE.exists():
        return json.loads(QUEUE_FILE.read_text())
    return SAMPLE_PATHS

def pill(t: str) -> str:
    meta = THEOREM_META.get(t, {})
    cls  = {'axiom': 'pill-axiom', 'checkpoint': 'pill-checkpoint', 'theorem': 'pill-theorem'}.get(meta.get('type','theorem'), 'pill-theorem')
    return f'<span class="theorem-pill {cls}">{t}</span>'


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ∴ Math-RL")
    st.markdown('<p style="color:#64748B;font-family:\'DM Mono\',monospace;font-size:0.78rem;">Theorem Discovery · Human Feedback</p>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", ["📋 Rate Paths", "📊 Dashboard", "🗺️ Knowledge Map", "⚙️ Settings"], label_visibility="collapsed")

    st.markdown("---")
    ratings = load_ratings()
    queue   = load_queue()
    rated_ids = {r['path_id'] for r in ratings}
    pending   = [p for p in queue if p['id'] not in rated_ids]

    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(pending)}</div><div class="metric-label">Paths Pending</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(ratings)}</div><div class="metric-label">Ratings Collected</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="color:#64748B;font-size:0.72rem;font-family:\'DM Mono\',monospace;">Reward model retrains every 50 ratings</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Rate Paths
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📋 Rate Paths":
    st.markdown("# Rate Discovery Paths")
    st.markdown('<p style="color:#64748B;">Evaluate the theorem paths discovered by the RL agent. Your ratings train the reward model.</p>', unsafe_allow_html=True)
    st.markdown("---")

    if not pending:
        st.success("✓ All current paths have been rated. Check back after the next training run.")
    else:
        # Pick current path
        if 'path_idx' not in st.session_state:
            st.session_state.path_idx = 0
        st.session_state.path_idx = min(st.session_state.path_idx, len(pending) - 1)
        path_data = pending[st.session_state.path_idx]

        # Progress
        total = len(queue)
        done  = len(rated_ids)
        st.progress(done / max(total, 1), text=f"Rated {done} of {total} paths")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Path Display ──────────────────────────────────────────────────────
        col_info, col_stats = st.columns([3, 1])

        with col_info:
            st.markdown(f"**Path `{path_data['id']}`** · Episode {path_data['episode']}")
            pills_html = ' <span class="path-arrow">→</span> '.join(pill(t) for t in path_data['path'])
            st.markdown(f'<div class="path-container">{pills_html}</div>', unsafe_allow_html=True)

            # Checkpoint coverage
            hit_cps = [t for t in path_data['path'] if t in CHECKPOINTS]
            st.markdown(f"**Checkpoints hit:** {len(hit_cps)} / {len(CHECKPOINTS)}")
            cp_pills = " ".join(pill(t) for t in hit_cps)
            st.markdown(cp_pills, unsafe_allow_html=True)

        with col_stats:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{path_data["total_reward"]:.0f}</div><div class="metric-label">Agent Reward</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(path_data["path"])}</div><div class="metric-label">Steps</div></div>', unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Theorem Details ───────────────────────────────────────────────────
        with st.expander("📖 View theorem details", expanded=False):
            for t in path_data['path']:
                meta = THEOREM_META.get(t, {})
                badge = '<span class="badge badge-cp">checkpoint</span>' if meta.get('type') == 'checkpoint' else ''
                st.markdown(f"**{t}** — {meta.get('name','')} {badge}", unsafe_allow_html=True)
                st.markdown(f'<div class="formula-box">{meta.get("formula","")}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Rating Form ───────────────────────────────────────────────────────
        st.markdown("### Your Assessment")
        st.markdown('<p style="color:#64748B;font-size:0.87rem;">Rate this discovery path on three dimensions. These scores directly train the reward model.</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            elegance = st.slider("**Elegance**", 1, 5, 3,
                help="Is the path concise and non-redundant? Does it avoid unnecessary detours?")
            st.caption("1 = convoluted · 5 = beautifully direct")

        with col2:
            depth = st.slider("**Mathematical Depth**", 1, 5, 3,
                help="Does the path traverse genuinely significant theorems? Does it build toward important results?")
            st.caption("1 = trivial · 5 = profound")

        with col3:
            novelty = st.slider("**Novelty / Surprise**", 1, 5, 3,
                help="Did the agent find an unexpected or non-obvious ordering that a human might not have chosen?")
            st.caption("1 = obvious · 5 = surprising")

        notes = st.text_area("Notes (optional)",
            placeholder="Any observations about this path — e.g. 'Derives excluded middle unusually early' or 'Misses a more elegant route through T5'",
            height=90)

        st.markdown("<br>", unsafe_allow_html=True)
        col_skip, col_submit = st.columns([1, 3])

        with col_skip:
            if st.button("⏭ Skip", use_container_width=True):
                st.session_state.path_idx = (st.session_state.path_idx + 1) % len(pending)
                st.rerun()

        with col_submit:
            if st.button("✓ Submit Rating", type="primary", use_container_width=True):
                rating = {
                    'path_id':   path_data['id'],
                    'path':      path_data['path'],
                    'source':    path_data.get('source', 'agent'),
                    'elegance':  elegance,
                    'depth':     depth,
                    'novelty':   novelty,
                    'composite': round((elegance + depth + novelty) / 3, 2),
                    'notes':     notes,
                    'timestamp': datetime.utcnow().isoformat(),
                }
                save_rating(rating)
                st.success(f"✓ Rated! Composite score: **{rating['composite']:.1f} / 5.0**")
                st.session_state.path_idx = min(st.session_state.path_idx + 1, len(pending) - 1)
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("# Ratings Dashboard")
    st.markdown("---")

    ratings = load_ratings()
    if not ratings:
        st.info("No ratings yet. Rate some paths first.")
    else:
        import pandas as pd

        df = pd.DataFrame(ratings)

        # Top stats
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df)}</div><div class="metric-label">Total Ratings</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df["composite"].mean():.1f}</div><div class="metric-label">Avg Score</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df["elegance"].mean():.1f}</div><div class="metric-label">Avg Elegance</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{df["depth"].mean():.1f}</div><div class="metric-label">Avg Depth</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Ratings table
        st.markdown("### All Ratings")
        display_cols = ['path_id', 'elegance', 'depth', 'novelty', 'composite', 'notes', 'timestamp']
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].sort_values('composite', ascending=False),
                     use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Dimension breakdown
        st.markdown("### Dimension Breakdown")
        dim_data = {
            'Dimension': ['Elegance', 'Depth', 'Novelty'],
            'Average':   [df['elegance'].mean(), df['depth'].mean(), df['novelty'].mean()],
        }
        dim_df = pd.DataFrame(dim_data)
        st.bar_chart(dim_df.set_index('Dimension'), color='#E8A838')

        # Export
        st.markdown("---")
        st.download_button(
            "⬇ Export Ratings JSON",
            data=RATINGS_FILE.read_text(),
            file_name="math_rl_ratings.json",
            mime="application/json",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Knowledge Map
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Knowledge Map":
    st.markdown("# Knowledge Map")
    st.markdown('<p style="color:#64748B;">The dependency graph of propositional logic. Edges = logical prerequisites.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Legend
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown('<span class="theorem-pill pill-axiom">A1, A2, A3</span> Hilbert Axioms', unsafe_allow_html=True)
    with col2: st.markdown('<span class="theorem-pill pill-checkpoint">T1, T3…</span> Checkpoints', unsafe_allow_html=True)
    with col3: st.markdown('<span class="theorem-pill pill-theorem">T2, T5…</span> Derived Theorems', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Theorem table
    st.markdown("### Theorem Reference")
    for tid, meta in THEOREM_META.items():
        col_id, col_name, col_formula, col_type = st.columns([1, 2, 4, 1.5])
        with col_id:   st.markdown(pill(tid), unsafe_allow_html=True)
        with col_name: st.markdown(f"**{meta['name']}**")
        with col_formula: st.markdown(f'<div class="formula-box" style="margin:2px 0;padding:6px 12px;">{meta["formula"]}</div>', unsafe_allow_html=True)
        with col_type:
            badge_cls = 'badge-cp' if meta['type'] != 'axiom' else 'badge-new'
            st.markdown(f'<span class="badge {badge_cls}">{meta["type"]}</span>', unsafe_allow_html=True)

    if (ROOT.parent / "knowledge_graph.png").exists():
        st.markdown("---")
        st.image(str(ROOT.parent / "knowledge_graph.png"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Settings
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown("# Settings")
    st.markdown("---")

    st.markdown("### Add New Path to Queue")
    st.markdown('<p style="color:#64748B;font-size:0.87rem;">Paste a path from the RL agent output to add it for rating.</p>', unsafe_allow_html=True)

    raw = st.text_input("Path (comma-separated theorem IDs)",
                        placeholder="A1, A2, A3, T1, T2, T3, T4, T6, T8, T9, T13, T14")
    ep  = st.number_input("Episode", min_value=0, value=0)
    rew = st.number_input("Agent Reward", value=0.0)

    if st.button("Add to Queue", type="primary"):
        if raw:
            path_ids = [t.strip() for t in raw.split(',')]
            new_path = {
                'id':           f"path_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                'path':         path_ids,
                'episode':      ep,
                'total_reward': rew,
                'source':       'manual',
            }
            queue = load_queue()
            queue.append(new_path)
            QUEUE_FILE.write_text(json.dumps(queue, indent=2))
            st.success(f"Added `{new_path['id']}` to rating queue.")

    st.markdown("---")
    st.markdown("### Data Management")

    col_dl, col_reset = st.columns(2)
    with col_dl:
        if RATINGS_FILE.exists():
            st.download_button("⬇ Download All Ratings",
                               data=RATINGS_FILE.read_text(),
                               file_name="ratings.json",
                               mime="application/json",
                               use_container_width=True)
    with col_reset:
        if st.button("🗑 Clear All Ratings", use_container_width=True):
            if RATINGS_FILE.exists():
                RATINGS_FILE.unlink()
            st.warning("Ratings cleared.")
            st.rerun()
