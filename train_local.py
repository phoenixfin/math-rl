"""
train_local.py — Run on your machine to train the generative agent.

Usage:
    python train_local.py                     # default 800 episodes
    python train_local.py --episodes 1500
    python train_local.py --blend 0.4         # 40% learned reward, 60% checkpoint

After running:
    git add data/ models/
    git commit -m "training run"
    git push
    → Streamlit Cloud picks up new paths automatically.
"""

import argparse, os, sys, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx, numpy as np
from collections import defaultdict

from math_rl.env          import GenerativeLogicEnv
from math_rl.agent        import DQNAgent
from math_rl.train        import train, plot
from math_rl.exporter     import export_paths
from math_rl.reward_model import RewardModel
from math_rl.formula      import CHECKPOINTS_DEF


# ── Knowledge graph image (static, formula-level) ─────────────────────────────
def generate_checkpoint_graph(data_dir: Path):
    """Generate a graph showing checkpoint relationships for the formula system."""
    out = data_dir / "graph_propositional_logic_generative.png"

    G = nx.DiGraph()
    nodes = {
        'Axioms':       {'type': 'axiom',      'label': 'A1, A2, A3\n(Hilbert Axioms)'},
        'Weakening':    {'type': 'checkpoint',  'label': 'p→(q→p)\nWeakening'},
        'Identity':     {'type': 'checkpoint',  'label': '(p→p)\nIdentity'},
        'HypSyll':      {'type': 'checkpoint',  'label': '(p→q)→((q→r)→(p→r))\nHyp. Syllogism'},
        'DblNeg':       {'type': 'checkpoint',  'label': '(¬¬p→p)\nDbl Neg. Elim'},
        'Contrapos':    {'type': 'checkpoint',  'label': '(p→q)→(¬q→¬p)\nContrapositive'},
        'ExFalso':      {'type': 'checkpoint',  'label': '(¬p→(p→q))\nEx Falso'},
        'Peirce':       {'type': 'checkpoint',  'label': '(((p→q)→p)→p)\nPeirce\'s Law'},
    }
    edges = [
        ('Axioms','Weakening'), ('Axioms','Identity'), ('Axioms','HypSyll'),
        ('Weakening','Identity'), ('Identity','DblNeg'),
        ('HypSyll','Contrapos'), ('DblNeg','Contrapos'),
        ('Contrapos','ExFalso'), ('ExFalso','Peirce'), ('DblNeg','Peirce'),
    ]
    COLOR = {'axiom': '#2E4057', 'checkpoint': '#E05C2A'}

    for nid, nd in nodes.items():
        G.add_node(nid, **nd)
    G.add_edges_from(edges)

    pos = {
        'Axioms':    (0, 4),
        'Weakening': (-2, 3),  'Identity':  (0, 3),
        'HypSyll':   (2, 3),
        'DblNeg':    (-1, 2),  'Contrapos': (1, 2),
        'ExFalso':   (0, 1),   'Peirce':    (2, 1),
    }

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0D1117'); ax.set_facecolor('#0D1117')

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=18,
        edge_color='#3A4A5C', width=1.4, connectionstyle='arc3,rad=0.05',
        min_source_margin=35, min_target_margin=35)

    for ntype in ['axiom', 'checkpoint']:
        nl  = [n for n, d in G.nodes(data=True) if d['node_type' if 'node_type' in d else 'type'] == ntype]
        nl  = [n for n, d in G.nodes(data=True) if d.get('type') == ntype]
        sz  = 3500 if ntype == 'checkpoint' else 3000
        nx.draw_networkx_nodes(G, pos, nodelist=nl, ax=ax,
            node_color=COLOR[ntype], node_size=sz, alpha=0.9,
            linewidths=2, edgecolors='#FFFFFF' if ntype == 'axiom' else COLOR[ntype])

    labels = {n: d['label'] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
        font_color='white', font_size=7.5, font_weight='bold')

    legend = [
        mpatches.Patch(color=COLOR['axiom'],      label='Axiom schemas'),
        mpatches.Patch(color=COLOR['checkpoint'],  label='Checkpoint theorem'),
    ]
    ax.legend(handles=legend, loc='lower right', facecolor='#1A2233',
              edgecolor='#3A4A5C', labelcolor='white', fontsize=10)
    ax.set_title('Propositional Logic — Generative Checkpoint Map\n'
                 'Agent builds formulas from scratch via A1/A2/A3 + Modus Ponens',
                 color='white', fontsize=12, fontweight='bold', pad=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"  Graph image -> {out}")


# ── Reward model training ──────────────────────────────────────────────────────
def maybe_train_reward_model(data_dir: Path, models_dir: Path) -> RewardModel:
    rm      = RewardModel()
    rm_path = models_dir / "reward_model_generative"
    if (rm_path.parent / (rm_path.name + '.pt')).exists():
        rm.load(str(rm_path))
        print(f"  Reward model loaded <- {rm_path}.pt")
    else:
        loss = rm.train(str(data_dir / "ratings.json"))
        if rm.trained:
            rm.save(str(rm_path))
            print(f"  Reward model saved -> {rm_path}  (loss={loss:.4f})")
    return rm


# ── Main run ──────────────────────────────────────────────────────────────────
def run(episodes: int, max_steps: int, blend: float,
        data_dir: Path, models_dir: Path):

    print(f"\n{'='*58}")
    print(f"  Math-RL v3 — Generative Propositional Logic")
    print(f"  Episodes: {episodes}  |  Steps: {max_steps}  |  Blend: {blend:.0%}")
    print(f"{'='*58}")

    # 1. Graph image
    print("\n[1/4] Generating checkpoint map...")
    generate_checkpoint_graph(data_dir)

    # 2. Reward model
    print("\n[2/4] Training reward model from ratings...")
    rm = maybe_train_reward_model(data_dir, models_dir)

    # 3. Build reward_fn if model is ready
    reward_fn = None
    if rm.trained:
        def reward_fn(proven_list):
            # Build minimal path record from current proven set
            record = {
                'steps': [{'formula': f, 'rule': '?', 'cp': None} for f in proven_list],
                'checkpoints_hit': [],
            }
            return rm.predict(record)

    # 4. RL training
    print("\n[3/4] Running RL training...")
    agent_path = models_dir / "agent_generative"

    agent = DQNAgent(
        state_size  = GenerativeLogicEnv().state_size,
        action_size = GenerativeLogicEnv().total_actions,
        epsilon_decay = 0.995,
        lr = 5e-4,
    )
    if (agent_path.parent / (agent_path.name + '.pt')).exists():
        try:
            agent.load(str(agent_path))
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * 0.5)
            print(f"  Agent loaded <- {agent_path}.pt")
        except Exception:
            pass

    env = GenerativeLogicEnv(
        max_steps    = max_steps,
        reward_fn    = reward_fn,
        reward_blend = blend,
    )

    _, metrics = train(
        episodes  = episodes,
        max_steps = max_steps,
        verbose   = True,
        env       = env,
        agent     = agent,
    )

    # 5. Export + save
    print("\n[4/4] Exporting paths and saving...")
    agent.save(str(agent_path))
    export_paths(
        env_histories = metrics['ep_histories'],
        data_dir      = str(data_dir),
        top_n         = 30,
        min_checkpoints = 1,
    )
    plot(metrics, out=str(data_dir / "training_generative.png"))

    print(f"\nDone!")
    print(f"  git add data/ models/")
    print(f"  git commit -m 'generative training run'")
    print(f"  git push")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int,   default=800)
    parser.add_argument('--steps',    type=int,   default=300)
    parser.add_argument('--blend',    type=float, default=0.0,
                        help='0=checkpoint reward only, 1=learned reward only')
    parser.add_argument('--data',     default='data')
    parser.add_argument('--models',   default='models')
    args = parser.parse_args()

    data_dir   = ROOT / args.data;   data_dir.mkdir(exist_ok=True)
    models_dir = ROOT / args.models; models_dir.mkdir(exist_ok=True)

    run(args.episodes, args.steps, args.blend, data_dir, models_dir)
