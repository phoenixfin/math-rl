"""
train.py — Training loop for GenerativeLogicEnv.
"""
import os, sys
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import warnings; warnings.filterwarnings('ignore')

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))
from math_rl.env   import GenerativeLogicEnv
from math_rl.agent import DQNAgent
from math_rl.formula import CHECKPOINTS_DEF, canonicalize


def train(
    episodes:    int                = 800,
    max_steps:   int                = 300,
    pool_size:   int                = 5,
    verbose:     bool               = True,
    log_file:    str                = None,
    env:         GenerativeLogicEnv = None,
    agent:       DQNAgent           = None,
) -> Tuple[DQNAgent, dict]:

    if env is None:
        env = GenerativeLogicEnv(max_steps=max_steps, pool_size=pool_size)
    if agent is None:
        agent = DQNAgent(
            state_size    = env.state_size,
            action_size   = env.total_actions,
            epsilon_decay = 0.995,
            lr            = 5e-4,
        )

    metrics = {
        'rewards':        [],
        'checkpoints':    [],
        'proven_counts':  [],
        'losses':         [],
        'discovery':      {},     # cp_id → first episode
        'best_formulas':  [],     # novel formulas discovered
        'ep_histories':   [],     # for exporter
    }

    best_cps = 0

    def log(msg):
        if verbose: print(msg, flush=True)
        if log_file:
            with open(log_file, 'a') as f: f.write(msg + '\n')

    log("=" * 60)
    log("  Math-RL — Generative Propositional Logic")
    log(f"  Action space: {env.total_actions}  |  State size: {env.state_size}")
    log(f"  Checkpoints:  {len(CHECKPOINTS_DEF)}")
    log("=" * 60)

    for ep in range(episodes):
        state, _     = env.reset()
        total_reward = 0.0
        ep_losses    = []
        ep_formulas  = []

        for _ in range(max_steps):
            valid  = env.get_valid_actions()
            action = agent.act(state, valid_actions=valid if valid else None)
            next_state, reward, done, truncated, info = env.step(action)

            agent.remember(state, action, reward, next_state, done or truncated)
            loss = agent.replay()
            if loss: ep_losses.append(loss)

            if info.get('derived'):
                ep_formulas.append(info['derived'])
            if info.get('cp'):
                cp = info['cp']
                if cp not in metrics['discovery']:
                    metrics['discovery'][cp] = ep + 1
                    log(f"\n  *** CHECKPOINT: {CHECKPOINTS_DEF[cp]['name']} at ep {ep+1} ***")
                    log(f"      Formula: {info['derived']}")

            state        = next_state
            total_reward += reward
            if done or truncated: break

        metrics['rewards'].append(total_reward)
        metrics['checkpoints'].append(len(env.checkpoints_hit))
        metrics['proven_counts'].append(len(env.proven))
        if ep_losses: metrics['losses'].append(np.mean(ep_losses))
        if ep_formulas: metrics['best_formulas'].extend(ep_formulas[-3:])

        # Store for exporter
        metrics['ep_histories'].append({
            'history':      list(env.history),
            'checkpoints':  list(env.checkpoints_hit),
            'total_reward': total_reward,
            'episode':      ep + 1,
        })

        if len(env.checkpoints_hit) > best_cps:
            best_cps = len(env.checkpoints_hit)

        if verbose and (ep + 1) % 100 == 0:
            w     = min(100, ep + 1)
            avg_r = np.mean(metrics['rewards'][-w:])
            avg_c = np.mean(metrics['checkpoints'][-w:])
            avg_p = np.mean(metrics['proven_counts'][-w:])
            log(f"  Ep {ep+1:4d} | Reward {avg_r:7.1f} | "
                f"CPs {avg_c:.2f}/{len(CHECKPOINTS_DEF)} | "
                f"Proven ~{avg_p:.0f} | ε {agent.epsilon:.3f}")

    if verbose:
        log("\n  Checkpoint discoveries:")
        for cp_id, ep_ in sorted(metrics['discovery'].items(), key=lambda x: x[1]):
            log(f"    {CHECKPOINTS_DEF[cp_id]['name']}: ep {ep_}")
        undiscovered = set(CHECKPOINTS_DEF) - set(metrics['discovery'])
        if undiscovered:
            names = [CHECKPOINTS_DEF[c]['name'] for c in undiscovered]
            log(f"    Not discovered: {names}")

    return agent, metrics


def plot(metrics: dict, out: str = 'data/training_generative.png'):
    Path(out).parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor('#0D1117')
    for ax in axes.flat:
        ax.set_facecolor('#141B24'); ax.tick_params(colors='#AABBCC')
        ax.spines[:].set_color('#2A3A4A'); ax.title.set_color('white')
        ax.xaxis.label.set_color('#AABBCC'); ax.yaxis.label.set_color('#AABBCC')
    fig.suptitle('Math-RL: Generative Theorem Discovery\nProof-by-construction from axioms',
                 color='white', fontsize=13, fontweight='bold')

    W  = max(1, len(metrics['rewards']) // 20)
    sm = lambda x: np.convolve(x, np.ones(W)/W, 'valid') if len(x) >= W else x
    N  = len(metrics['rewards'])
    eps = list(range(1, N+1))

    # Reward
    ax = axes[0,0]
    ax.plot(eps, metrics['rewards'], alpha=0.15, color='#4A9EFF', lw=0.8)
    ax.plot(range(W, N+1), sm(metrics['rewards']), color='#4A9EFF', lw=2)
    ax.set_title('Episode Reward'); ax.set_xlabel('Episode'); ax.grid(True, alpha=0.15)

    # Checkpoints
    ax = axes[0,1]
    n_cp = len(CHECKPOINTS_DEF)
    ax.plot(eps, metrics['checkpoints'], alpha=0.15, color='#F5A623', lw=0.8)
    ax.plot(range(W, N+1), sm(metrics['checkpoints']), color='#F5A623', lw=2)
    ax.axhline(n_cp, color='#E05C2A', ls='--', lw=1.5, label=f'Max ({n_cp})')
    ax.set_title('Checkpoints per Episode'); ax.set_xlabel('Episode')
    ax.set_ylim(0, n_cp + 0.5)
    ax.legend(facecolor='#141B24', edgecolor='#2A3A4A', labelcolor='white')
    ax.grid(True, alpha=0.15)

    # Proven formulas per episode
    ax = axes[1,0]
    ax.plot(eps, metrics['proven_counts'], alpha=0.15, color='#22C55E', lw=0.8)
    ax.plot(range(W, N+1), sm(metrics['proven_counts']), color='#22C55E', lw=2)
    ax.set_title('Formulas Proven per Episode'); ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.15)

    # Discovery timeline
    ax = axes[1,1]
    disc = metrics['discovery']
    if disc:
        items  = sorted(disc.items(), key=lambda x: x[1])
        labels = [CHECKPOINTS_DEF[t]['name'] for t, _ in items]
        vals   = [ep for _, ep in items]
        bars   = ax.barh(labels, vals, color='#E05C2A', alpha=0.85, height=0.6)
        ax.set_title('First Discovery — Episode'); ax.invert_yaxis()
        ax.grid(True, alpha=0.15, axis='x')
        for bar, val in zip(bars, vals):
            ax.text(val+2, bar.get_y()+bar.get_height()/2,
                    str(val), va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved → {out}")
    return out


if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)
    agent, metrics = train(episodes=800, max_steps=300)
    plot(metrics)
