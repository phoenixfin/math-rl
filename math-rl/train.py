"""
Training loop for the Math-RL propositional logic agent.

Runs Double-DQN on PropLogicEnv and plots:
  1. Episode reward (smoothed)
  2. Checkpoints discovered per episode
  3. Training loss
  4. Checkpoint discovery timeline
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from typing import Dict, List, Tuple

from env   import PropLogicEnv, CHECKPOINTS, CHECKPOINT_REWARDS, NAMES, FORMULAS
from agent import DQNAgent


# ── Training ─────────────────────────────────────────────────────────────────
def train(
    episodes:  int = 600,
    max_steps: int = 120,
    save_path: str = None,
    verbose:   bool = True,
) -> Tuple[DQNAgent, dict]:

    env   = PropLogicEnv(max_steps=max_steps)
    agent = DQNAgent(
        state_size  = env.observation_space.shape[0],
        action_size = env.action_space.n,
    )

    metrics = {
        'rewards':     [],
        'checkpoints': [],
        'steps':       [],
        'losses':      [],
        'discovery':   {},   # cp → first episode discovered
        'proven_sets': [],   # final proven set each episode
    }

    best_checkpoints = 0

    if verbose:
        print("=" * 60)
        print("  Math-RL: Propositional Logic Theorem Discovery")
        print("=" * 60)
        print(f"  Theorems  : {env.n}")
        print(f"  Checkpoints: {sorted(CHECKPOINTS)}")
        print(f"  Episodes  : {episodes}  |  Max steps: {max_steps}")
        print("=" * 60)

    for ep in range(episodes):
        state, _       = env.reset()
        total_reward   = 0.0
        ep_losses      = []

        for _ in range(max_steps):
            valid  = env.get_valid_actions()
            action = agent.act(state, valid_actions=valid)

            next_state, reward, done, truncated, info = env.step(action)
            agent.remember(state, action, reward, next_state, done or truncated)

            loss = agent.replay()
            if loss:
                ep_losses.append(loss)

            state        = next_state
            total_reward += reward
            if done or truncated:
                break

        # Record metrics
        n_cp = len(env.checkpoints_hit)
        metrics['rewards'].append(total_reward)
        metrics['checkpoints'].append(n_cp)
        metrics['steps'].append(env.steps)
        metrics['proven_sets'].append(frozenset(env.proven))
        if ep_losses:
            metrics['losses'].append(np.mean(ep_losses))

        # First-discovery tracking
        for cp in env.checkpoints_hit:
            if cp not in metrics['discovery']:
                metrics['discovery'][cp] = ep + 1

        # Save best model
        if n_cp > best_checkpoints:
            best_checkpoints = n_cp
            if save_path:
                agent.save(save_path)

        # Console log every 50 episodes
        if verbose and (ep + 1) % 50 == 0:
            w = min(50, ep + 1)
            avg_r  = np.mean(metrics['rewards'][-w:])
            avg_cp = np.mean(metrics['checkpoints'][-w:])
            avg_st = np.mean(metrics['steps'][-w:])
            print(f"  Ep {ep+1:4d} | Reward {avg_r:7.2f} | "
                  f"CPs {avg_cp:.1f}/{len(CHECKPOINTS)} | "
                  f"Steps {avg_st:.0f} | ε {agent.epsilon:.3f}")

    if verbose:
        print("\n  First checkpoint discoveries:")
        for cp in sorted(metrics['discovery'], key=metrics['discovery'].get):
            ep_ = metrics['discovery'][cp]
            print(f"    {cp} ({NAMES[cp]}): episode {ep_}")
        undiscovered = CHECKPOINTS - set(metrics['discovery'])
        if undiscovered:
            print(f"    Not yet discovered: {undiscovered}")

    return agent, metrics


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(agent: DQNAgent, episodes: int = 20) -> dict:
    """Run agent greedily (ε=0) and report performance."""
    saved_eps   = agent.epsilon
    agent.epsilon = 0.0

    env    = PropLogicEnv(max_steps=200)
    solves = 0
    cp_counts = []
    paths  = []

    for _ in range(episodes):
        state, _ = env.reset()
        path     = []
        for _ in range(200):
            valid  = env.get_valid_actions()
            action = agent.act(state, valid_actions=valid)
            next_state, _, done, truncated, info = env.step(action)
            if info['derived']:
                path.append(info['theorem'])
            state = next_state
            if done or truncated:
                break
        solves    += int(env.checkpoints_hit == CHECKPOINTS)
        cp_counts.append(len(env.checkpoints_hit))
        paths.append(path)

    agent.epsilon = saved_eps
    return {
        'solve_rate': solves / episodes,
        'avg_cps':    np.mean(cp_counts),
        'best_path':  paths[np.argmax(cp_counts)],
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(metrics: dict, out: str = '/mnt/user-data/outputs/training_results.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor('#0D1117')
    for ax in axes.flat:
        ax.set_facecolor('#141B24')
        ax.tick_params(colors='#AABBCC')
        ax.spines[:].set_color('#2A3A4A')
        ax.title.set_color('white')
        ax.xaxis.label.set_color('#AABBCC')
        ax.yaxis.label.set_color('#AABBCC')

    fig.suptitle('Math-RL: Propositional Logic Theorem Discovery\n'
                 'Double DQN — Training Metrics',
                 color='white', fontsize=14, fontweight='bold')

    W = 30   # smoothing window

    def smooth(x):
        if len(x) < W:
            return x
        return np.convolve(x, np.ones(W)/W, 'valid')

    episodes = list(range(1, len(metrics['rewards']) + 1))

    # ── 1. Reward ─────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(episodes, metrics['rewards'], alpha=0.2, color='#4A9EFF', linewidth=0.8)
    sm = smooth(metrics['rewards'])
    ax.plot(range(W, len(metrics['rewards']) + 1), sm, color='#4A9EFF', linewidth=2)
    ax.set_title('Episode Reward')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.15)

    # ── 2. Checkpoints ────────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(episodes, metrics['checkpoints'], alpha=0.2, color='#F5A623', linewidth=0.8)
    sm = smooth(metrics['checkpoints'])
    ax.plot(range(W, len(metrics['checkpoints']) + 1), sm, color='#F5A623', linewidth=2)
    ax.axhline(len(CHECKPOINTS), color='#E05C2A', linestyle='--', linewidth=1.5,
               label=f'Max ({len(CHECKPOINTS)})')
    ax.set_title('Checkpoints per Episode')
    ax.set_xlabel('Episode')
    ax.set_ylim(0, len(CHECKPOINTS) + 0.5)
    ax.legend(facecolor='#141B24', edgecolor='#2A3A4A', labelcolor='white')
    ax.grid(True, alpha=0.15)

    # ── 3. Loss ───────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    if metrics['losses']:
        ax.plot(metrics['losses'], alpha=0.6, color='#E05C5C', linewidth=1)
        if len(metrics['losses']) >= W:
            sm = smooth(metrics['losses'])
            ax.plot(range(W-1, len(metrics['losses'])), sm, color='#FF7C7C', linewidth=2)
        ax.set_yscale('log')
    ax.set_title('Training Loss (log scale)')
    ax.set_xlabel('Episode')
    ax.grid(True, alpha=0.15)

    # ── 4. Discovery timeline ─────────────────────────────────────────────────
    ax = axes[1, 1]
    disc = metrics['discovery']
    if disc:
        items = sorted(disc.items(), key=lambda x: x[1])
        names  = [f"{t}\n({NAMES[t]})" for t, _ in items]
        values = [ep for _, ep in items]
        colors = ['#E05C2A' if t in CHECKPOINTS else '#4A7C6F' for t, _ in items]
        bars   = ax.barh(names, values, color=colors, alpha=0.85, height=0.6)
        ax.set_title('First Discovery — Episode #')
        ax.set_xlabel('Episode')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.15, axis='x')
        for bar, val in zip(bars, values):
            ax.text(val + 3, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nPlot saved → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    agent, metrics = train(episodes=600, max_steps=120)

    plot(metrics)

    print("\n── Evaluation (greedy, 20 episodes) ──")
    result = evaluate(agent, episodes=20)
    print(f"  Solve rate  : {result['solve_rate']*100:.0f}%")
    print(f"  Avg CPs hit : {result['avg_cps']:.1f}/{len(CHECKPOINTS)}")
    print(f"  Best path   : {' → '.join(result['best_path'])}")
