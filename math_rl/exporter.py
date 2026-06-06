"""
exporter.py — Packages generative env history into pending_paths.json.

Path format (generative):
{
  "id":             "gen_proplogic_v1_20240101",
  "graph":          "Propositional Logic (Generative)",
  "type":           "generative",
  "steps": [
    {"rule": "A1", "args": ["p","q"],       "formula": "(p→(q→p))",  "cp": null},
    {"rule": "MP", "args": ["(p→(q→p))", ...], "formula": "(p→p)", "cp": "T_identity"}
  ],
  "checkpoints_hit": ["T_identity"],
  "n_checkpoints":  1,
  "total_reward":   42.5,
  "episode":        15,
  "source":         "agent_v1",
  "timestamp":      "2024-01-01T00:00:00"
}
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


def export_paths(
    env_histories:  List[Dict],   # list of env.history dicts per episode
    graph_name:     str   = "Propositional Logic (Generative)",
    data_dir:       str   = "data",
    top_n:          int   = 20,
    agent_version:  str   = "v1",
    min_checkpoints:int   = 1,    # only export if at least this many CPs hit
) -> int:
    """
    env_histories: list of dicts with keys:
        history      (List[dict])  — env.history after an episode
        checkpoints  (set/list)    — env.checkpoints_hit
        total_reward (float)
        episode      (int)
    Returns: number of paths exported.
    """
    data_dir   = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    queue_file = data_dir / "pending_paths.json"

    existing: List[Dict] = []
    if queue_file.exists():
        existing = json.loads(queue_file.read_text())

    # Deduplicate by checkpoint signature + path length
    existing_sigs = {
        (tuple(sorted(p.get('checkpoints_hit', []))), len(p.get('steps', [])))
        for p in existing if p.get('type') == 'generative'
    }

    # Sort by reward, filter by min checkpoints
    candidates = sorted(
        [h for h in env_histories if len(h.get('checkpoints', [])) >= min_checkpoints],
        key=lambda h: -h['total_reward']
    )

    added = 0
    for h in candidates:
        if added >= top_n:
            break

        cps  = list(h.get('checkpoints', []))
        sig  = (tuple(sorted(cps)), len(h.get('history', [])))
        if sig in existing_sigs:
            continue

        # Build clean step list from history
        steps = []
        for entry in h.get('history', []):
            if entry.get('derived') is None:
                continue   # skip failed actions
            steps.append({
                'rule':    entry.get('rule', '?'),
                'args':    entry.get('args', []),
                'formula': entry['derived'],
                'cp':      entry.get('cp'),
            })

        if not steps:
            continue

        pid = (f"gen_{graph_name.replace(' ','_').replace('(','').replace(')','')}"
               f"_{agent_version}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}")

        record = {
            'id':              pid,
            'graph':           graph_name,
            'type':            'generative',
            'steps':           steps,
            'checkpoints_hit': cps,
            'n_checkpoints':   len(cps),
            'total_reward':    round(float(h['total_reward']), 2),
            'episode':         h.get('episode', 0),
            'source':          f'agent_{agent_version}',
            'timestamp':       datetime.utcnow().isoformat(),
        }
        existing.append(record)
        existing_sigs.add(sig)
        added += 1

    queue_file.write_text(json.dumps(existing, indent=2))
    print(f"Exported {added} new paths → {queue_file}  (total: {len(existing)})")
    return added
