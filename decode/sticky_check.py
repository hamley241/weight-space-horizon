"""Paired version: the sweep's exact env/action seed pairs (10M+ep, 20M+ep), sticky OFF, cap 10,000,
first 200 episodes. Then the same 200 episodes' pre-cap terminations from the sweep's own records."""
import json, os, sys, numpy as np
from minatar import Environment
sys.path.insert(0, ".")
from behavioural_check import obs, net_probs
lost = 0
for ep in range(200):
    env = Environment("breakout", sticky_action_prob=0.0, difficulty_ramping=True)
    env.seed(10_000_000 + ep); arng = np.random.RandomState(20_000_000 + ep); env.reset()
    for step in range(1, 10_001):
        p = net_probs(obs(env)); a = min(int(np.searchsorted(np.cumsum(p), arng.random_sample())), 5)
        _, done = env.act(a)
        if done: lost += 1; break
recs = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "breakout.episodes.json")))["records"][:200]
sticky_lost = sum(1 for r in recs if r["length"] < 10_000)
print(f"same 200 seed pairs, sticky OFF: {lost}/200 lost before 10,000 (2,000,000 steps)")
print(f"same 200 seed pairs, sticky 0.1 (sweep records): {sticky_lost}/200 lost before 10,000 ({sticky_lost/2:.1f}%)")

