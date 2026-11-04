"""The decoded program versus the network under the FROZEN
horizon protocol (MinAtar 1.0.15, sticky actions 0.1, softmax sampling with the sweep's action RNG
and seeds), decoder built from OBSERVATIONS ONLY, one shared trajectory driven by the network's
sampled action, zero disagreements required; u4/u5 vs internal ball_dir exceptions reported
separately (env internals used only for that report, never by the decoder).
"""
import os, sys
from collections import Counter

import numpy as np
from minatar import Environment

NET = os.path.join(os.path.dirname(__file__), "..", "data", "networks", "breakout.hand.npz")
with np.load(NET, allow_pickle=False) as z:
    W1 = z["actor/dense1/kernel"].astype(float); b1 = z["actor/dense1/bias"].astype(float)
    W2 = z["actor/dense2/kernel"].astype(float); b2 = z["actor/dense2/bias"].astype(float)
    W3 = z["actor/logits/kernel"].astype(float); b3 = z["actor/logits/bias"].astype(float)
G = W1.reshape(10, 10, 6, 32)
BALLMAP = {j: G[:, :, 3, j] / 0.02 for j in range(4)}
QUAD = {(+1, +1): 0, (-1, +1): 1, (+1, -1): 2, (-1, -1): 3}
DIR = {0: (-1, -1), 1: (+1, -1), 2: (+1, +1), 3: (-1, +1)}
CAP = 10_000

def obs(env):
    s = env.state(); ch = env.env.channels
    g = np.zeros((10, 10, 6), bool)
    g[:, :, 0] = s[:, :, ch["paddle"]]; g[:, :, 2] = s[:, :, ch["brick"]]
    g[:, :, 3] = s[:, :, ch["ball"]];   g[:, :, 4] = s[:, :, ch["trail"]]
    return g

def net_probs(g):
    x = g.reshape(600).astype(float)
    h1 = np.tanh(x @ W1 + b1); h2 = np.tanh(h1 @ W2 + b2)
    lg = h2 @ W3 + b3; p = np.exp(lg - lg.max()); return p / p.sum()

def decoder(g):
    """Observation-only decoded program. Returns action, (dc, dr) it used."""
    x = g.reshape(600).astype(float)
    pre = x @ W1 + b1                                    # the direction units, read as a lookup
    dc, dr = int(np.sign(pre[4])), int(np.sign(pre[5]))
    if dc == 0 or dr == 0:
        return None, (dc, dr)
    br, bc = np.argwhere(g[:, :, 3])[0]                  # ball from the observation
    paddle = int(np.argwhere(g[9, :, 0])[0][0])          # paddle from the observation
    delta = BALLMAP[QUAD[(dc, dr)]][br, bc] - paddle
    return (1 if delta < -0.75 else 3 if delta > 0.75 else 0), (dc, dr)

def main(episodes=20):
    states = agree = 0; dis = Counter(); undefined = 0
    dir_total = dir_bad = 0; bad_kinds = Counter(); examples = []
    for ep in range(episodes):
        env = Environment("breakout", sticky_action_prob=0.1, difficulty_ramping=True)
        env.seed(10_000_000 + ep); arng = np.random.RandomState(20_000_000 + ep)   # the sweep's seeds
        env.reset()
        for step in range(1, CAP + 1):
            g = obs(env); e = env.env
            p = net_probs(g)
            a_net = min(int(np.searchsorted(np.cumsum(p), arng.random_sample())), 5)   # the sweep's sampler
            a_dec, (dc, dr) = decoder(g)
            states += 1
            if a_dec is None:
                undefined += 1
            elif a_dec == a_net:
                agree += 1
            else:
                dis[(a_dec, a_net)] += 1
            # semantic report only: do the direction units match the env's internal direction?
            tdc, tdr = DIR[e.ball_dir]
            dir_total += 1
            if (dc, dr) != (tdc, tdr):
                dir_bad += 1
                same_row = e.ball_y == e.last_y; same_col = e.ball_x == e.last_x
                kind = ("same-row frame" if same_row else "same-col frame" if same_col else "diagonal")
                bad_kinds[(kind, f"units=({dc},{dr})", f"true=({tdc},{tdr})")] += 1
                if len(examples) < 5:
                    examples.append(dict(ball=(int(e.ball_y), int(e.ball_x)), trail=(int(e.last_y), int(e.last_x)),
                                         paddle=int(e.pos), units=(dc, dr), true=(tdc, tdr)))
            _, done = env.act(a_net)
            if done:
                break
    print(f"protocol: sticky 0.1, softmax sampling, sweep seeds, cap {CAP}, {episodes} episodes")
    print(f"states {states:,}   decoder == sampled network action: {agree:,} ({agree/max(states-undefined,1):.5%} of defined)   "
          f"disagreements {sum(dis.values())}   decoder-undefined (a direction unit exactly 0) {undefined}")
    for k, v in dis.most_common(5):
        print(f"   disagreement (decoder, net) = {k}: {v}")
    print(f"\ndirection units vs internal ball_dir: {dir_total-dir_bad:,}/{dir_total:,} match ({(dir_total-dir_bad)/dir_total:.4%}); "
          f"exceptions {dir_bad} ({dir_bad/dir_total:.4%})")
    for k, v in bad_kinds.most_common(8):
        print(f"   {k}: {v}")
    for x in examples:
        print("   e.g.", x)

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
