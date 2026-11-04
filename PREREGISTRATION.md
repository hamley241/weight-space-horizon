# Preregistration

Written and frozen **before any evaluation was run**. The protocol, the acceptance gate and the
list of outputs below were fixed in advance; nothing in them was changed after seeing results.

## Question

How does the reported return of the three released one-shot networks depend on the evaluation
horizon, and what fraction of episodes is censored by the cap at each horizon?

## Scope, and what cannot be done

The source release includes the three networks written by the model
(`exp5_write_from_scratch/networks/<game>/hand.npz`) but **not the PPO checkpoints** they are
compared against ("results only; their checkpoints are not included"). The sweep therefore covers
the one-shot networks only. PPO enters solely as the authors' reported mean at the 10,000-step cap,
and no PPO number is re-derived here. Any claim requiring PPO at another horizon is out of scope.

## Protocol

- MinAtar 1.0.15, `Environment(game, sticky_action_prob=0.1, difficulty_ramping=True)` — the package
  defaults, matching "the official MinAtar 1.0.15 game".
- `act(a)` takes `a = 0..5` over the full action map `['n','l','u','r','d','f']`, per each prompt's
  own statement. Sticky actions are MinAtar's own: with probability 0.1 the previous action repeats.
- Observation: the prompt's 6-channel map, OR over source channels, flattened as
  `i = (row * 10 + column) * 6 + channel`. Per game:

  | target channel | breakout | asterix | seaquest |
  |---|---|---|---|
  | 0 player | paddle | player | sub_front, sub_back |
  | 1 danger | — | enemy | enemy_fish, enemy_sub, enemy_bullet |
  | 2 reward | brick | gold | enemy_fish, enemy_sub, diver |
  | 3 projectile | ball | — | friendly_bullet |
  | 4 trail | trail | trail | trail, sub_back |
  | 5 gauge | — | — | oxygen_guage, diver_guage |

- Forward pass: `h1 = tanh(x @ W1 + b1)`, `h2 = tanh(h1 @ W2 + b2)`, `logits = h2 @ W3 + b3`; the
  action is sampled from `softmax(logits)`.
- 1,000 episodes per game, seeds fixed and recorded in the code. Each episode is run **once** to a
  maximum of 30,000 steps; the return at horizon C is the cumulative reward up to
  `min(termination, C)`. Horizons are therefore **nested and perfectly paired**, which is the reason
  for a single pass rather than one run per horizon.
- Horizon ladder: 100, 300, 1,000, 3,000, 10,000, 30,000.

## Acceptance gate, declared in advance

For each game, the mean return at horizon 10,000 must agree with the authors' reported mean within
**3 × SE_combined**, where `SE_combined = sqrt(SE_mine² + SE_authors²)`. Reference values:

| game | mean | SE |
|---|---|---|
| breakout | 517.411 | 7.4170 |
| seaquest | 53.110 | 1.3574 |
| asterix | 53.178 | 1.0233 |

A game that fails the gate is reported as a failed reproduction and **its sweep is not
interpreted**. Bit-level reproduction is neither expected nor claimed: the action-sampling RNG
stream cannot be reconstructed from the release, so this is a distributional check on the mean, and
a discrepancy screen rather than a formal equivalence test.

## Outputs, fixed in advance regardless of what the numbers show

1. Mean return per game at each horizon, with standard errors.
2. Censoring fraction at each horizon — the share of episodes cut by the cap rather than ended by
   the game.
3. The survival function of episode length. No Kaplan-Meier estimator is required: episodes are run
   to 30,000, so the only censoring is administrative at 30,000 and is reported directly.
4. Pooled reward per observed step at each horizon, reported **alongside** return and explicitly not
   as a replacement objective. It conditions on endogenous episode length, and in Breakout survival
   creates scoring opportunity.
5. The empirical shape of return versus horizon. Linearity in the cap is a hypothesis to be tested,
   not an assumption.

## Falsifier

If return flattens with horizon well below 10,000 **for all three games**, the reported scores are
not horizon-sensitive and the motivating concern is withdrawn.
