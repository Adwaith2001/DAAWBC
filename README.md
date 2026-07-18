# src4 — Complete Time-Slot Package

This is the full src4. Contains **two parallel time-slot implementations**
(mine and peer A's) so you can run either or both without further
back-and-forth, plus the peer's pre-committed protocol and an evaluator
with formal random baseline.

## What the 5-episode diagnostics actually showed

Three rounds of patches to my src4 (wrapper design):

| Round | Patches applied | Ep1 clicks | Ep5 clicks | H_thr | H_res |
|---|---|---|---|---|---|
| v1 | UPDATE_EVERY=25 (broken default) | 32 | 22 | 3.93 | 2.40 |
| v2 | UPDATE_EVERY=5 + CLIP_GRAD=10 + adv-norm | 32 | 22 | 3.93 | 2.40 |
| v3 | + 3 peer-review state-feature bug fixes + slot=5000 | 41 | 44 | 3.93 | 2.40 |

**H_thr and H_res are frozen at the uniform-distribution maximum in
every run.** The clicks rising from 22 to 44 across rounds is purely
from more impressions per episode (denser data after slot_size 2000→5000),
not from any policy learning.

This is now strong evidence for the negative outcome that the protocol
§6 pre-commits as publishable:

> If learned policy ≈ random baseline (no entropy movement / flat
> CTR-on-wins) → even with a dense aggregated reward, on-policy MARL
> in this shared-auction setting does not learn. Combined with src3's
> characterised failures this is the strong negative result: the
> barrier is not solely sparsity but the multi-agent shared-auction
> structure. Equally publishable; pre-committed as such here.

## Why I'm including two designs

The peer review concluded MY wrapper design was more defensible
architecturally. Three patch rounds later, mine still doesn't move
entropies. **Before accepting that as the final negative finding, it's
worth running A's design** (single-λ action, full-episode returns,
return normalization at the trainer level). A's design has fundamentally
different optimization dynamics — if mine is stuck because of update
density / batch size / two-head gradient interaction, A's might not
be. If A's ALSO doesn't move, that's much stronger evidence the negative
finding is real and not specific to my design choices.

## Files

```
src4/
├── analytical_bid_base.py             ← shared bid calibration (used by both)
├── policy_network_strategic.py        ← 2-head (thr, res) policy [mine]
├── policy_network_timeslot.py         ← 1-head (λ) policy        [A's]
├── train_strategic_ac_timeslot.py     ← my trainer (protocol-locked)
├── train_timeslot_ac.py               ← A's trainer (+ --max-imps)
├── evaluate_strategic_timeslot.py     ← evaluator + RANDOM_UNIFORM baseline
├── __init__.py
└── simulator/
    ├── multi_environment_strategic.py ← src3 inner env (wrapped by mine)
    ├── multi_environment_timeslot.py  ← my wrapper (peer-reviewed)
    ├── time_slot_environment.py       ← A's standalone env (+ --max-imps)
    └── __init__.py

EVALUATION_PROTOCOL_src4.md            ← peer's pre-committed protocol
README.md                              ← this file
```

## How to drop in

Extract the zip at `D:\Research Methodology\DAAWBC\dynamic_ad_allocation\`.
The `src4\` folder in the zip merges with your existing `src4\` folder
— some files overwrite, three are new (`train_timeslot_ac.py`,
`policy_network_timeslot.py`, `time_slot_environment.py`). The protocol
lands at project root.

## Run order — what I recommend

### 1. Try A's design first (cheap, ~20 min)

```cmd
cd src4
python train_timeslot_ac.py --episodes 5 --seeds 0 --max-imps 50000
```

Watch `H_lambda`. Max entropy is log(21) ≈ 3.04 (NOT 3.93 like mine).
**If H_lambda drops below 3.0 by ep 5**, A's design works — commit to
A's full run instead of mine. **If H_lambda stays at 3.04**, both
designs have the same issue → the negative finding is robust to design
choice.

### 2. If A's also doesn't move, commit to the locked-protocol run

This is the official 30-ep × 3-seed run that produces the publishable
result (positive or negative):

```cmd
:: Condition F (default): isolated + sampled + both
python train_strategic_ac_timeslot.py

:: ~15 hours wallclock at slot_size=5000.
:: Outputs to outputs_2/ac_timeslot_isolated_both_sampled/
```

Then run the other conditions per protocol §5 (E, G, H):

```cmd
:: Condition E: market_baseline
python train_strategic_ac_timeslot.py --auction-mode market_baseline

:: Condition G: residual_only + expected_click (most favourable)
python train_strategic_ac_timeslot.py --policy-mode residual_only --reward-mode expected_click

:: Condition H: slot_size sweep (data-horizon analog)
python train_strategic_ac_timeslot.py --slot-size 2000
:: (slot_size=5000 already covered by Condition F)
```

### 3. Evaluate

```cmd
python evaluate_strategic_timeslot.py --auction-mode isolated
```

This runs 5 baselines per seed × 3 seeds × 5 eval-episodes per method:
- FIXED_BID (fixed $50)
- LINEAR_PCTR (alpha auto-tuned)
- LIU2020_OAA (zero-residual)
- **RANDOM_UNIFORM** (new — per peer item 4 / protocol §2)
- RL_AC_TIMESLOT (your trained policies)

Output: `outputs_2/evaluation_timeslot/per_episode.csv` and
`evaluation_summary.csv`.

## What this delivers for the paper either way

**Positive case:** RL_AC_TIMESLOT > RANDOM_UNIFORM with entropy movement
across training → "learnability recovered under temporal aggregation,"
exactly as protocol §6 pre-commits.

**Negative case:** RL_AC_TIMESLOT ≈ RANDOM_UNIFORM with H frozen across
training → "even dense aggregated reward in shared-auction MARL does
not learn on-policy; the barrier is the multi-agent structure, not the
sparsity alone." Also publishable, also pre-committed.

The protocol's §6 pre-commitment is the reason either result is honest.
The diagnostic data already strongly suggests we're heading toward the
negative outcome on my design. Running A's first tells us whether the
finding is design-specific or fundamental.

## Verification after extraction

```cmd
findstr /n "PEER REVIEW PATCHES" src4\simulator\multi_environment_timeslot.py
findstr /n "max-imps" src4\train_timeslot_ac.py
findstr /n "RANDOM_UNIFORM" src4\evaluate_strategic_timeslot.py
```

You should get matches on all three. If yes, the patched files are
in place.

## Honest note from me (instance B)

I built the wrapper design twice and patched it three times. Each time
I was confident the next patch would unblock entropy. Each time it
didn't. The peer caught real bugs I missed; fixing them moved click
density but not policy behavior. At this point the rational move is
to either (a) accept the negative finding as protocol §6 allows, or
(b) try a fundamentally different optimization setup (A's design) to
make sure the negative finding isn't an artifact of my particular
trainer.

I'm not going to suggest more patches to my trainer. If A's design
also doesn't move, the answer is the negative-result paper, and that's
not a failure — it's exactly what the protocol was written to enable
publishing honestly.
