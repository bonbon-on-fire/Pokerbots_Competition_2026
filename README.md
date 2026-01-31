# Pokerbots Competition 2026

## Project Overview

### What I Built

This repo contains a **competition-ready Pokerbots codebase** (copied from the original competition repository), plus our **final bot** implementation and experiments. The bot’s core is a **fast bitmask hand evaluator** + **Monte Carlo (MC) rollouts** to make discard and betting decisions in a custom poker variant.

### Why I Built It

The variant’s biggest “lever” isn’t just betting — it’s the **discard mechanic** (you temporarily hold 3 private cards and must discard one onto the board). A bot that discards well can create strong board states for itself while denying the opponent good reaction options. The fastest path to a strong bot was:

- build an evaluator that’s _fast enough to simulate a lot_, and
- use simulation to choose discards and avoid “spew” (bad big calls/raise-wars).

## Technical Overview

### System Architecture

- **Engine / skeleton**: Competition harness calls your bot’s `handle_new_round`, `get_action`, `handle_round_over`.
- **Bot policy** (`player.py`): Converts visible cards → ids, runs discard/betting logic, returns an Action.
- **Fast evaluator** (`bitmask_tables.py` + evaluator helpers): Uses rank/suit bitmasks to quickly rank hands.
- **Simulation**: MC rollouts estimate win/tie rate for (a) which card to discard and (b) whether continuing is worth it.

### Key Components

- **Variant rules (the one this repo targets)**

  - You are dealt **3 hole cards**.
  - The “flop” reveals **2 board cards**.
  - Each player **discards 1** of their 3 hole cards **onto the board** (discard order matters because the second discarder can react to what they see).
  - The board is completed to **6 total board cards** (after both discards + remaining community cards).
  - Showdown: each player effectively has **8 cards** (2 private + 6 board) and plays the **best 7-card hand**.

- **Bitmask evaluator**

  - Each card maps to `(rank_idx, suit_idx, rank_bit)`.
  - We maintain:
    - `rank_counts[13]` for pairs/trips/quads detection
    - `rank_mask` (13-bit) for straights
    - `suit_masks[4]` (13-bit each) for flush / straight-flush
  - Output is a comparable tuple like `(category, kicker1, kicker2, ...)` where bigger is better.

- **Monte Carlo discard chooser**

  - For each discard option `d ∈ {0,1,2}`:
    - simulate the rest of the hand many times,
    - estimate win/tie score,
    - pick the discard with highest score.
  - Opponent discarding is approximated with a simple heuristic (can be upgraded with better opponent modeling).

- **Preflop equities (optional speed-up)**
  - We also experimented with **precomputing equity estimates** for starting hand “cores” and saving them to disk.
  - Those values can be loaded quickly (instead of recomputing from scratch every round), which helps when time limits are strict.

## Code in Action: One Round Flow Example

### 1. New round starts

- `handle_new_round` runs.
- We reset per-round trackers (e.g., raise counters) and optionally sample a light “style mode”.

### 2. Pre-discard betting

- `get_action` runs on the preflop street(s).
- Basic logic: don’t do anything fancy yet—avoid massive mistakes and keep stacks intact.

### 3. Flop appears (2 board cards)

- `get_action` sees a new `street` and updates counters.

### 4. Discard phase (the key decision)

- If `DiscardAction` is legal:
  - convert `my_cards` + `board_cards` into ids,
  - run `choose_discard_mc(...)`,
  - return `DiscardAction(best_index)`.

### 5. Post-discard betting + showdown

- After both discards, we use MC-estimated win probability + pot odds to decide:
  - **fold** when clearly behind and facing big pressure,
  - **call/check** as the default,
  - **raise** sparingly (unless we later add stronger strength/texture logic).

## How the Bot Works (High-Level Strategy)

1. **Win the discard game first.**  
   Discard selection is equity-driven using Monte Carlo. That’s the single most important edge in this variant.

2. **Don’t donate chips.**  
   The baseline betting policy is conservative: mostly check/call, fold to big pressure unless the estimated win probability clears pot-odds + margin.

3. **Adapt (lightly) to opponent behavior.**  
   We track simple opponent action frequencies (raise/fold/check-call) and can widen/tighten our continue ranges over time.

4. **Game-theory note we learned after the competition:**  
   Some teams used a “lock-in” strategy: once they were comfortably ahead in bankroll, they would **fold most or all remaining rounds** because the win condition is “finish ahead,” not “maximize margin.” It’s ugly, but it works under that scoring rule.

## Project Structure & File Guide

### Directory Overview

- `player.py`  
  Main bot logic: action selection, discard MC, (optionally) EV / pot-odds gating.
- `bitmask_tables.py`  
  Card-id mappings + straight masks / helpers used by the evaluator.
- `skeleton/`  
  Competition harness: states, actions, runner, networking.

## Current Status

- Final bot is functional and stable.
- Strong discard logic via MC + fast evaluator.
- Betting is intentionally conservative (safe baseline).

## Challenges and How I Solved Them

- **Runtime limits**: MC is expensive. We used a fast evaluator and reduced iterations where possible.
- **Variant complexity**: discard order matters; we built the discard logic so the “try all 3 discards” structure stays reusable.
- **Language ecosystem**: we tried converting the bot to **Java** because languages were scored separately and we suspected fewer Java entries. We ran out of time to finish the port — and it turned out there was basically only **one** Java bot anyway.

## Future Possibilities

- Better opponent range modeling (condition MC samples on opponent actions).
- Board texture features (wet/dry board) to size bets and reduce bluffs on dangerous boards.
- A proper raise ladder / “anti-spew” policy to prevent accidental stack-offs.
- Reinforcement learning / CFR-style offline training (expensive, but possible if you can generate enough data and compute).

## TL;DR

This is a copy of the original repo for the MIT Pokerbots 2026 competition, it involves a poker variant where each player starts with 3 cards and discards 1. Our bot’s core is a fast bitmask hand evaluator plus Monte Carlo rollouts to pick the best discard and make disciplined continue/fold decisions.

---

**Competition Duration**: January 2026  
**Technologies**: Python, C++, CMake, JSON, Git
