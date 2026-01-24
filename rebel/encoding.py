"""
ReBeL-style state encoding for the 3-card poker variant.

Card ids 0..51 (SUITS cdhs × RANKS 23456789TJQKA). 3-card hand index 0..22099 = C(52,3).

Query format (44,226 floats total), matching epoch390.torchscript:
  1. Active player ID (1)
  2. Traverser (1)
  3. Last action one-hot (15): 0=fold, 1=call/check, 2-11=bet sizes, 12-14=discard
  4. Board cards (6): 0-51 or -1
  5. Discard choices (2): per-player discard index 0/1/2 or -1
  6. Street (1): 0=preflop, 1=flop, 2=turn, 3=river
  7. Beliefs player 0 (22,100)
  8. Beliefs player 1 (22,100)
"""
from __future__ import annotations

from itertools import combinations
import math

RANKS = "23456789TJQKA"
SUITS = "cdhs"

CARD_STR_BY_ID: list[str] = []
CARD_ID_BY_STR: dict[str, int] = {}

for suit_idx, s in enumerate(SUITS):
    for rank_idx, r in enumerate(RANKS):
        card_id = suit_idx * 13 + rank_idx
        card_str = f"{r}{s}"
        CARD_STR_BY_ID.append(card_str)
        CARD_ID_BY_STR[card_str] = card_id

NUM_THREE_CARD_HANDS = 22_100  # C(52, 3)
QUERY_DIM = 44_226  # 1+1+15+6+2+1+22100+22100

# Precompute 3-card hand <-> index mappings
_COMBS = list(combinations(range(52), 3))
_HAND3_TO_IDX: dict[tuple[int, int, int], int] = {t: i for i, t in enumerate(_COMBS)}
_IDX_TO_HAND3: tuple[tuple[int, int, int], ...] = tuple(_COMBS)

# Uniform beliefs when unknown
_UNIFORM_BELIEF: list[float] = [1.0 / NUM_THREE_CARD_HANDS] * NUM_THREE_CARD_HANDS


def card_ids(card_strs: list[str]) -> list[int]:
    """Convert card strings like ['Ah','2c'] to sorted list of card ids 0..51."""
    return sorted(CARD_ID_BY_STR[s] for s in card_strs if s in CARD_ID_BY_STR)


def hand3_to_index(c0: int, c1: int, c2: int) -> int:
    """Map three card ids (order-independent) to index in [0, 22100)."""
    t = tuple(sorted((c0, c1, c2)))
    return _HAND3_TO_IDX[t]


def index_to_hand3(idx: int) -> tuple[int, int, int]:
    """Map index in [0, 22100) to sorted triple of card ids."""
    return _IDX_TO_HAND3[idx]


def nCk(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def point_mass_belief(hand_idx: int) -> list[float]:
    """Belief that player holds exactly the given 3-card hand (index 0..22099)."""
    b = [0.0] * NUM_THREE_CARD_HANDS
    if 0 <= hand_idx < NUM_THREE_CARD_HANDS:
        b[hand_idx] = 1.0
    return b


def game_street_to_model_street(street: int) -> float:
    """Map game streets 0,2,3,4,5,6 -> model 0=preflop, 1=flop, 2=turn, 3=river."""
    if street == 0 or street in (2, 3):
        return 0.0  # preflop + discard phases
    if street == 4:
        return 1.0  # flop
    if street == 5:
        return 2.0  # turn
    if street == 6:
        return 3.0  # river
    return 0.0


def build_query(
    player_id: int,
    traverser: int,
    last_action: int,
    board_cards: list[int],
    discard_choice_0: int,
    discard_choice_1: int,
    street: int,
    beliefs_player0: list[float],
    beliefs_player1: list[float],
) -> list[float]:
    """
    Build ReBeL query vector of length 44,226.

    - last_action: 0=fold, 1=call/check, 2-11=bet sizes, 12-14=discard choices.
    - board_cards: card ids 0-51; up to 6, use -1 for absent slots.
    - discard_choice_0/1: 0/1/2 = which of initial 3 cards discarded, or -1.
    - street: game street 0,2,3,4,5,6 (mapped internally to model 0..3).
    - beliefs_player0/1: 22,100 floats each, normalized.
    """
    q = [0.0] * QUERY_DIM
    idx = 0

    q[idx] = float(player_id)
    idx += 1
    q[idx] = float(traverser)
    idx += 1

    for a in range(15):
        q[idx] = 1.0 if a == last_action else 0.0
        idx += 1

    for i in range(6):
        q[idx] = float(board_cards[i]) if i < len(board_cards) else -1.0
        idx += 1

    q[idx] = float(discard_choice_0)
    idx += 1
    q[idx] = float(discard_choice_1)
    idx += 1

    q[idx] = game_street_to_model_street(street)
    idx += 1

    for i in range(NUM_THREE_CARD_HANDS):
        q[idx] = beliefs_player0[i] if i < len(beliefs_player0) else _UNIFORM_BELIEF[i]
        idx += 1
    for i in range(NUM_THREE_CARD_HANDS):
        q[idx] = beliefs_player1[i] if i < len(beliefs_player1) else _UNIFORM_BELIEF[i]
        idx += 1

    assert idx == QUERY_DIM
    return q
