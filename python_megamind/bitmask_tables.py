from __future__ import annotations

# variables for card representation
RANKS = "23456789TJQKA"
SUITS = "cdhs"

CARD_STR_BY_ID = []
CARD_ID_BY_STR = {}

CARD_RANK_IDX = [0] * 52
CARD_SUIT_IDX = [0] * 52
CARD_RANK_BIT = [0] * 52

# precompute all card mappings
for suit_idx, s in enumerate(SUITS):
    for rank_idx, r in enumerate(RANKS):
        card_id = suit_idx * 13 + rank_idx
        card_str = f"{r}{s}"

        CARD_STR_BY_ID.append(card_str)
        CARD_ID_BY_STR[card_str] = card_id

        CARD_RANK_IDX[card_id] = rank_idx
        CARD_SUIT_IDX[card_id] = suit_idx
        CARD_RANK_BIT[card_id] = 1 << rank_idx


def encode_card(card_str):
    """
    Convert a card string like "Ah" into fast integer features.

    Arguments:
    card_str: str, like "Ah", "2c", "Td"

    Returns:
    tuple(rank_idx, suit_idx, rank_bit, card_id) where:
      - rank_idx: 0..12 (2..A)
      - suit_idx: 0..3 (c,d,h,s)
      - rank_bit: 1<<rank_idx
      - card_id: 0..51
    """
    card_id = CARD_ID_BY_STR[card_str]
    return (
        CARD_RANK_IDX[card_id],
        CARD_SUIT_IDX[card_id],
        CARD_RANK_BIT[card_id],
        card_id,
    )


def decode_card(card_id):
    """
    Convert a card id (0..51) back into a readable string like "2c".

    Arguments:
    card_id: int in [0, 51]

    Returns:
    str card representation like "2c", "Ah"
    """
    return CARD_STR_BY_ID[card_id]


STRAIGHT_MASKS = []
STRAIGHT_HIGH_BY_MASK = {}

# normal straights: 23456 up to TJQKA
for high in range(4, 13):  # high rank idx (4=6) ... (12=A)
    mask = 0
    for r in range(high - 4, high + 1):
        mask |= 1 << r
    STRAIGHT_MASKS.append(mask)
    STRAIGHT_HIGH_BY_MASK[mask] = high

# wheel: A2345 -> bits A,2,3,4,5 => idx 12,0,1,2,3
WHEEL_MASK = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
STRAIGHT_MASKS.append(WHEEL_MASK)
STRAIGHT_HIGH_BY_MASK[WHEEL_MASK] = 3  # 5-high


def straight_high(rank_mask):
    """
    Detect whether a 13-bit rank mask contains any straight.

    Arguments:
    rank_mask: int bitmask over ranks (bit i means rank_idx i present)

    Returns:
    int high_rank_idx if a straight exists, else None
    """
    best = None
    for m in STRAIGHT_MASKS:
        if (rank_mask & m) == m:
            high = STRAIGHT_HIGH_BY_MASK[m]
            if best is None or high > best:
                best = high
    return best
