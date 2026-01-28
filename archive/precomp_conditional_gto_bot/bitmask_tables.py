RANKS = "23456789TJQKA"
RANK_TO_INDEX = {r: i for i, r in enumerate(RANKS)}
INDEX_TO_RANK = {i: r for r, i in RANK_TO_INDEX.items()}

# Generate all straight bitmasks
STRAIGHT_MASKS = {}
for start in range(9):  # 0..8 → A2345 handled separately
    mask = 0
    for i in range(5):
        mask |= 1 << (start + i)
    STRAIGHT_MASKS[mask] = start + 4  # high card index

# Wheel straight (A-2-3-4-5)
WHEEL_MASK = (1 << 12) | (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
STRAIGHT_MASKS[WHEEL_MASK] = 3  # 5-high straight

STRAIGHT_MASK_SET = set(STRAIGHT_MASKS.keys())
