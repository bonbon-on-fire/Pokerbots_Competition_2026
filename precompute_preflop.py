"""
Precompute preflop hand winning probabilities using Monte Carlo simulation.
"""

import pickle
import random
from itertools import combinations
from collections import defaultdict
import time
from tqdm import tqdm

from bitmask_tables import STRAIGHT_MASK_SET, STRAIGHT_MASKS, RANK_TO_INDEX


def _popcount(x: int) -> int:
    """
    Compatibility popcount for older Python versions where int.bit_count() may not exist.
    """
    try:
        return x.bit_count()  # Python 3.8+
    except AttributeError:
        return bin(x).count("1")


class HandEvaluator:
    RANKS = "23456789TJQKA"
    SUITS = "hdcs"
    SUITS_DICT = {"h": 0, "d": 1, "c": 2, "s": 3}

    def best_hand_rank_8(self, cards):
        """
        Returns a lexicographically comparable tuple.
        Higher tuple = stronger hand.
        """

        rank_counts = [0] * 13
        suit_masks = [0, 0, 0, 0]
        rank_mask = 0

        # Build masks
        for c in cards:
            r, s = c[0], c[1]
            ri = RANK_TO_INDEX[r]
            si = self.SUITS_DICT[s]

            rank_counts[ri] += 1
            suit_masks[si] |= 1 << ri
            rank_mask |= 1 << ri

        # ---------- 1. STRAIGHT FLUSH ----------
        for suit in range(4):
            sm = suit_masks[suit]
            if _popcount(sm) >= 5:
                for mask, high in STRAIGHT_MASKS.items():
                    if sm & mask == mask:
                        return [(8, high), 8]

        # ---------- 2. FOUR OF A KIND ----------
        quad = -1
        kicker = -1
        for r in range(12, -1, -1):
            if rank_counts[r] == 4:
                quad = r
            elif rank_counts[r] > 0 and kicker == -1:
                kicker = r
        if quad != -1:
            return [(7, quad, kicker), 7]

        # ---------- 3. FULL HOUSE ----------
        trips = []
        pairs = []
        for r in range(12, -1, -1):
            if rank_counts[r] >= 3:
                trips.append(r)
            elif rank_counts[r] >= 2:
                pairs.append(r)
        if trips:
            trip = trips[0]
            pair = trips[1] if len(trips) > 1 else (pairs[0] if pairs else -1)
            if pair != -1:
                return [(6, trip, pair), 6]

        # ---------- 4. FLUSH ----------
        for suit in range(4):
            sm = suit_masks[suit]
            if _popcount(sm) >= 5:
                kickers = [r for r in range(12, -1, -1) if sm & (1 << r)]
                return [(5, kickers[:5]), 5]

        # ---------- 5. STRAIGHT ----------
        for mask, high in STRAIGHT_MASKS.items():
            if rank_mask & mask == mask:
                return [(4, high), 4]

        # ---------- 6. THREE OF A KIND ----------
        if trips:
            trip = trips[0]
            kickers = [r for r in range(12, -1, -1) if rank_counts[r] == 1]
            return [(3, trip, kickers[:2]), 3]

        # ---------- 7. TWO PAIR ----------
        if len(pairs) >= 2:
            high_pair, low_pair = pairs[:2]
            kickers = [r for r in range(12, -1, -1) if rank_counts[r] == 1]
            kicker = max(kickers) if kickers else -1
            return [(2, high_pair, low_pair, kicker), 2]

        # ---------- 8. ONE PAIR ----------
        if pairs:
            pair = pairs[0]
            kickers = [r for r in range(12, -1, -1) if rank_counts[r] == 1]
            return [(1, pair, kickers[:3]), 1]

        # ---------- 9. HIGH CARD ----------
        kickers = [r for r in range(12, -1, -1) if rank_counts[r]]
        return [(0, kickers[:5]), 0]

    def compare_hands(self, p1_cards, p2_cards):
        """
        Returns:
            1 if Player 1 wins
            0 if Player 2 wins (ties included)
        """

        p1_rank = self.best_hand_rank_8(p1_cards)
        p2_rank = self.best_hand_rank_8(p2_cards)

        if p1_rank[0] > p2_rank[0]:
            return 1
        elif p1_rank[0] < p2_rank[0]:
            return 0
        else:
            return 0.5  # tie

    def two_card_strength(self, cards, board):
        """
        Cheap heuristic for opponent strength.
        Higher = stronger.
        """
        score = 0

        # rank values
        ranks = [RANK_TO_INDEX[c[0]] for c in cards]
        suits = [c[1] for c in cards]

        # High cards
        score += max(ranks) * 2
        score += sum(ranks) * 0.3

        # Pair
        if ranks[0] == ranks[1]:
            score += 20

        # Suited
        if suits[0] == suits[1]:
            score += 5

        # Board interaction
        board_ranks = {c[0] for c in board}
        score += sum(5 for c in cards if c[0] in board_ranks)

        return score

    def choose_opponent_discard_simple(self, opp_cards, board):
        best_score = -1
        best_discard = 0

        for i in range(3):
            kept = [c for j, c in enumerate(opp_cards) if j != i]
            score = self.two_card_strength(kept, board)
            if score > best_score:
                best_score = score
                best_discard = i

        return best_discard


def get_all_preflop_hands():
    """Generate all possible 3-card preflop hands."""
    deck = [r + s for r in HandEvaluator.RANKS for s in HandEvaluator.SUITS]
    hands = list(combinations(deck, 3))
    return hands


def simulate_game(my_cards, opp_cards, board):
    """Simulate a full game and return 1 if my_cards win, 0.5 for tie, 0 for loss."""
    deck = [r + s for r in HandEvaluator.RANKS for s in HandEvaluator.SUITS]
    remaining_deck = [
        c for c in deck if c not in my_cards and c not in opp_cards and c not in board
    ]
    random.shuffle(remaining_deck)

    evaluator = HandEvaluator()
    opp_discard_idx = evaluator.choose_opponent_discard_simple(opp_cards, board)
    opp_kept = [c for i, c in enumerate(opp_cards) if i != opp_discard_idx]

    full_board = remaining_deck[:4] + board + [opp_cards[opp_discard_idx]]

    my_final = my_cards + full_board
    opp_final = opp_kept + full_board

    # print(my_final, opp_final)
    return evaluator.compare_hands(my_final, opp_final)


def compute_preflop_equity_mc(hand, board, num_simulations):
    """Compute preflop equity using Monte Carlo simulation."""
    deck = [r + s for r in HandEvaluator.RANKS for s in HandEvaluator.SUITS]
    remaining_deck = [c for c in deck if c not in hand and c not in board]

    wins = 0
    total = 0

    for _ in range(num_simulations):
        random.shuffle(remaining_deck)
        opp_cards = remaining_deck[:3]
        result = simulate_game(hand, opp_cards, board)
        wins += result
        total += 1

    return wins / total


def main():
    print("Precomputing preflop hand equities using Monte Carlo...")
    hands = get_all_preflop_hands()
    equities = {}

    start_time = time.time()
    for hand in tqdm(hands):
        # for hand in hands:
        # Sort hand for consistent key
        hand_key = tuple(sorted(hand))
        p1 = compute_preflop_equity_mc([hand[0], hand[1]], [hand[2]], 1000)
        p2 = compute_preflop_equity_mc([hand[0], hand[2]], [hand[1]], 1000)
        p3 = compute_preflop_equity_mc([hand[1], hand[2]], [hand[0]], 1000)
        equities[hand_key] = max(p1, p2, p3)

        # print(f"Hand: {hand_key}, Equity: {equities[hand_key]:.4f}")

    end_time = time.time()
    print(f"Precomputation completed in {end_time - start_time:.2f} seconds.")

    # Save to pickle
    with open("preflop_equities_mc.pkl", "wb") as f:
        pickle.dump(equities, f)

    print("Equities saved to preflop_equities_mc.pkl")


if __name__ == "__main__":
    main()
