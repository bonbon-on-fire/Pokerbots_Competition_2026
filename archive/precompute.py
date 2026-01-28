"""
Precompute exact winning probabilities for poker hands using enumeration.
This script precomputes win probabilities for all possible 2-card starting hands
at pre-flop, and provides a function to compute exact probabilities for later stages.
"""

import pickle
import json
from itertools import combinations
from collections import defaultdict
import time
from tqdm import tqdm

# Copy necessary constants and functions from player.py
RANKS = "23456789TJQKA"
SUITS = "hdcs"
RANK_TO_INDEX = {r: i for i, r in enumerate(RANKS)}
SUITS_DICT = {"h": 0, "d": 1, "c": 2, "s": 3}
# DECK = [r + s for r in RANKS for s in SUITS]
DECK = ["2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "Ts", "Js", "Qs", "Ks", "As"]
PROBS = {}

# Need to import STRAIGHT_MASKS and STRAIGHT_MASK_SET from bitmask_tables
# For now, assume they are available or define them
try:
    from bitmask_tables import STRAIGHT_MASK_SET, STRAIGHT_MASKS
except ImportError:
    # Define basic straight masks if not available
    STRAIGHT_MASKS = {
        0b1111100000000: 12,  # A K Q J T
        0b0111110000000: 11,  # K Q J T 9
        0b0011111000000: 10,  # Q J T 9 8
        0b0001111100000: 9,  # J T 9 8 7
        0b0000111110000: 8,  # T 9 8 7 6
        0b0000011111000: 7,  # 9 8 7 6 5
        0b0000001111100: 6,  # 8 7 6 5 4
        0b0000000111110: 5,  # 7 6 5 4 3
        0b0000000011111: 3,  # 6 5 4 3 2
        0b1000000001111: 3,  # A 5 4 3 2 (wheel)
    }
    STRAIGHT_MASK_SET = set(STRAIGHT_MASKS.keys())


def _popcount(x: int) -> int:
    """
    Compatibility popcount for older Python versions.
    """
    try:
        return x.bit_count()
    except AttributeError:
        return bin(x).count("1")


class HandEvaluator:
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
            si = SUITS_DICT[s]

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
            0.5 if tie
            0 if Player 2 wins
        """
        p1_rank = self.best_hand_rank_8(p1_cards)
        p2_rank = self.best_hand_rank_8(p2_cards)

        # print(p1_rank, p2_rank)

        if p1_rank[0] > p2_rank[0]:
            return [1, p2_rank[1]]
        elif p1_rank[0] < p2_rank[0]:
            return [0, p2_rank[1]]
        else:
            return [0.5, p2_rank[1]]


def compute_probs(cards, prev_index):
    # print(len(cards))
    if len(cards) == 10:
        # print(PROBS)
        # cards = [m1, m2, b1, b2, b3, b4, b5, b6, o1, o2]
        # print(cards)
        my_hand = cards[:2]
        opp_hand = cards[8:10]
        board = cards[2:8]

        my_full_hand = my_hand + board
        opp_full_hand = opp_hand + board

        # print(my_full_hand, opp_full_hand)

        evaluator = HandEvaluator()
        result = evaluator.compare_hands(my_full_hand, opp_full_hand)
        win = result[0]
        index = result[1]
        # print(result)

        my_hand.sort()
        board.sort()
        key = my_hand + board
        key_str = " ".join(key)
        if key_str not in PROBS:
            PROBS[key_str] = [0] * 9
            PROBS[key_str][index] += win
        else:
            PROBS[key_str][index] += win

        # if cards[:-1] == ["2d", "2h", "2c", "2s", "3c", "3d", "3h", "4h", "4d"]:
        #     print(cards)
        #     print(PROBS[key_str])
        return PROBS[key_str]
    else:
        temp_count = [0] * 9

        my_hand = cards[:2]
        board = cards[2:]

        my_hand.sort()
        board.sort()
        key = my_hand + board
        key_str = " ".join(key)

        for i in range(prev_index, len(DECK) - (10 - len(cards)) + 1):
            c = DECK[i]
            if c not in cards:
                cards.append(c)
                result = compute_probs(cards, i + 1)
                if len(cards) < 8:
                    temp_count = [a + b for a, b in zip(temp_count, result)]

                # if key_str == "2d 2h 2c 2s 3c 3d 3h":
                #     print(cards)
                #     print(temp_count)

                # if key_str == "2d 2h 2c 2s 3c 3d 3h 4h":
                #     print(cards)
                #     print(temp_count)

                cards.pop()

        if len(cards) == 8:
            # print("test")
            temp_count = PROBS[key_str]
            print(key_str)
            print(temp_count)

        if (len(cards) >= 5 and len(cards) <= 8) or len(cards) == 3:
            # print(key_str)

            if key_str not in PROBS:
                PROBS[key_str] = temp_count
            else:
                current = PROBS[key_str]
                new = [a + b for a, b in zip(temp_count, current)]
                PROBS[key_str] = new

        return temp_count


def save_probs(probs, filename="preflop_probs.pkl"):
    """
    Save the precomputed probabilities to a file.
    """
    with open(filename, "wb") as f:
        pickle.dump(probs, f, protocol=4)
    print(f"Saved probabilities to {filename}")


def load_probs(filename="preflop_probs.pkl"):
    """
    Load precomputed probabilities from a file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def convert_pickle_protocol(filename, new_protocol=4):
    """
    Load a pickle file and resave it with a different protocol.
    """
    with open(filename, "rb") as f:
        data = pickle.load(f)
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=new_protocol)
    print(f"Converted {filename} to pickle protocol {new_protocol}")


if __name__ == "__main__":
    # Convert existing pickle file to protocol 4
    convert_pickle_protocol("preflop_equities_mc.pkl", 4)

    # compute_probs([], 0)
    # save_probs(PROBS)
