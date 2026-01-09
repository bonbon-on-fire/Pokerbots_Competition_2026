from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random
import itertools


RANKS = "23456789TJQKA"
RANK_MAP = {r: i for i, r in enumerate(RANKS)}


def card_rank(card):
    return RANK_MAP[card[0]]


def card_suit(card):
    return card[1]


def evaluate_strength(cards, board):
    """
    Cheap heuristic hand strength estimator.
    Returns a number in [0, 1].
    """
    all_cards = cards + board
    ranks = [card_rank(c) for c in all_cards]
    suits = [card_suit(c) for c in all_cards]

    rank_counts = {r: ranks.count(r) for r in set(ranks)}
    suit_counts = {s: suits.count(s) for s in set(suits)}

    score = 0.0

    # Made hands
    if 3 in rank_counts.values():
        score += 0.9
    elif list(rank_counts.values()).count(2) >= 2:
        score += 0.85
    elif 2 in rank_counts.values():
        score += 0.65

    # Flush potential
    if max(suit_counts.values()) >= 4:
        score += 0.15

    # Straight potential
    unique_ranks = sorted(set(ranks))
    for i in range(len(unique_ranks) - 3):
        if unique_ranks[i + 3] - unique_ranks[i] <= 4:
            score += 0.1

    # High card
    score += max(ranks) / 20.0

    return min(score, 1.0)


class Player(Bot):

    def __init__(self):
        pass

    def handle_new_round(self, game_state, round_state, active):
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        pass

    def get_action(self, game_state, round_state, active):

        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board = round_state.board

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]

        continue_cost = opp_pip - my_pip
        pot = sum(round_state.pips)

        # === DISCARD LOGIC ===
        if DiscardAction in legal_actions:
            best_score = -1
            discard_idx = 0
            for i in range(3):
                kept = [c for j, c in enumerate(my_cards) if j != i]
                score = evaluate_strength(kept, board)
                if score > best_score:
                    best_score = score
                    discard_idx = i
            return DiscardAction(discard_idx)

        # === HAND STRENGTH ===
        strength = evaluate_strength(my_cards, board)

        # === FOLD LOGIC ===
        if continue_cost > 0:
            pot_odds = continue_cost / (pot + continue_cost)
            if strength < pot_odds:
                return FoldAction()

        # === BET / RAISE LOGIC ===
        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

            # Value bet strong hands only
            if strength > 0.75:
                raise_amt = min(
                    max_raise,
                    my_pip + int((pot + continue_cost) * 0.7)
                )
                return RaiseAction(raise_amt)

        # === CHECK / CALL ===
        if CheckAction in legal_actions:
            return CheckAction()

        return CallAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
