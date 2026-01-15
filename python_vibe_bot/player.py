"""
Simple example pokerbot, written in Python.
"""

from skeleton.actions import (
    FoldAction,
    CallAction,
    CheckAction,
    RaiseAction,
    DiscardAction,
)
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random


class Player(Bot):
    """
    A pokerbot.
    """

    def __init__(self):
        """
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        """

        self.mode_p = 0.65
        self.mode = "p"

        pass

    def handle_new_round(self, game_state, round_state, active):
        """
        Called when a new round starts. Called NUM_ROUNDS times.
        Pick playing style (passive/aggressive) for round.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        """

        self.mode = "p" if random.random() < self.mode_p else "a"

        pass

    def handle_round_over(self, game_state, terminal_state, active):
        """
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        """

        pass

    def parse_card(self, card):
        """
        Parses a card string into its rank and suit for easier evaluation.
        """

        rank = card[0]
        suit = card[1]
        rank_map = {
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "T": 10,
            "J": 11,
            "Q": 12,
            "K": 13,
            "A": 14,
        }

        return rank_map[rank], suit

    def core_tier(self, c1, c2):
        """
        Tier a 2-card core from pre-flop cards.
        Tier 1 (raise): TT+ OR AK/AQ/KQ OR strong suited OR top suited connectors.
        Tier 2 (raise/call): 66–99 OR suited aces (Axs) OR other broadways (AT/KJ/QJ/JT) OR suited connectors 76s+.
        Tier 3 (call/fold): 22–55 OR weak suited hands OR “one decent high card + junk” that can continue cheaply.
        Tier 4 (fold): low offsuit, unconnected hands with no pair/draw plan.
        """
        r1, s1 = self.parse_card(c1)
        r2, s2 = self.parse_card(c2)
        hi, lo = max(r1, r2), min(r1, r2)
        suited = s1 == s2
        gap = hi - lo

        if r1 == r2:
            if hi >= 10:
                return 1
            if hi >= 6:
                return 2
            return 3

        def is_broadway(r):
            return r >= 10

        broadway = is_broadway(r1) and is_broadway(r2)

        if (hi, lo) in [(14, 13), (14, 12), (13, 12)]:
            return 1
        if suited and broadway and lo >= 11:
            return 1
        if suited and gap == 1 and lo >= 11:
            return 1

        if hi == 14 and suited:
            return 2
        if broadway:
            return 2
        if suited and gap == 1 and lo >= 6:
            return 2

        if suited:
            return 3
        if hi >= 10 and lo >= 7:
            return 3

        return 4

    def preflop_tier(self, my_cards):
        """
        Evaluate all (3 choose 2) 2-card cores from 3-card hand and return tier.
        Tier 1 is best, Tier 4 is worst.
        """

        cores = [
            ((0, 1), self.core_tier(my_cards[0], my_cards[1])),
            ((0, 2), self.core_tier(my_cards[0], my_cards[2])),
            ((1, 2), self.core_tier(my_cards[1], my_cards[2])),
        ]
        best_pair_idx, best_pair_tier = min(cores, key=lambda x: x[1])

        return best_pair_tier, best_pair_idx

    def _core_value(self, c1, c2):
        """
        Convert your core tier (1 best .. 4 worst) into a numeric value.
        Bigger is better.
        """
        tier = self.core_tier(c1, c2)
        return {1: 120, 2: 90, 3: 60, 4: 20}[tier]

    def _board_penalty_after_discard(self, tossed_card, board_cards):
        """
        Penalize discards that make the shared board scarier / more useful to opponent.
        board_cards is whatever is currently public (usually 2 cards at discard time).
        """
        tr, ts = self.parse_card(tossed_card)

        board_ranks = []
        board_suits = []
        for bc in board_cards:
            r, s = self.parse_card(bc)
            board_ranks.append(r)
            board_suits.append(s)

        penalty = 0.0

        if tr in board_ranks:
            penalty += 35.0
        if ts in board_suits:
            penalty += 12.0

        for br in board_ranks:
            d = abs(tr - br)
            if d == 1:
                penalty += 10.0
            elif d == 2:
                penalty += 6.0
            elif d == 3:
                penalty += 3.0

        penalty += max(0, tr - 10) * 1.5

        return penalty

    def choose_discard_index(self, my_cards, board_cards):
        """
        Decide which of the 3 hole cards to discard to the board.

        Strategy:
        - maximize kept 2-card core value
        - minimize how much the tossed card strengthens/coordinates the public board
        """
        best_idx = 0
        best_score = -1e9

        for toss_idx in range(3):
            tossed = my_cards[toss_idx]
            kept = [my_cards[i] for i in range(3) if i != toss_idx]

            keep_value = self._core_value(kept[0], kept[1])
            board_penalty = self._board_penalty_after_discard(tossed, board_cards)

            score = keep_value - board_penalty

            if score > best_score:
                best_score = score
                best_idx = toss_idx

        mix_p = 0.12 if self.mode == "a" else 0.05
        if random.random() < mix_p:
            scores = []
            for toss_idx in range(3):
                tossed = my_cards[toss_idx]
                kept = [my_cards[i] for i in range(3) if i != toss_idx]
                keep_value = self._core_value(kept[0], kept[1])
                board_penalty = self._board_penalty_after_discard(tossed, board_cards)
                scores.append((keep_value - board_penalty, toss_idx))
            scores.sort(reverse=True)
            if len(scores) > 1:
                return scores[1][1]

        return best_idx

    def choose_discard_index(self, my_cards, board_cards):
        """
        Discard/toss decision (simple first version):
        Try discarding each card; keep the remaining 2-card private core with the best tier.
        If tie, keep the higher-ranked private cards (very mild tie-break).
        """

        def core_key(ca, cb):
            t = self.core_tier(ca, cb)
            ra, _ = self.parse_card(ca)
            rb, _ = self.parse_card(cb)
            hi, lo = max(ra, rb), min(ra, rb)
            return (t, -hi, -lo)

        candidates = []
        for k in (0, 1, 2):
            kept = [my_cards[i] for i in (0, 1, 2) if i != k]
            key = core_key(kept[0], kept[1])
            candidates.append((key, k))

        _, best_k = min(candidates, key=lambda x: x[0])
        return best_k

    def get_action(self, game_state, round_state, active):
        """
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        """

        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        board_cards = round_state.board

        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]

        continue_cost = opp_pip - my_pip
        _my_contribution = STARTING_STACK - my_stack
        _opp_contribution = STARTING_STACK - opp_stack

        # 1) Discard / toss phase
        if DiscardAction in legal_actions:
            idx = self.choose_discard_index(my_cards, board_cards)
            return DiscardAction(idx)

        # 2) Preflop (street == 0)
        if street == 0:
            tier, _best_pair_idx = self.preflop_tier(my_cards)

            # If no bet to call: check or raise
            if continue_cost == 0:
                if tier == 1 and RaiseAction in legal_actions:
                    min_raise, _ = round_state.raise_bounds()
                    return RaiseAction(min_raise)

                if tier == 2 and RaiseAction in legal_actions:
                    min_raise, _ = round_state.raise_bounds()
                    p_raise = 0.65 if self.mode == "a" else 0.35
                    if random.random() < p_raise:
                        return RaiseAction(min_raise)

                if CheckAction in legal_actions:
                    return CheckAction()
                return CallAction()

            # Facing a bet: fold/call/raise
            if tier == 4 and FoldAction in legal_actions:
                return FoldAction()

            if tier == 3:
                # "call once, fold more often to pressure" (a rough first cut)
                if (
                    self.mode == "p"
                    and random.random() < 0.35
                    and FoldAction in legal_actions
                ):
                    return FoldAction()
                return CallAction()

            if tier == 2:
                if RaiseAction in legal_actions:
                    min_raise, _ = round_state.raise_bounds()
                    p_raise = 0.55 if self.mode == "a" else 0.25
                    if random.random() < p_raise:
                        return RaiseAction(min_raise)
                return CallAction()

            # tier == 1
            if RaiseAction in legal_actions:
                min_raise, _ = round_state.raise_bounds()
                return RaiseAction(min_raise)
            return CallAction()

        # 3) Postflop fallback (conservative placeholder)
        # We'll replace this later with "made hand / draw / board texture" logic.
        if continue_cost == 0:
            if CheckAction in legal_actions:
                return CheckAction()
            # If check isn't legal (rare), call.
            return CallAction()

        # Facing a bet postflop: be cautious for now
        if FoldAction in legal_actions and continue_cost > 0:
            # passive folds more; aggressive calls more
            p_fold = 0.55 if self.mode == "p" else 0.35
            if random.random() < p_fold:
                return FoldAction()
        return CallAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
