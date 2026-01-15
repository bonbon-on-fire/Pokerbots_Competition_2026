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

        self.mode_p = 0.65  # probablity of playing passively (CUSTOMIZE)
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

        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # the total number of seconds your bot has left to play this game
        game_clock = game_state.game_clock
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind

        if random.random() < self.mode_p:
            self.mode = "p"
        else:
            self.mode = "a"

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

        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0,2,3,4,5,6 representing when this round ended
        my_cards = previous_state.hands[active]  # your cards
        # opponent's cards or [] if not revealed
        opp_cards = previous_state.hands[1 - active]

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
        """

        r1, s1 = self.parse_card(c1)
        r2, s2 = self.parse_card(c2)
        hi, lo = max(r1, r2), min(r1, r2)
        score = 0

        # Pair is highest ranked
        if r1 == r2:
            if hi >= 10:
                return
            return score

        def highcard_bonus(rank):
            if rank >= 11:
                return 6  # bonus for J or higher (CUSTOMIZE)
            elif rank >= 9:
                return 2  # bonus for 9 or 10 (CUSTOMIZE)
            else:
                return 0

        # High card bonus
        score += highcard_bonus(c1) + highcard_bonus(c2)
        score += hi * 1.2  # weight for high card (CUSTOMIZE)
        score += lo * 0.8  # weight for low card (CUSTOMIZE)

        # Suitedness bonus
        if s1 == s2:
            score += 5  # bonus for suited (CUSTOMIZE)

        # Connectedness bonus
        gap = hi - lo
        if gap <= 1:
            score += 7  # bonus for connected or one-gap (CUSTOMIZE)
        elif gap == 2:
            score += 4  # bonus for two-gap (CUSTOMIZE)
        elif gap == 3:
            score += 1  # bonus for three-gap (CUSTOMIZE)

        # Ace high bonus
        if hi == 14:
            score += 7  # bonus for Ace high (CUSTOMIZE)
            if s1 == s2 and lo >= 10:
                score += 4  # bonus for Ace suited with 10 or higher (CUSTOMIZE)
            elif s1 == s2:
                score += 2  # bonus for Ace suited with lower card (CUSTOMIZE)

        return score

    def preflop_tier(self, c1, c2):
        """
        Evaluate all (3 choose 2) 2-card cores from 3-card hand and return tier.
        Tier 1 is best, Tier 4 is worst.
        """

        cores = [
            ((0, 1), self.core_tier(c1, c2)),
            ((0, 2), self.core_tier(c1, c2)),
            ((1, 2), self.core_tier(c1, c2)),
        ]
        best_pair_idx, best_pair_tier = min(cores, key=lambda x: x[1])

        return best_pair_tier, best_pair_idx

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

        legal_actions = (
            round_state.legal_actions()
        )  # the actions you are allowed to take
        # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        street = round_state.street
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.board  # the board cards
        # the number of chips you have contributed to the pot this round of betting
        my_pip = round_state.pips[active]
        # the number of chips your opponent has contributed to the pot this round of betting
        opp_pip = round_state.pips[1 - active]
        # the number of chips you have remaining
        my_stack = round_state.stacks[active]
        # the number of chips your opponent has remaining
        opp_stack = round_state.stacks[1 - active]
        continue_cost = (
            opp_pip - my_pip
        )  # the number of chips needed to stay in the pot
        # the number of chips you have contributed to the pot
        my_contribution = STARTING_STACK - my_stack
        # the number of chips your opponent has contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack

        # Only use DiscardAction if it's in legal_actions (which already checks street)
        # legal_actions() returns DiscardAction only when street is 2 or 3
        if DiscardAction in legal_actions:
            # Always discards the first card in the bot's hand
            return DiscardAction(0)
        if RaiseAction in legal_actions:
            # the smallest and largest numbers of chips for a legal bet/raise
            min_raise, max_raise = round_state.raise_bounds()
            min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
            max_cost = max_raise - my_pip  # the cost of a maximum bet/raise
            if random.random() < 0.5:
                return RaiseAction(min_raise)
        if CheckAction in legal_actions:  # check-call
            return CheckAction()
        if random.random() < 0.25:
            return FoldAction()
        return CallAction()

        # Pre-flop decision logic
        if street == 0:
            tier, core_idxs, core_score = self._preflop_tier(my_cards)


if __name__ == "__main__":
    run_bot(Player(), parse_args())
