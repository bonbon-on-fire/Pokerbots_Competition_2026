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
        pass

    def handle_new_round(self, game_state, round_state, active):
        """
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        """
        my_bankroll = (
            game_state.bankroll
        )  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # the total number of seconds your bot has left to play this game
        game_clock = game_state.game_clock
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
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
            discard_idx = self.choose_discard_mc(my_cards, board_cards)
            return DiscardAction(discard_idx)

        win_probability = self._calc_winning_prob(my_cards, board_cards)
        ev = (
            win_probability * (my_contribution + opp_contribution)
            - (1 - win_probability) * continue_cost
        )

        # ev negative
        if ev < 0:
            if CheckAction in legal_actions:
                return CheckAction()
            return FoldAction()

        # ev positive
        if RaiseAction in legal_actions:
            min_raise, _ = round_state.raise_bounds()
            return RaiseAction(max(min_raise, continue_cost))

        # fallback
        if CheckAction in legal_actions:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()

        return FoldAction()

    ####### MONTE CARLO DISCARD LOGIC #######

    RANKS = "23456789TJQKA"
    SUITS = "hdcs"
    MC_ITERATIONS = 100

    def full_deck(self):
        return [r + s for r in self.RANKS for s in self.SUITS]

    def remaining_deck(self, my_cards, board_cards):
        deck = set(self.full_deck())
        for c in my_cards + board_cards:
            if c in deck:
                deck.remove(c)
        return list(deck)

    def hand_strength(self, cards):
        ranks = [c[0] for c in cards]
        values = [self.RANKS.index(r) for r in ranks]

        score = 0
        score += max(values) * 2
        score -= len(set(values))  # pairs/trips help
        return score

    def mc_once(self, my_cards, board_cards, discard_idx):
        discarded = my_cards[discard_idx]
        kept_cards = [c for i, c in enumerate(my_cards) if i != discard_idx]

        new_board = board_cards + [discarded]

        deck = self.remaining_deck(my_cards, board_cards)
        random.shuffle(deck)

        opp_cards = deck[:2]

        needed = 6 - len(new_board)
        future_board = deck[2 : 2 + needed]

        my_hand = kept_cards + new_board + future_board
        opp_hand = opp_cards + new_board + future_board

        return self.hand_strength(my_hand) > self.hand_strength(opp_hand)

    def choose_discard_mc(self, my_cards, board_cards):
        wins = [0, 0, 0]

        for i in range(3):
            for _ in range(self.MC_ITERATIONS):
                if self.mc_once(my_cards, board_cards, i):
                    wins[i] += 1

        return max(range(3), key=lambda i: wins[i])

    # bash all possible games to completion
    # def _simulate_game(self, my_cards, board_opp_cards, cards_reamining):
    #     if len(board_opp_cards) == 8:
    #         self.count += 1
    #         opp_cards = board_opp_cards[3:5]
    #         board_cards = board_opp_cards[:3] + board_opp_cards[5:]
    #         my_score = self.hand_strength(my_cards + board_cards)
    #         opp_score = self.hand_strength(opp_cards + board_cards)
    #         if my_score > opp_score:
    #             self.wins += 1
    #         print(self.count, ":", self.wins)
    #         return

    #     for i in range(len(cards_reamining)):
    #         next_card = cards_reamining[i]
    #         new_board_opp_cards = board_opp_cards + [next_card]
    #         new_remaining = cards_reamining[:i] + cards_reamining[i + 1 :]
    #         self._simulate_game(my_cards, new_board_opp_cards, new_remaining)

    def _calc_winning_prob(self, my_cards, board_cards):
        """
        Calculate winning probability using Monte Carlo simulation.
        Based on the current board state, simulates remaining board cards
        and opponent cards to estimate win probability.

        Args:
            my_cards: List of your cards (e.g., ["Ah", "Kd"])
            board_cards: List of current board cards (e.g., ["Ts", "Jd", "9h"])

        Returns:
            float: Probability of winning (0.0 to 1.0)
        """
        wins = 0

        for _ in range(self.MC_ITERATIONS):
            # Get remaining deck
            deck = self.remaining_deck(my_cards, board_cards)
            random.shuffle(deck)

            # Deal opponent cards (assuming 2 cards for opponent)
            opp_cards = deck[:2]

            # Calculate how many more board cards are needed (total should be 6)
            needed_board_cards = max(0, 6 - len(board_cards))
            future_board = (
                deck[2 : 2 + needed_board_cards] if needed_board_cards > 0 else []
            )

            # Construct final hands
            my_hand = my_cards + board_cards + future_board
            opp_hand = opp_cards + board_cards + future_board

            # Compare hand strengths
            if self.hand_strength(my_hand) > self.hand_strength(opp_hand):
                wins += 1
            elif self.hand_strength(my_hand) == self.hand_strength(opp_hand):
                # Count ties as half wins
                wins += 0.5

        return wins / self.MC_ITERATIONS


if __name__ == "__main__":
    # test = Player()
    # my_cards = ["Ah", "Kd"]
    # board_cards = ["Ts", "Jd", "9h"]
    # prob = test._calc_winning_prob(my_cards, board_cards)
    # print(f"Winning probability: {prob:.4f}")
    run_bot(Player(), parse_args())
