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
from bitmask_tables import STRAIGHT_MASK_SET, STRAIGHT_MASKS, RANK_TO_INDEX
from itertools import combinations


def _popcount(x: int) -> int:
    """
    Compatibility popcount for older Python versions where int.bit_count() may not exist.
    """
    try:
        return x.bit_count()  # Python 3.8+
    except AttributeError:
        return bin(x).count("1")


class Player(Bot):
    """
    A pokerbot.
    """

    # FULL_DECK = []
    REMAINING_DECK = []

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

        # remove cards
        if street == 0:
            self.REMAINING_DECK = [
                r + s for r in self.RANKS for s in self.SUITS if r + s not in my_cards
            ]
        if street == 2:
            # Flop is revealed - remove board cards (2 cards)
            for c in board_cards:
                if c in self.REMAINING_DECK:
                    self.REMAINING_DECK.remove(c)
        if street == 3:
            # After first discard - remove the newly discarded card
            if len(board_cards) > 0 and board_cards[-1] in self.REMAINING_DECK:
                self.REMAINING_DECK.remove(board_cards[-1])
        if street == 4:
            # After second discard - remove the newly discarded card
            if len(board_cards) > 0 and board_cards[-1] in self.REMAINING_DECK:
                self.REMAINING_DECK.remove(board_cards[-1])
        if street == 5:
            # Turn is revealed
            if len(board_cards) > 0 and board_cards[-1] in self.REMAINING_DECK:
                self.REMAINING_DECK.remove(board_cards[-1])
        if street == 6:
            # River is revealed
            if len(board_cards) > 0 and board_cards[-1] in self.REMAINING_DECK:
                self.REMAINING_DECK.remove(board_cards[-1])

        # Only use DiscardAction if it's in legal_actions (which already checks street)
        # legal_actions() returns DiscardAction only when street is 2 or 3
        if DiscardAction in legal_actions:
            discard_idx = self.choose_discard_mc(my_cards, board_cards)
            return DiscardAction(discard_idx)

        # print("starting probability calculation...")
        win_probability = self._calc_winning_prob(my_cards, board_cards, street) - 0.07
        if street == 0:
            if CheckAction in legal_actions:
                return CheckAction()
            elif win_probability > opp_pip / 400:
                if CallAction in legal_actions:
                    return CallAction()
            return FoldAction
        # print("finished probability calculation")
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
            # play conservatively by calling
            if CallAction in legal_actions:
                return CallAction()
            # if CheckAction in legal_actions: # extra conservative
            #     return CheckAction()

            # fallback raise
            min_raise, max_raise = round_state.raise_bounds()
            raise_val = int(
                min(
                    max_raise,
                    # 0.75 * (win_probability * (my_contribution + opp_contribution)) / (1 - win_probability),
                    3 * win_probability * (my_contribution + opp_contribution),
                    # my_contribution + opp_contribution
                )
            )

            # return RaiseAction(max(min_raise, raise_val))  # fallback raise
            return RaiseAction(max(min_raise, continue_cost))

        # fallback
        if CheckAction in legal_actions:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()

        return FoldAction()

    ####### MONTE CARLO DISCARD LOGIC #######

    RANKS = "23456789TJQKA"
    RANK_TO_VALUE = {r: i for i, r in enumerate(RANKS, start=2)}
    SUITS_DICT = {"h": 0, "d": 1, "c": 2, "s": 3}

    SUITS = "hdcs"
    MC_ITERATIONS = 150

    # def full_deck(self):
    #     return [r + s for r in self.RANKS for s in self.SUITS]

    # def remaining_deck(self, my_cards, board_cards):
    #     deck = set(self.FULL_DECK)
    #     for c in my_cards + board_cards:
    #         if c in deck:
    #             deck.remove(c)
    #     return list(deck)

    # def hand_strength(self, cards):
    #     ranks = [c[0] for c in cards]
    #     values = [self.RANKS.index(r) for r in ranks]

    #     score = 0
    #     score += max(values) * 2
    #     score -= len(set(values))  # pairs/trips help
    #     return score

    def mc_once(self, my_cards, board_cards, discard_idx):
        new_board = board_cards.copy()
        kept_cards = my_cards.copy()
        if discard_idx != -1:
            discarded = my_cards[discard_idx]
            kept_cards = [c for i, c in enumerate(my_cards) if i != discard_idx]

            new_board = board_cards + [discarded]

        # deck = self.remaining_deck(my_cards, board_cards)
        deck = self.REMAINING_DECK
        random.shuffle(deck)

        opp_cards = deck[:2]

        needed = 6 - len(new_board)
        future_board = deck[2 : 2 + needed]

        my_hand = kept_cards + new_board + future_board
        opp_hand = opp_cards + new_board + future_board

        # return self.hand_strength(my_hand) > self.hand_strength(opp_hand)

        increase = self.compare_hands(my_hand, opp_hand)
        # print(increase)
        return increase

    def choose_discard_mc(self, my_cards, board_cards):
        wins = [0, 0, 0]

        for i in range(3):
            for _ in range(self.MC_ITERATIONS):
                # if self.mc_once(my_cards, board_cards, i):
                #     wins[i] += 1
                wins[i] += self.mc_once(my_cards, board_cards, i)[0]

        return max(range(3), key=lambda i: wins[i])

    ####### HAND EVALUATION LOGIC #######
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
            # if sm.bit_count() >= 5:
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
            # if sm.bit_count() >= 5:
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

        value = 1 if p1_rank[0] > p2_rank[0] else 0.5 if p1_rank[0] == p2_rank[0] else 0
        return [value, p1_rank[1], p2_rank[1]]

    def _calc_winning_prob(self, my_cards, board_cards, street=0):
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
        total = 0

        for _ in range(self.MC_ITERATIONS):
            # # Get remaining deck
            # deck = self.remaining_deck(my_cards, board_cards)
            # random.shuffle(deck)

            # # Deal opponent cards (assuming 2 cards for opponent)
            # opp_cards = deck[:2]

            # # Calculate how many more board cards are needed (total should be 6)
            # needed_board_cards = max(0, 6 - len(board_cards))
            # future_board = (
            #     deck[2 : 2 + needed_board_cards] if needed_board_cards > 0 else []
            # )

            # # Construct final hands
            # my_hand = my_cards + board_cards + future_board
            # opp_hand = opp_cards + board_cards + future_board
            #
            # increase = self.compare_hands(my_hand, opp_hand)
            # if increase != 1:
            #     print(
            #         f"My hand: {' '.join(my_hand)} vs Opponent hand: {' '.join(opp_hand)} => Increase: {increase}"
            #     )

            increase = self.mc_once(my_cards, board_cards, discard_idx=-1)

            # if increase[1] - increase[2] <= 6 - street:
            if increase[1] != 0 or street <= 3:
                # print("increase:", increase)
                wins += increase[0]
                total += 1

        total = self.MC_ITERATIONS if street <= 3 else total
        # print(f"Wins: {wins}, Total: {total}")
        return wins / self.MC_ITERATIONS


if __name__ == "__main__":
    # bot = Player()
    # test_cases = [
    #     # (["Ah", "Kh", "Qh"], [], 0, "Pre-flop with premium hand"),
    #     # (["Ah", "Kh", "Qh"], ["Jh", "Th"], 2, "Flop with straight flush draw"),
    #     # (["As", "Ks"], ["Ac", "Kc", "Qc", "Jc"], 4, "Turn with two pair"),
    #     # (["2h", "3h", "4h"], ["5h", "6h"], 2, "Flop with low straight"),
    #     # (["Qs", "Jc"], ["4c", "7c", "4s", "Jh", "2c", "6h"], 6, "game replay"),
    #     (["4s", "9d", "Qd"], [], 0, "game replay"),
    # ]
    # print("Python Win Probability Test Results:")
    # print("=" * 80)
    # for my_cards, board_cards, street, desc in test_cases:
    #     bot.REMAINING_DECK = [
    #         r + s for r in bot.RANKS for s in bot.SUITS if r + s not in my_cards
    #     ]
    #     for card in board_cards:
    #         if card in bot.REMAINING_DECK:
    #             bot.REMAINING_DECK.remove(card)
    #     win_prob = bot._calc_winning_prob(my_cards, board_cards, street) - 0.07
    #     print(f"{desc}")
    #     print(f"  Cards: {my_cards}, Board: {board_cards}, Street: {street}")
    #     print(f"  Win probability: {win_prob:.6f}\n")

    #     ev = win_prob * (0 + 400) - (1 - win_prob) * 400
    #     print(f"  EV: {ev:.2f}")
    # print("=" * 80)
    run_bot(Player(), parse_args())
