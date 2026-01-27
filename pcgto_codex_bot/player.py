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
import pickle
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
        with open("preflop_equities_mc.pkl", "rb") as f:
            self.preflop_equities = pickle.load(f)

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
        win_probability = self._calc_winning_prob(my_cards, board_cards, street)
        # print("finished probability calculation")
        pot_size = my_contribution + opp_contribution
        pot_after_call = pot_size + continue_cost
        pot_odds = (
            continue_cost / pot_after_call
            if pot_after_call > 0 and continue_cost > 0
            else 0
        )

        # Small safety margin to avoid marginal bluff-catches versus large bets.
        call_threshold = min(0.95, pot_odds + 0.03)

        # If facing a bet, use pot odds to decide whether to continue.
        if continue_cost > 0:
            if win_probability < call_threshold:
                return FoldAction() if FoldAction in legal_actions else CheckAction()

            # With very strong hands, raise for value sometimes.
            if RaiseAction in legal_actions and win_probability > 0.82:
                min_raise, max_raise = round_state.raise_bounds()
                value_raise = int(0.9 * pot_after_call)
                raise_amount = max(min_raise, min(max_raise, value_raise))
                if raise_amount > min_raise and random.random() < 0.6:
                    return RaiseAction(raise_amount)

            return CallAction() if CallAction in legal_actions else CheckAction()

        # If not facing a bet, add some initiative with value bets and light bluffs.
        if continue_cost == 0 and RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()

            # Street-tuned thresholds. Later streets require higher equity to bet.
            if street == 0:
                value_threshold = 0.60
            elif street <= 4:
                value_threshold = 0.64
            elif street == 5:
                value_threshold = 0.68
            else:
                value_threshold = 0.72

            wants_value_bet = win_probability >= value_threshold
            wants_bluff = (
                0.46 <= win_probability < value_threshold and random.random() < 0.12
            )

            if wants_value_bet or wants_bluff:
                # Choose a pot-scaled sizing but respect engine bounds.
                target = max(BIG_BLIND, int(0.65 * max(pot_size, BIG_BLIND)))
                raise_amount = max(min_raise, min(max_raise, target))
                return RaiseAction(raise_amount)

        # Default to checking when available.
        if CheckAction in legal_actions:
            return CheckAction()

        # Otherwise call if we can.
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

        # Work on a copy so shuffling does not bias future simulations.
        deck = self.REMAINING_DECK.copy()
        random.shuffle(deck)

        opp_cards = deck[:2]
        deck = deck[2:]

        needed = 6 - len(new_board)
        future_board = deck[:needed]

        my_hand = kept_cards + new_board + future_board
        opp_hand = opp_cards + new_board + future_board

        # return self.hand_strength(my_hand) > self.hand_strength(opp_hand)

        increase = self.compare_hands(my_hand, opp_hand)
        # print(increase)
        return increase

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

    def sim_mc_once(self, my_cards, board_cards, discard_idx):
        new_board = board_cards.copy()
        kept_cards = my_cards.copy()
        if discard_idx != -1:
            discarded = my_cards[discard_idx]
            kept_cards = [c for i, c in enumerate(my_cards) if i != discard_idx]

            new_board = board_cards + [discarded]

        # deck = self.remaining_deck(my_cards, board_cards)
        deck = self.REMAINING_DECK.copy()
        random.shuffle(deck)

        opp_cards = deck[:3]
        deck = deck[3:]
        opp_discard_idx = self.choose_opponent_discard_simple(opp_cards, new_board)
        opp_kept_cards = [c for i, c in enumerate(opp_cards) if i != opp_discard_idx]

        needed = 6 - len(new_board)
        future_board = deck[2 : 2 + needed]

        my_hand = kept_cards + new_board + future_board
        opp_hand = opp_kept_cards + new_board + future_board

        # return self.hand_strength(my_hand) > self.hand_strength(opp_hand)

        increase = self.compare_hands(my_hand, opp_hand)
        # print(increase)
        return increase

    def choose_discard_mc(self, my_cards, board_cards):
        wins = [0, 0, 0]

        for i in range(3):
            for _ in range(self.MC_ITERATIONS):
                wins[i] += self.sim_mc_once(my_cards, board_cards, i)[0]

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
        return [value, p2_rank[1]]

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
        if street == 0:
            hand_key = tuple(sorted(my_cards))
            return self.preflop_equities.get(hand_key)

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

            increase = self.sim_mc_once(my_cards, board_cards, discard_idx=-1)

            wins += increase[0] if (increase[1] != 0 or street <= 3) else 0
            total += 1 if (increase[1] != 0 or street <= 3) else 0

            # # Compare hand strengths
            # if self.hand_strength(my_hand) > self.hand_strength(opp_hand):
            #     wins += 1
            # elif self.hand_strength(my_hand) == self.hand_strength(opp_hand):
            #     # Count ties as half wins
            #     wins += 0.5

        total = self.MC_ITERATIONS if street <= 3 else total
        # print(f"Wins: {wins}, Total: {total}")
        return wins / self.MC_ITERATIONS


if __name__ == "__main__":
    # bot = Player()
    # test_cases = [
    #     (["Ah", "Kh", "Qh"], [], 0, "Pre-flop with premium hand"),
    #     (["Ah", "Kh", "Qh"], ["Jh", "Th"], 2, "Flop with straight flush draw"),
    #     (["As", "Ks", "Qs"], ["Ac", "Kc", "Qc", "Jc"], 5, "Turn with two pair"),
    #     (["2h", "3h", "4h"], ["5h", "6h"], 2, "Flop with low straight"),
    #     (["Ah", "Ad", "As"], [], 0, "Pre-flop with three aces"),
    # ]
    # print("Python Win Probability Test Results:")
    # print("=" * 80)
    # for my_cards, board_cards, street, desc in test_cases:
    #     bot.REMAINING_DECK = [
    #         r + s for r in bot.RANKS for s in bot.SUITS
    #         if r + s not in my_cards
    #     ]
    #     for card in board_cards:
    #         if card in bot.REMAINING_DECK:
    #             bot.REMAINING_DECK.remove(card)
    #     win_prob = bot._calc_winning_prob(my_cards, board_cards, street)
    #     print(f"{desc}")
    #     print(f"  Cards: {my_cards}, Board: {board_cards}, Street: {street}")
    #     print(f"  Win probability: {win_prob:.6f}\n")
    # print("=" * 80)
    run_bot(Player(), parse_args())
