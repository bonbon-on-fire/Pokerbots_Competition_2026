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
from bitmask_tables import (
    CARD_ID_BY_STR,
    CARD_RANK_IDX,
    CARD_SUIT_IDX,
    CARD_RANK_BIT,
    straight_high,
)

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
        self._prev_street = None
        self.raises_this_street = 0

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
        self.mode = "p" if random.random() < self.mode_p else "a"
        self._prev_street = None
        self.raises_this_street = 0

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

    def _popcount(self, x):
        """
        Count number of set bits in an integer.

        Arguments:
        x: integer

        Returns:
        int number of 1-bits in x
        """
        return x.bit_count()

    def _ranks_desc_from_counts(self, counts):
        """
        Return ranks in descending order, repeating by multiplicity.

        Arguments:
        counts: list[int] length 13, counts per rank.

        Returns:
        list[int] ranks (rank_idx) sorted high->low, repeated.
        """
        out = []
        for r in range(12, -1, -1):
            out.extend([r] * counts[r])
        return out

    def _kickers_desc_excluding(self, counts, exclude_ranks, n):
        """
        Return top n kicker ranks excluding some ranks.

        Arguments:
        counts: list[int] length 13
        exclude_ranks: set[int]
        n: int number of kickers

        Returns:
        list[int] of length n (or shorter if impossible)
        """
        out = []
        for r in range(12, -1, -1):
            if r in exclude_ranks:
                continue
            for _ in range(counts[r]):
                out.append(r)
                if len(out) == n:
                    return out
        return out

    def hand_rank_7(self, card_ids):
        """
        Evaluate a 7-card hand and return a comparable rank tuple.

        Arguments:
        card_ids: list[int] length 7, each in [0, 51]

        Returns:
        tuple that compares correctly via normal tuple comparison (bigger is better)
        Format: (category, t1, t2, t3, t4, t5)
        Category mapping:
        8: straight flush
        7: four of a kind
        6: full house
        5: flush
        4: straight
        3: three of a kind
        2: two pair
        1: one pair
        0: high card
        """
        rank_counts = [0] * 13
        suit_masks = [0, 0, 0, 0]
        rank_mask = 0

        for cid in card_ids:
            r = CARD_RANK_IDX[cid]
            s = CARD_SUIT_IDX[cid]
            rank_counts[r] += 1
            rb = CARD_RANK_BIT[cid]
            rank_mask |= rb
            suit_masks[s] |= rb

        flush_suit = None
        for s in range(4):
            if self._popcount(suit_masks[s]) >= 5:
                flush_suit = s
                break

        if flush_suit is not None:
            sf_high = straight_high(suit_masks[flush_suit])
            if sf_high is not None:
                return (8, sf_high, 0, 0, 0, 0)

        quads = []
        trips = []
        pairs = []

        for r in range(12, -1, -1):
            c = rank_counts[r]
            if c == 4:
                quads.append(r)
            elif c == 3:
                trips.append(r)
            elif c == 2:
                pairs.append(r)

        if quads:
            q = quads[0]
            kicker = self._kickers_desc_excluding(rank_counts, {q}, 1)[0]
            return (7, q, kicker, 0, 0, 0)

        if trips:
            t = trips[0]
            if len(trips) >= 2:
                p = trips[1]
                return (6, t, p, 0, 0, 0)
            if pairs:
                p = pairs[0]
                return (6, t, p, 0, 0, 0)

        if flush_suit is not None:
            mask = suit_masks[flush_suit]
            flush_ranks = []
            for r in range(12, -1, -1):
                if mask & (1 << r):
                    flush_ranks.append(r)
                    if len(flush_ranks) == 5:
                        break
            return (
                5,
                flush_ranks[0],
                flush_ranks[1],
                flush_ranks[2],
                flush_ranks[3],
                flush_ranks[4],
            )

        st_high = straight_high(rank_mask)
        if st_high is not None:
            return (4, st_high, 0, 0, 0, 0)

        if trips:
            t = trips[0]
            kickers = self._kickers_desc_excluding(rank_counts, {t}, 2)
            return (3, t, kickers[0], kickers[1], 0, 0)

        if len(pairs) >= 2:
            p1, p2 = pairs[0], pairs[1]
            kicker = self._kickers_desc_excluding(rank_counts, {p1, p2}, 1)[0]
            return (2, p1, p2, kicker, 0, 0)

        if len(pairs) == 1:
            p = pairs[0]
            kickers = self._kickers_desc_excluding(rank_counts, {p}, 3)
            return (1, p, kickers[0], kickers[1], kickers[2], 0)

        ranks = self._ranks_desc_from_counts(rank_counts)
        return (0, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4])

    def compare_hands_7(self, my7, opp7):
        """
        Compare two 7-card hands.

        Arguments:
        - my7: list[int] length 7
        - opp7: list[int] length 7

        Returns:
        - float: 1.0 if my hand wins, 0.5 if tie, 0.0 if lose
        """
        a = self.hand_rank_7(my7)
        b = self.hand_rank_7(opp7)
        if a > b:
            return 1.0
        if a == b:
            return 0.5
        return 0.0

    def hand_rank_8(self, card_ids):
        """
        Evaluate an 8-card hand by taking the best 7-card subset.

        Arguments:
        card_ids: list[int] length 8

        Returns:
        tuple comparable rank (bigger is better)
        """
        best = None
        for drop in range(8):
            seven = card_ids[:drop] + card_ids[drop + 1 :]
            r = self.hand_rank_7(seven)
            if best is None or r > best:
                best = r
        return best

    def compare_hands_8(self, my8, opp8):
        """
        Compare two 8-card hands.

        Arguments:
        my8: list[int] length 8
        opp8: list[int] length 8

        Returns:
        float 1.0 win, 0.5 tie, 0.0 loss
        """
        a = self.hand_rank_8(my8)
        b = self.hand_rank_8(opp8)
        if a > b:
            return 1.0
        if a == b:
            return 0.5
        return 0.0

    def _to_ids(self, card_strs):
        """
        Convert a list of card strings into card ids.

        Arguments:
        - card_strs: list[str] of cards like ['Ah','Td','2c'].

        Returns:
        - list[int] card ids in [0, 51].
        """
        return [CARD_ID_BY_STR[c] for c in card_strs]

    def _remaining_deck(self, known_ids):
        """
        Build a list of available card ids excluding known cards.

        Arguments:
        - known_ids: set[int] of card ids already seen/used.

        Returns:
        - list[int] of remaining card ids.
        """
        return [cid for cid in range(52) if cid not in known_ids]

    def _draw_n(self, deck, n):
        """
        Draw n distinct cards uniformly at random from a deck list.

        Arguments:
        - deck: list[int] card ids remaining.
        - n: int number of cards to draw.

        Returns:
        - list[int] of length n (distinct).
        """
        return random.sample(deck, n)

    def _opp_choose_discard_simple(self, opp3_ids, board_ids):
        """
        Choose which of opponent's 3 hole cards they discard (simple heuristic).

        Arguments:
        opp3_ids: list[int] length 3
        board_ids: list[int] current board ids (after our discard may be shown)

        Returns:
        int index in [0,2] to discard
        """
        ranks = [CARD_RANK_IDX[c] for c in opp3_ids]

        best_keep = None
        for i in range(3):
            keep = [j for j in range(3) if j != i]
            r1, r2 = ranks[keep[0]], ranks[keep[1]]
            hi, lo = (r1, r2) if r1 >= r2 else (r2, r1)
            pair_bonus = 100 if r1 == r2 else 0
            score = pair_bonus + hi * 2 + lo
            cand = (score, tuple(keep))
            if best_keep is None or cand > best_keep:
                best_keep = cand

        keep_indices = best_keep[1]
        discard_idx = [i for i in range(3) if i not in keep_indices][0]
        return discard_idx

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
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack
        my_ids = self._to_ids(my_cards)
        board_ids = self._to_ids(board_cards)

        if DiscardAction in legal_actions:
            return DiscardAction(0)
        if RaiseAction in legal_actions:
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


if __name__ == "__main__":
    run_bot(Player(), parse_args())
