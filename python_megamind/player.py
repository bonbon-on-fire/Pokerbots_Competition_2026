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

        self._mc_cache = {}

        # Runtime knobs (keep conservative so we don't time out)
        self.DISCARD_ITERS_P = 25
        self.DISCARD_ITERS_A = 35

        self.POST_ITERS_FLOP = 55  # board_len=4
        self.POST_ITERS_TURN = 75  # board_len=5
        self.POST_ITERS_RIVER = 95  # board_len=6

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
        return

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
        - card_strs: list[str] like ['Ah','Td','2c']

        Returns:
        - list[int] card ids in [0, 51]
        """
        return [CARD_ID_BY_STR[c] for c in card_strs]

    def _remaining_deck(self, known_ids):
        """
        Build a list of available card ids excluding known cards.

        Arguments:
        - known_ids: set[int]

        Returns:
        - list[int]
        """
        return [cid for cid in range(52) if cid not in known_ids]

    def _draw_n(self, deck, n):
        """
        Draw n distinct cards uniformly at random from a deck list.

        Arguments:
        - deck: list[int]
        - n: int

        Returns:
        - list[int] of length n
        """
        return random.sample(deck, n)

    def _opp_choose_discard_simple(self, opp3_ids):
        """
        Choose which of opponent's 3 hole cards they discard (simple heuristic).

        Arguments:
        opp3_ids: list[int] length 3

        Returns:
        int index in [0,2]
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

    def _mc_once_discard(self, my3_ids, board_ids, discard_idx, i_am_first_discarder):
        """
        Run one Monte Carlo rollout for a chosen discard.

        Arguments:
        my3_ids: list[int] length 3
        board_ids: list[int] len 2 or 3
        discard_idx: int in [0,2]
        i_am_first_discarder: bool

        Returns:
        float 1.0 win, 0.5 tie, 0.0 loss
        """
        my_discard = my3_ids[discard_idx]
        my_hole2 = [my3_ids[i] for i in range(3) if i != discard_idx]
        board = list(board_ids) + [my_discard]

        if i_am_first_discarder:
            known = set(board) | set(my_hole2)
            opp3 = self._draw_n(self._remaining_deck(known), 3)
            opp_disc_idx = self._opp_choose_discard_simple(opp3)
            opp_discard = opp3[opp_disc_idx]
            opp_hole2 = [opp3[i] for i in range(3) if i != opp_disc_idx]
            board.append(opp_discard)
        else:
            known = set(board) | set(my_hole2)
            opp_hole2 = self._draw_n(self._remaining_deck(known), 2)

            if len(board) == 3:
                known2 = set(board) | set(my_hole2) | set(opp_hole2)
                board.append(self._draw_n(self._remaining_deck(known2), 1)[0])

        need = 6 - len(board)
        if need > 0:
            known3 = set(board) | set(my_hole2) | set(opp_hole2)
            board.extend(self._draw_n(self._remaining_deck(known3), need))

        my8 = my_hole2 + board
        opp8 = opp_hole2 + board
        return self.compare_hands_8(my8, opp8)

    def choose_discard_mc(self, my3_ids, board_ids, i_am_first_discarder, iters):
        """
        Choose discard index by Monte Carlo win/tie score.

        Arguments:
        my3_ids: list[int] length 3
        board_ids: list[int] length 2 or 3
        i_am_first_discarder: bool
        iters: int rollouts per discard option

        Returns:
        int discard index 0/1/2
        """
        scores = [0.0, 0.0, 0.0]
        for d in range(3):
            s = 0.0
            for _ in range(iters):
                s += self._mc_once_discard(my3_ids, board_ids, d, i_am_first_discarder)
            scores[d] = s
        return max(range(3), key=lambda i: scores[i])

    def _calc_win_prob_postdiscard(self, my2_ids, board_ids, iters):
        """
        Monte Carlo estimate of win probability after discards.

        Arguments:
        my2_ids: list[int] length 2
        board_ids: list[int] length 4,5,6
        iters: int rollouts

        Returns:
        float win probability in [0,1] where tie counts as 0.5
        """
        known = set(my2_ids) | set(board_ids)
        deck = [cid for cid in range(52) if cid not in known]

        need_board = max(0, 6 - len(board_ids))
        score = 0.0
        for _ in range(iters):
            draw = random.sample(deck, 2 + need_board)
            opp2 = draw[:2]
            future = draw[2:]
            full_board = list(board_ids) + future

            my8 = list(my2_ids) + full_board
            opp8 = list(opp2) + full_board
            score += self.compare_hands_8(my8, opp8)

        return score / float(iters)

    def _pot_odds(self, pot, cost):
        """
        Compute pot odds (minimum win prob needed to call).

        Arguments:
        - pot: int current pot before calling
        - cost: int amount to call

        Returns:
        - float pot odds in [0,1]
        """
        if cost <= 0:
            return 0.0
        return cost / float(pot + cost)

    def _safety_margin(self, street):
        """
        Safety margin added on top of pot-odds.

        Arguments:
        - street: int

        Returns:
        - float margin
        """
        # Streets: 0 pre, 3/4/5 post
        if street == 0:
            base = 0.02
        elif street <= 3:
            base = 0.03
        else:
            base = 0.04

        if self.mode == "p":
            base += 0.01
        else:
            base -= 0.005

        return max(0.0, base)

    def _is_big_pressure(self, cost, pot, my_stack):
        """
        Detect big bet/raise pressure.

        Arguments:
        - cost: int continue_cost
        - pot: int current pot
        - my_stack: int

        Returns:
        - bool
        """
        return (cost > 0.50 * pot) or (cost > 0.25 * my_stack)

    def _choose_raise_to(self, round_state, my_pip, continue_cost, pot, win_p):
        """
        Choose a raise_to (pip total) using pot-fraction sizing, clamped to bounds.

        Arguments:
        - round_state: RoundState
        - my_pip: int
        - continue_cost: int
        - pot: int current pot before calling
        - win_p: float

        Returns:
        - int raise_to (total pip after raise)
        """
        min_raise, max_raise = round_state.raise_bounds()

        # Strength band
        if win_p >= 0.80:
            frac = 0.90
        elif win_p >= 0.68:
            frac = 0.60
        else:
            frac = 0.33

        # Slightly bigger in aggressive mode
        if self.mode == "a":
            frac = min(0.95, frac + 0.10)

        # We raise over the current price: call + add a pot-fraction
        target_add = int(frac * (pot + continue_cost))
        target_to = my_pip + continue_cost + max(1, target_add)

        if target_to < min_raise:
            target_to = min_raise
        if target_to > max_raise:
            target_to = max_raise
        return target_to

    def get_action(self, game_state, round_state, active):
        """
        Decide and return an action for the current game state.

        Arguments:
        - game_state: GameState
        - round_state: RoundState
        - active: int player index (0 or 1)

        Returns:
        - Action instance
        """
        legal_actions = round_state.legal_actions()

        try:
            street = round_state.street
            my_cards = round_state.hands[active]
            board_cards = round_state.board

            my_pip = round_state.pips[active]
            opp_pip = round_state.pips[1 - active]
            my_stack = round_state.stacks[active]

            continue_cost = opp_pip - my_pip
            my_contribution = STARTING_STACK - my_stack
            opp_contribution = STARTING_STACK - round_state.stacks[1 - active]
            pot_now = my_contribution + opp_contribution

            if self._prev_street != street:
                self._prev_street = street
                self.raises_this_street = 0

            my_ids = self._to_ids(my_cards)
            board_ids = self._to_ids(board_cards)

            # -------- DISCARD PHASE --------
            if DiscardAction in legal_actions:
                i_am_first = len(board_ids) == 2
                iters = (
                    self.DISCARD_ITERS_P if self.mode == "p" else self.DISCARD_ITERS_A
                )
                key = (
                    "discard",
                    tuple(sorted(my_ids)),
                    tuple(sorted(board_ids)),
                    i_am_first,
                    iters,
                    self.mode,
                )
                cached = self._mc_cache.get(key, None)
                if cached is None:
                    d = self.choose_discard_mc(
                        my_ids, board_ids, i_am_first_discarder=i_am_first, iters=iters
                    )
                    if len(self._mc_cache) > 6000:
                        self._mc_cache.clear()
                    self._mc_cache[key] = d
                else:
                    d = cached
                return DiscardAction(d)

            # -------- ALWAYS: NO FREE FOLD --------
            if continue_cost <= 0:
                if CheckAction in legal_actions:
                    # rare tiny probe (mainly in aggressive mode)
                    if (
                        RaiseAction in legal_actions
                        and self.mode == "a"
                        and self.raises_this_street == 0
                        and random.random() < 0.07
                        and street != 0
                    ):
                        min_raise, _ = round_state.raise_bounds()
                        self.raises_this_street += 1
                        return RaiseAction(min_raise)
                    return CheckAction()
                return CallAction() if CallAction in legal_actions else CheckAction()

            # -------- POST-DISCARD: USE MC + POT ODDS --------
            if len(my_ids) == 2 and len(board_ids) >= 4:
                if len(board_ids) >= 6:
                    iters = self.POST_ITERS_RIVER
                elif len(board_ids) == 5:
                    iters = self.POST_ITERS_TURN
                else:
                    iters = self.POST_ITERS_FLOP

                # High-stakes: spend a bit more compute where it matters
                if self._is_big_pressure(continue_cost, pot_now, my_stack):
                    iters = int(iters * 1.6)

                key = ("post", tuple(sorted(my_ids)), tuple(sorted(board_ids)), iters)
                win_p = self._mc_cache.get(key, None)
                if win_p is None:
                    win_p = self._calc_win_prob_postdiscard(
                        my_ids, board_ids, iters=iters
                    )
                    if len(self._mc_cache) > 6000:
                        self._mc_cache.clear()
                    self._mc_cache[key] = win_p

                pot_odds = self._pot_odds(pot_now, continue_cost)
                need = pot_odds + self._safety_margin(street)

                # Big pressure: require extra confidence
                if self._is_big_pressure(continue_cost, pot_now, my_stack):
                    need += 0.04 if self.mode == "p" else 0.03

                # Fold if not meeting pot-odds threshold
                if win_p < need and FoldAction in legal_actions:
                    return FoldAction()

                # Raise ladder: avoid raise wars unless very strong
                can_raise = (
                    RaiseAction in legal_actions
                    and (self.raises_this_street < 2 or win_p >= 0.78)
                    and not self._is_big_pressure(continue_cost, pot_now, my_stack)
                )

                if can_raise and win_p >= (0.66 if self.mode == "a" else 0.70):
                    raise_to = self._choose_raise_to(
                        round_state, my_pip, continue_cost, pot_now, win_p
                    )
                    self.raises_this_street += 1
                    return RaiseAction(raise_to)

                return CallAction() if CallAction in legal_actions else CheckAction()

            # -------- PREFLOP: SIMPLE TIGHT DISCIPLINE --------
            # If we don't have a post-discard MC estimate yet, do not spew.
            if street == 0:
                # very tight vs big bets
                if self._is_big_pressure(continue_cost, pot_now, my_stack):
                    if FoldAction in legal_actions:
                        return FoldAction()
                    return CallAction()

                # Otherwise just call a lot, raise rarely
                if (
                    RaiseAction in legal_actions
                    and self.mode == "a"
                    and self.raises_this_street == 0
                ):
                    if random.random() < 0.10:
                        min_raise, _ = round_state.raise_bounds()
                        self.raises_this_street += 1
                        return RaiseAction(min_raise)

                return CallAction() if CallAction in legal_actions else FoldAction()

            # -------- DEFAULT FALLBACK --------
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CheckAction()

        except Exception:
            # crash-proof fallback to avoid socket disconnects
            if CheckAction in legal_actions:
                return CheckAction()
            if CallAction in legal_actions:
                return CallAction()
            if FoldAction in legal_actions:
                return FoldAction()
            return CheckAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
