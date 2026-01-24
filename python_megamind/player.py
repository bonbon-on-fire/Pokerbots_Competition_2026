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
        self.MC_ITERS = 200
        self.mode_p = 0.65
        self.mode = "p"
        self._prev_street = None
        self.raises_this_street = 0
        self._mc_cache = {}
        self._debug = False

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

    def _iters_for_discard(self):
        """
        Choose MC iterations for discard phase.

        Arguments:
        Nothing.

        Returns:
        int iterations
        """
        return 35 if self.mode == "p" else 50

    def _iters_for_postdiscard(self, board_len):
        """
        Choose MC iterations for post-discard betting based on board length.

        Arguments:
        board_len: int length of board_ids (4, 5, or 6)

        Returns:
        int iterations
        """
        if board_len >= 6:
            return 110
        if board_len == 5:
            return 85
        return 65

    def _mc_once_discard(self, my3_ids, board_ids, discard_idx, i_am_first_discarder):
        """
        Run one Monte Carlo rollout for a chosen discard.

        Arguments:
        my3_ids: list[int] length 3 (our pre-discard hole)
        board_ids: list[int] current board (len 2 if we discard first; len 3 if opp already discarded)
        discard_idx: int in [0,2] which of our 3 to discard to board
        i_am_first_discarder: bool

        Returns:
        float 1.0 win, 0.5 tie, 0.0 loss
        """
        my_discard = my3_ids[discard_idx]
        my_hole2 = [my3_ids[i] for i in range(3) if i != discard_idx]
        board = list(board_ids) + [my_discard]

        if i_am_first_discarder:
            opp3 = self._draw_n(self._remaining_deck(set(board) | set(my_hole2)), 3)
            opp_disc_idx = self._opp_choose_discard_simple(opp3, board)
            opp_discard = opp3[opp_disc_idx]
            opp_hole2 = [opp3[i] for i in range(3) if i != opp_disc_idx]
            board.append(opp_discard)
        else:
            opp_hole2 = self._draw_n(
                self._remaining_deck(set(board) | set(my_hole2)), 2
            )
            if len(board) == 3:
                extra = self._draw_n(
                    self._remaining_deck(set(board) | set(my_hole2) | set(opp_hole2)), 1
                )[0]
                board.append(extra)

        need = 6 - len(board)
        if need > 0:
            fill = self._draw_n(
                self._remaining_deck(set(board) | set(my_hole2) | set(opp_hole2)), need
            )
            board.extend(fill)

        my8 = my_hole2 + board
        opp8 = opp_hole2 + board
        return self.compare_hands_8(my8, opp8)

    def _cache_get(self, key):
        return self._mc_cache.get(key, None)

    def _cache_set(self, key, val):
        if len(self._mc_cache) > 5000:
            self._mc_cache.clear()
        self._mc_cache[key] = val

    def choose_discard_mc(self, my3_ids, board_ids, i_am_first_discarder, iters=None):
        """
        Choose discard index by Monte Carlo win/tie score.

        Arguments:
        my3_ids: list[int] length 3
        board_ids: list[int] current board (len 2 or 3)
        i_am_first_discarder: bool
        iters: int MC iterations per discard option (optional)

        Returns:
        int discard index 0/1/2
        """
        if iters is None:
            iters = self.MC_ITERS if hasattr(self, "MC_ITERS") else 200

        scores = [0.0, 0.0, 0.0]
        for d in range(3):
            s = 0.0
            for _ in range(iters):
                s += self._mc_once_discard(my3_ids, board_ids, d, i_am_first_discarder)
            scores[d] = s

        return max(range(3), key=lambda i: scores[i])

    def _rank_from_idx(self, r_idx):
        """
        Convert rank index [0..12] back to rank value [2..14].

        Arguments:
        r_idx: int

        Returns:
        int rank value (2..14)
        """
        return r_idx + 2

    def _preflop_core_tier_ids(self, cida, cidb):
        """
        Tier a 2-card core using card ids (preflop heuristic).

        Arguments:
        cida: int card id [0..51]
        cidb: int card id [0..51]

        Returns:
        int tier in {1,2,3,4} where 1 is strongest
        """
        ra = self._rank_from_idx(CARD_RANK_IDX[cida])
        rb = self._rank_from_idx(CARD_RANK_IDX[cidb])
        sa = CARD_SUIT_IDX[cida]
        sb = CARD_SUIT_IDX[cidb]

        hi, lo = (ra, rb) if ra >= rb else (rb, ra)
        suited = sa == sb
        gap = hi - lo

        if ra == rb:
            if hi >= 10:
                return 1
            if hi >= 6:
                return 2
            return 3

        broadway = (ra >= 10) and (rb >= 10)

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

    def preflop_tier(self, my3_ids):
        """
        Evaluate all (3 choose 2) 2-card cores from 3-card hand and return best tier.

        Arguments:
        my3_ids: list[int] length 3

        Returns:
        (best_tier, best_pair_idx)
        best_tier: int 1..4 (1 best)
        best_pair_idx: tuple[int,int] indices of kept core inside the 3 cards
        """
        cores = [
            ((0, 1), self._preflop_core_tier_ids(my3_ids[0], my3_ids[1])),
            ((0, 2), self._preflop_core_tier_ids(my3_ids[0], my3_ids[2])),
            ((1, 2), self._preflop_core_tier_ids(my3_ids[1], my3_ids[2])),
        ]
        best_pair_idx, best_tier = min(cores, key=lambda x: x[1])
        return best_tier, best_pair_idx

    def _calc_win_prob_postdiscard(self, my2_ids, board_ids, iters):
        """
        Monte Carlo estimate of win probability AFTER discards (we have 2 hole cards).

        Arguments:
        my2_ids: list[int] length 2
        board_ids: list[int] length 4,5,6
        iters: int number of rollouts

        Returns:
        float win probability in [0,1] where tie counts as 0.5
        """
        known = set(my2_ids) | set(board_ids)
        deck = [cid for cid in range(52) if cid not in known]

        need_board = 6 - len(board_ids)
        if need_board < 0:
            need_board = 0

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

    def _choose_raise_size(
        self, round_state, my_pip, win_p, mode, pot_now, continue_cost, my_stack
    ):
        min_raise, max_raise = round_state.raise_bounds()

        if mode == "a":
            frac = max(0.0, min(1.0, (win_p - 0.50) / 0.35))
        else:
            frac = max(0.0, min(1.0, (win_p - 0.55) / 0.35))

        target = int(min_raise + frac * (max_raise - min_raise))

        cap_cost = int(0.75 * pot_now) + max(0, continue_cost)
        cap_cost = max(2, cap_cost)
        cap_cost = min(cap_cost, my_stack)
        cap_raise_to = my_pip + cap_cost

        raise_to = max(min_raise, min(max_raise, target))
        raise_to = min(raise_to, cap_raise_to)
        raise_to = max(min_raise, raise_to)
        return raise_to

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
        pot_now = my_contribution + opp_contribution

        if self._prev_street != street:
            self._prev_street = street
            self.raises_this_street = 0

        my_ids = self._to_ids(my_cards)
        board_ids = self._to_ids(board_cards)

        if DiscardAction in legal_actions:
            i_am_first = len(board_ids) == 2
            iters = self._iters_for_discard()
            cache_key = (
                "discard",
                tuple(sorted(my_ids)),
                tuple(sorted(board_ids)),
                i_am_first,
                iters,
            )

            cached = self._cache_get(cache_key)
            if cached is None:
                d = self.choose_discard_mc(
                    my_ids, board_ids, i_am_first_discarder=i_am_first, iters=iters
                )
                self._cache_set(cache_key, d)
            else:
                d = cached

            return DiscardAction(d)

        if street == 0:
            tier, _best_pair = self.preflop_tier(my_ids)

            cheap_defend = continue_cost > 0 and continue_cost <= max(
                2, int(0.25 * pot_now)
            )

            if continue_cost == 0:
                if RaiseAction in legal_actions:
                    if tier == 1:
                        min_raise, _ = round_state.raise_bounds()
                        self.raises_this_street += 1
                        return RaiseAction(min_raise)

                    if tier == 2:
                        p_raise = 0.60 if self.mode == "a" else 0.25
                        if random.random() < p_raise:
                            min_raise, _ = round_state.raise_bounds()
                            self.raises_this_street += 1
                            return RaiseAction(min_raise)

                if CheckAction in legal_actions:
                    return CheckAction()
                return CallAction() if CallAction in legal_actions else FoldAction()

            if tier == 4:
                if cheap_defend and CallAction in legal_actions and continue_cost <= 1:
                    return CallAction()
                return FoldAction() if FoldAction in legal_actions else CallAction()

            if tier == 3:
                if cheap_defend and CallAction in legal_actions:
                    return CallAction()
                if (
                    self.mode == "p"
                    and random.random() < 0.35
                    and FoldAction in legal_actions
                ):
                    return FoldAction()
                return CallAction() if CallAction in legal_actions else FoldAction()

            if tier == 2:
                if RaiseAction in legal_actions:
                    p_reraise = 0.35 if self.mode == "a" else 0.10
                    if random.random() < p_reraise:
                        min_raise, _ = round_state.raise_bounds()
                        self.raises_this_street += 1
                        return RaiseAction(min_raise)
                return CallAction() if CallAction in legal_actions else FoldAction()

            if RaiseAction in legal_actions:
                min_raise, _ = round_state.raise_bounds()
                self.raises_this_street += 1
                return RaiseAction(min_raise)
            return CallAction() if CallAction in legal_actions else FoldAction()

        if len(my_ids) == 2 and len(board_ids) >= 4:
            iters = self._iters_for_postdiscard(len(board_ids))
            cache_key = ("winp", tuple(sorted(my_ids)), tuple(sorted(board_ids)), iters)
            cached = self._cache_get(cache_key)
            if cached is None:
                win_p = self._calc_win_prob_postdiscard(my_ids, board_ids, iters=iters)
                self._cache_set(cache_key, win_p)
            else:
                win_p = cached

            if continue_cost > 0:
                required = continue_cost / float(pot_now + continue_cost)
            else:
                required = 0.0

            safety = 0.01 if self.mode == "a" else 0.03
            tiny_call = continue_cost > 0 and (
                continue_cost <= max(2, int(0.15 * pot_now))
                or continue_cost <= max(2, int(0.03 * my_stack))
            )

            if continue_cost == 0:
                if RaiseAction in legal_actions and self.raises_this_street < 2:
                    bet_thresh = 0.55 if self.mode == "p" else 0.50
                    bluff_freq = 0.03 if self.mode == "p" else 0.10

                    should_value_bet = win_p >= bet_thresh
                    should_bluff = (win_p <= 0.40) and (random.random() < bluff_freq)

                    if should_value_bet or should_bluff:
                        raise_to = self._choose_raise_size(
                            round_state,
                            my_pip,
                            win_p,
                            self.mode,
                            pot_now,
                            continue_cost,
                            my_stack,
                        )
                        self.raises_this_street += 1
                        return RaiseAction(raise_to)

                return CheckAction() if CheckAction in legal_actions else CallAction()

            if win_p < 0.10 and FoldAction in legal_actions:
                return FoldAction()

            if (
                (win_p < required + safety)
                and (not tiny_call)
                and FoldAction in legal_actions
            ):
                return FoldAction()

            raise_thresh = 0.64 if self.mode == "p" else 0.58
            if (
                RaiseAction in legal_actions
                and win_p >= raise_thresh
                and self.raises_this_street < 2
            ):
                raise_to = self._choose_raise_size(
                    round_state,
                    my_pip,
                    win_p,
                    self.mode,
                    pot_now,
                    continue_cost,
                    my_stack,
                )
                self.raises_this_street += 1
                return RaiseAction(raise_to)

            return CallAction() if CallAction in legal_actions else FoldAction()

        if CheckAction in legal_actions:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
        return FoldAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
