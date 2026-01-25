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
from skeleton.states import STARTING_STACK
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
        # --- tunables ---
        self.MC_ITERS_DISCARD = 120
        self.MC_ITERS_WINPROB = 60
        self.mode_p = 0.65  # probability of passive mode
        self.mode = "p"

        # --- per-round/per-street ---
        self._prev_street = None
        self.raises_this_street = 0

        # --- parsing robustness ---
        self._rank_chars = set("23456789TJQKA")
        self._suit_chars = set("cdhs")

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

    # -------------------- card helpers --------------------

    def _card_to_str(self, c):
        """
        Normalize a card object or string to a 2-char string like 'Ah'.

        Arguments:
        - c: card object or str

        Returns:
        - str normalized card string.
        """
        if isinstance(c, str):
            s = c
        else:
            s = str(c)

        if len(s) == 2 and s[0] in self._rank_chars and s[1] in self._suit_chars:
            return s

        if len(s) >= 2:
            t = s[-2:]
            if t[0] in self._rank_chars and t[1] in self._suit_chars:
                return t

        for i in range(len(s) - 1):
            a, b = s[i], s[i + 1]
            if a in self._rank_chars and b in self._suit_chars:
                return a + b

        raise KeyError(s)

    def _to_ids(self, cards):
        """
        Convert a list of cards into card ids.

        Arguments:
        - cards: list of cards (strings or card objects)

        Returns:
        - list[int] card ids in [0, 51].
        """
        return [CARD_ID_BY_STR[self._card_to_str(c)] for c in cards]

    def _remaining_deck(self, known_ids_set):
        """
        Build a list of available card ids excluding known cards.

        Arguments:
        - known_ids_set: set[int] of card ids already used

        Returns:
        - list[int] remaining card ids
        """
        return [cid for cid in range(52) if cid not in known_ids_set]

    # -------------------- evaluator --------------------

    def _popcount(self, x):
        """
        Count number of set bits in an integer.

        Arguments:
        - x: integer

        Returns:
        - int number of 1-bits in x
        """
        return x.bit_count()

    def _ranks_desc_from_counts(self, counts):
        """
        Return ranks in descending order, repeating by multiplicity.

        Arguments:
        - counts: list[int] length 13, counts per rank

        Returns:
        - list[int] ranks sorted high->low, repeated
        """
        out = []
        for r in range(12, -1, -1):
            out.extend([r] * counts[r])
        return out

    def _kickers_desc_excluding(self, counts, exclude_ranks, n):
        """
        Return top n kicker ranks excluding some ranks.

        Arguments:
        - counts: list[int] length 13
        - exclude_ranks: set[int]
        - n: int number of kickers

        Returns:
        - list[int] of length n (or shorter if impossible)
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
        - card_ids: list[int] length 7

        Returns:
        - tuple comparable rank (bigger is better)
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

    def hand_rank_8(self, card_ids):
        """
        Evaluate an 8-card hand by taking the best 7-card subset.

        Arguments:
        - card_ids: list[int] length 8

        Returns:
        - tuple comparable rank (bigger is better)
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
        - my8: list[int] length 8
        - opp8: list[int] length 8

        Returns:
        - float: 1.0 win, 0.5 tie, 0.0 loss
        """
        a = self.hand_rank_8(my8)
        b = self.hand_rank_8(opp8)
        if a > b:
            return 1.0
        if a == b:
            return 0.5
        return 0.0

    # -------------------- MC discard --------------------

    def _opp_choose_discard_simple(self, opp3_ids):
        """
        Choose opponent discard with a simple heuristic (keep best two ranks/pair).

        Arguments:
        - opp3_ids: list[int] length 3

        Returns:
        - int discard index in [0,2]
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
        return [i for i in range(3) if i not in keep_indices][0]

    def _mc_once_discard(self, my3_ids, board_ids, discard_idx, i_am_first_discarder):
        """
        Run one MC rollout for a chosen discard.

        Arguments:
        - my3_ids: list[int] length 3
        - board_ids: list[int] current board (len 2 or 3)
        - discard_idx: int in [0,2]
        - i_am_first_discarder: bool

        Returns:
        - float: 1.0 win, 0.5 tie, 0.0 loss
        """
        my_discard = my3_ids[discard_idx]
        my_hole2 = [my3_ids[i] for i in range(3) if i != discard_idx]
        board = list(board_ids) + [my_discard]

        if i_am_first_discarder:
            known = set(board) | set(my_hole2)
            deck = self._remaining_deck(known)
            opp3 = random.sample(deck, 3)
            opp_disc_idx = self._opp_choose_discard_simple(opp3)
            opp_discard = opp3[opp_disc_idx]
            opp_hole2 = [opp3[i] for i in range(3) if i != opp_disc_idx]
            board.append(opp_discard)
        else:
            known = set(board) | set(my_hole2)
            deck = self._remaining_deck(known)
            opp_hole2 = random.sample(deck, 2)

            if len(board) == 3:
                known2 = set(board) | set(my_hole2) | set(opp_hole2)
                deck2 = self._remaining_deck(known2)
                board.append(random.sample(deck2, 1)[0])

        need = 6 - len(board)
        if need > 0:
            known3 = set(board) | set(my_hole2) | set(opp_hole2)
            deck3 = self._remaining_deck(known3)
            board.extend(random.sample(deck3, need))

        my8 = my_hole2 + board
        opp8 = opp_hole2 + board
        return self.compare_hands_8(my8, opp8)

    def choose_discard_mc(self, my3_ids, board_ids, i_am_first_discarder, iters):
        """
        Choose discard index by MC win/tie score.

        Arguments:
        - my3_ids: list[int] length 3
        - board_ids: list[int] current board (len 2 or 3)
        - i_am_first_discarder: bool
        - iters: int iterations per discard option

        Returns:
        - int discard index 0/1/2
        """
        scores = [0.0, 0.0, 0.0]
        for d in range(3):
            s = 0.0
            for _ in range(iters):
                s += self._mc_once_discard(my3_ids, board_ids, d, i_am_first_discarder)
            scores[d] = s
        return max(range(3), key=lambda i: scores[i])

    # -------------------- MC win probability (bet/call decisions) --------------------

    def _best_two_of_three(self, my3_ids, board_ids, iters):
        """
        Choose which 2 of 3 hole cards to treat as kept for win-prob estimation.

        Arguments:
        - my3_ids: list[int] length 3
        - board_ids: list[int] current board ids

        Returns:
        - list[int] length 2 (chosen keep cards)
        """
        # quick-and-cheap: pick the pair that performs best in small MC
        best_keep = None
        pairs = [(0, 1), (0, 2), (1, 2)]
        for a, b in pairs:
            my2 = [my3_ids[a], my3_ids[b]]
            p = self.calc_win_prob_mc(my2, board_ids, iters=max(10, iters // 3))
            cand = (p, (a, b))
            if best_keep is None or cand > best_keep:
                best_keep = cand
        a, b = best_keep[1]
        return [my3_ids[a], my3_ids[b]]

    def _mc_once_winprob(self, my2_ids, board_ids):
        """
        One MC rollout to estimate our win/tie chance from current state (no discard).

        Arguments:
        - my2_ids: list[int] length 2 (our kept hole cards)
        - board_ids: list[int] current board ids (len 0/2/4/5/6)

        Returns:
        - float: 1.0 win, 0.5 tie, 0.0 loss
        """
        board = list(board_ids)
        known = set(my2_ids) | set(board)
        deck = self._remaining_deck(known)

        opp2 = random.sample(deck, 2)
        known2 = known | set(opp2)
        deck2 = self._remaining_deck(known2)

        need = 6 - len(board)
        if need > 0:
            board.extend(random.sample(deck2, need))

        my8 = my2_ids + board
        opp8 = opp2 + board
        return self.compare_hands_8(my8, opp8)

    def calc_win_prob_mc(self, my2_ids, board_ids, iters):
        """
        Estimate win probability by Monte Carlo from current state.

        Arguments:
        - my2_ids: list[int] length 2
        - board_ids: list[int] current board ids
        - iters: int MC iterations

        Returns:
        - float win probability estimate in [0,1]
        """
        s = 0.0
        for _ in range(iters):
            s += self._mc_once_winprob(my2_ids, board_ids)
        return s / float(iters)

    # -------------------- action policy --------------------

    def _margin(self, street):
        """
        Compute a pot-odds safety margin based on street and mode.

        Arguments:
        - street: int

        Returns:
        - float margin
        """
        # streets commonly: 0 preflop, 2/3 discard phase, 4 turn, 5 river (engine-specific)
        base = 0.02 if street == 0 else 0.03
        if street >= 4:
            base = 0.04
        if self.mode == "p":
            base += 0.01
        else:
            base -= 0.005
        return max(0.0, base)

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

            continue_cost = max(0, opp_pip - my_pip)
            my_contribution = STARTING_STACK - my_stack
            opp_contribution = STARTING_STACK - round_state.stacks[1 - active]
            pot_now = my_contribution + opp_contribution

            if self._prev_street != street:
                self._prev_street = street
                self.raises_this_street = 0

            my_ids = self._to_ids(my_cards)
            board_ids = self._to_ids(board_cards)

            # scale MC if low time left
            clock = getattr(game_state, "game_clock", 999.0)
            scale = 1.0
            if clock < 15:
                scale = 0.35
            elif clock < 30:
                scale = 0.6

            # -------- DISCARD PHASE --------
            if DiscardAction in legal_actions:
                i_am_first = len(board_ids) == 2
                iters = max(20, int(self.MC_ITERS_DISCARD * scale))
                d = self.choose_discard_mc(
                    my_ids, board_ids, i_am_first_discarder=i_am_first, iters=iters
                )
                return DiscardAction(d)

            # -------- NO-FREE-FOLD RULE --------
            if continue_cost == 0 and CheckAction in legal_actions:
                # if we have raise available, occasionally probe (tiny)
                if RaiseAction in legal_actions and self.raises_this_street == 0:
                    # compute quick win prob to decide value bet
                    iters = max(12, int(self.MC_ITERS_WINPROB * 0.5 * scale))
                    if len(my_ids) == 3:
                        my2 = self._best_two_of_three(my_ids, board_ids, iters)
                    else:
                        my2 = my_ids[:2]
                    p = self.calc_win_prob_mc(my2, board_ids, iters)

                    if self.mode == "a":
                        thresh = 0.58 if street == 0 else 0.60
                        if p >= thresh and random.random() < 0.65:
                            min_raise, _ = round_state.raise_bounds()
                            self.raises_this_street += 1
                            return RaiseAction(min_raise)
                    else:
                        thresh = 0.62 if street == 0 else 0.64
                        if p >= thresh and random.random() < 0.40:
                            min_raise, _ = round_state.raise_bounds()
                            self.raises_this_street += 1
                            return RaiseAction(min_raise)

                return CheckAction()

            # -------- FACING A BET: pot-odds gating using MC --------
            pot_odds = (
                continue_cost / float(pot_now + continue_cost)
                if (pot_now + continue_cost) > 0
                else 1.0
            )
            margin = self._margin(street)

            iters = max(18, int(self.MC_ITERS_WINPROB * scale))
            if (
                continue_cost >= 0.15 * max(1, my_stack)
                or pot_now >= 0.40 * STARTING_STACK
            ):
                iters = int(iters * 1.8)

            if len(my_ids) == 3:
                my2 = self._best_two_of_three(my_ids, board_ids, iters)
            else:
                my2 = my_ids[:2]

            p = self.calc_win_prob_mc(my2, board_ids, iters)

            # Big raise guardrail
            big_raise = (continue_cost > 0.25 * my_stack) or (
                continue_cost > 0.5 * pot_now
            )
            if big_raise and FoldAction in legal_actions:
                # require extra safety under big pressure
                extra = 0.03 if self.mode == "p" else 0.02
                if p < pot_odds + margin + extra:
                    return FoldAction()

            # Normal call/fold decision
            if p < pot_odds + margin:
                if FoldAction in legal_actions:
                    return FoldAction()
                if CallAction in legal_actions:
                    return CallAction()
                return CheckAction() if CheckAction in legal_actions else FoldAction()

            # If we are continuing, sometimes raise (rare; avoid raise wars)
            if (
                RaiseAction in legal_actions
                and self.raises_this_street == 0
                and not big_raise
            ):
                if self.mode == "a":
                    if p >= 0.66 and random.random() < 0.22:
                        min_raise, _ = round_state.raise_bounds()
                        self.raises_this_street += 1
                        return RaiseAction(min_raise)
                else:
                    if p >= 0.70 and random.random() < 0.12:
                        min_raise, _ = round_state.raise_bounds()
                        self.raises_this_street += 1
                        return RaiseAction(min_raise)

            return CallAction() if CallAction in legal_actions else CheckAction()

        except Exception:
            # never crash the engine; pick safest legal action
            if DiscardAction in legal_actions:
                return DiscardAction(0)
            if CheckAction in legal_actions:
                return CheckAction()
            if CallAction in legal_actions:
                return CallAction()
            return FoldAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
