'''
ReBeL-style pokerbot using epoch390.torchscript value model.
Query format: 44,226 dims (player_id, traverser, last_action one-hot 15, board 6, discard_choice 2, street 1, beliefs 22,100 each).
Output: (batch, 22,100) expected values per 3-card hand.
'''
from __future__ import annotations

from pathlib import Path

from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

from encoding import (
    CARD_ID_BY_STR,
    card_ids,
    hand3_to_index,
    build_query,
    point_mass_belief,
    NUM_THREE_CARD_HANDS,
    QUERY_DIM,
    _UNIFORM_BELIEF,
)

import torch

_MODEL_PATH = Path(__file__).resolve().parent.parent / "epoch390.torchscript"

# Last action encoding: 0=fold, 1=call/check, 2-11=bet sizes, 12-14=discard
LAST_ACTION_CHECK = 1
LAST_ACTION_DISCARD_BASE = 12  # 12,13,14 = discard card 0,1,2


class Player(Bot):
    '''
    ReBeL-style bot using value network (epoch390.torchscript).
    '''

    def __init__(self) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.jit.load(str(_MODEL_PATH), map_location=self._device)
        self._model.eval()
        self._initial_3: list[int] = []
        self._our_discard_index: int = -1  # 0/1/2 after first discard, else -1
        self._last_action: int = LAST_ACTION_CHECK
        self._active_id: int = 0

    def handle_new_round(self, game_state: GameState, round_state: RoundState, active: int) -> None:
        my_cards = round_state.hands[active]
        self._initial_3 = card_ids(my_cards)
        self._our_discard_index = -1
        self._last_action = LAST_ACTION_CHECK
        self._active_id = active

    def handle_round_over(self, game_state: GameState, terminal_state: TerminalState, active: int) -> None:
        pass

    def _board_six(self, board_ids: list[int]) -> list[int]:
        """Six board slots: card ids 0–51 or -1."""
        out = list(board_ids)[:6]
        while len(out) < 6:
            out.append(-1)
        return out

    def _beliefs(self, hand_ids: list[int], active: int) -> tuple[list[float], list[float]]:
        """Beliefs for player 0 and 1. We use point mass at our hand, uniform for opponent."""
        b0 = _UNIFORM_BELIEF.copy()
        b1 = _UNIFORM_BELIEF.copy()
        if len(hand_ids) == 3:
            hidx = hand3_to_index(hand_ids[0], hand_ids[1], hand_ids[2])
            pm = point_mass_belief(hidx)
            if active == 0:
                b0, b1 = pm, _UNIFORM_BELIEF.copy()
            else:
                b0, b1 = _UNIFORM_BELIEF.copy(), pm
        return (b0, b1)

    def _encode(
        self,
        round_state: RoundState,
        active: int,
        hand_ids: list[int],
        discard_0: int,
        discard_1: int,
        last_action: int,
    ) -> torch.Tensor:
        board_ids = card_ids(round_state.board)
        board_six = self._board_six(board_ids)
        b0, b1 = self._beliefs(hand_ids, active)
        vec = build_query(
            player_id=active,
            traverser=active,
            last_action=last_action,
            board_cards=board_six,
            discard_choice_0=discard_0,
            discard_choice_1=discard_1,
            street=round_state.street,
            beliefs_player0=b0,
            beliefs_player1=b1,
        )
        return torch.tensor([vec], dtype=torch.float32, device=self._device)

    def _run_model(self, query: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model(query)

    def get_action(self, game_state: GameState, round_state: RoundState, active: int):
        legal_actions = round_state.legal_actions()
        street = round_state.street
        my_cards = round_state.hands[active]
        my_pip = round_state.pips[active]
        opp_pip = round_state.pips[1 - active]
        my_stack = round_state.stacks[active]
        opp_stack = round_state.stacks[1 - active]
        continue_cost = opp_pip - my_pip
        my_contribution = STARTING_STACK - my_stack
        opp_contribution = STARTING_STACK - opp_stack

        hand_ids = card_ids(my_cards)
        discard_0 = self._our_discard_index if active == 0 else -1
        discard_1 = self._our_discard_index if active == 1 else -1

        if DiscardAction in legal_actions:
            best_idx = 0
            best_val = -float("inf")
            for i in range(len(my_cards)):
                d0 = i if active == 0 else discard_0
                d1 = i if active == 1 else discard_1
                keep = [CARD_ID_BY_STR[my_cards[j]] for j in range(len(my_cards)) if j != i]
                keep = [x for x in keep if x is not None]
                if len(keep) < 2:
                    continue
                last = LAST_ACTION_DISCARD_BASE + i
                q = self._encode(round_state, active, keep, d0, d1, last)
                out = self._run_model(q)
                sc = out.float().mean().item()
                if sc > best_val:
                    best_val = sc
                    best_idx = i
            self._our_discard_index = best_idx
            return DiscardAction(best_idx)

        q = self._encode(round_state, active, hand_ids, discard_0, discard_1, self._last_action)
        out = self._run_model(q)
        vals = out[0]

        if len(hand_ids) == 3:
            hidx = hand3_to_index(hand_ids[0], hand_ids[1], hand_ids[2])
            my_val = vals[hidx].item()
        else:
            my_val = vals.float().mean().item()

        if RaiseAction in legal_actions:
            min_raise, max_raise = round_state.raise_bounds()
            max_cost = max_raise - my_pip
            if my_val > 0.1 and max_cost <= my_stack:
                self._last_action = 2
                return RaiseAction(min_raise)
        if CheckAction in legal_actions:
            self._last_action = LAST_ACTION_CHECK
            return CheckAction()
        if continue_cost <= 0:
            self._last_action = LAST_ACTION_CHECK
            return CheckAction()
        if my_val < -0.2 and FoldAction in legal_actions:
            return FoldAction()
        self._last_action = LAST_ACTION_CHECK
        return CallAction()


if __name__ == "__main__":
    run_bot(Player(), parse_args())
