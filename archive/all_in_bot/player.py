'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import random


class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        pass

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_bankroll = game_state.bankroll  # the total number of chips you've gained or lost from the beginning of the game to the start of this round
        # the total number of seconds your bot has left to play this game
        game_clock = game_state.game_clock
        round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS
        my_cards = round_state.hands[active]  # your cards
        big_blind = bool(active)  # True if you are the big blind
        pass

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        my_delta = terminal_state.deltas[active]  # your bankroll change from this round
        previous_state = terminal_state.previous_state  # RoundState before payoffs
        street = previous_state.street  # 0,2,3,4,5,6 representing when this round ended
        my_cards = previous_state.hands[active]  # your cards
        # opponent's cards or [] if not revealed
        opp_cards = previous_state.hands[1-active]
        pass

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        legal_actions = round_state.legal_actions()  # the actions you are allowed to take
        # 0, 3, 4, or 5 representing pre-flop, flop, turn, or river respectively
        street = round_state.street
        my_cards = round_state.hands[active]  # your cards
        board_cards = round_state.board  # the board cards
        # the number of chips you have contributed to the pot this round of betting
        my_pip = round_state.pips[active]
        # the number of chips your opponent has contributed to the pot this round of betting
        opp_pip = round_state.pips[1-active]
        # the number of chips you have remaining
        my_stack = round_state.stacks[active]
        # the number of chips your opponent has remaining
        opp_stack = round_state.stacks[1-active]
        continue_cost = opp_pip - my_pip  # the number of chips needed to stay in the pot
        # the number of chips you have contributed to the pot
        my_contribution = STARTING_STACK - my_stack
        # the number of chips your opponent has contributed to the pot
        opp_contribution = STARTING_STACK - opp_stack

        # Only use DiscardAction if it's in legal_actions (which already checks street)
        # legal_actions() returns DiscardAction only when street is 2 or 3
        if DiscardAction in legal_actions:
            discard_idx = self.choose_discard_mc(my_cards, board_cards)
            return DiscardAction(discard_idx)
        
        # if DiscardAction in legal_actions:
        #     # Always discards the lowest rank card in your hand

        #     ranks = "23456789TJQKA" # order of ranks
        #     rank_list = [-1, -1, -1] # uninitialized

        #     for card in range(3): # loop through cards in hand
        #         rank_list[card] = ranks.index(my_cards[card][0]) # get rank of each card

        #     # find the card with the minimum rank
        #     if rank_list[0] <= rank_list[1] and rank_list[0] <= rank_list[2]:
        #         return DiscardAction(0)
        #     elif rank_list[1] <= rank_list[2]:
        #         return DiscardAction(1)
        #     else:
        #         return DiscardAction(2)
            
        strong_cards = "TJQKA"

        if RaiseAction in legal_actions:
            # the smallest and largest numbers of chips for a legal bet/raise
            min_raise, max_raise = round_state.raise_bounds()
            min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
            max_cost = max_raise - my_pip  # the cost of a maximum bet/raise

            # if we have strong hole cards, let's raise a lot
            is_strong = True
            for card in my_cards: # Th
                if not (card[0] in strong_cards):
                    is_strong = False
                    break

            if is_strong:
                return RaiseAction(min(min_raise * 10, max_raise))
            
            else:
                if random.random() < 0.5:
                    return RaiseAction(min_raise)
                
        if CheckAction in legal_actions:  # check-call
            return CheckAction()
        if random.random() < 0.25:
            return FoldAction()
        return CallAction()

    # MONTE CARLO DISCARD LOGIC

    RANKS = "23456789TJQKA"
    SUITS = "hdcs"
    MC_ITERATIONS = 300

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
        score -= len(set(values))   # pairs/trips help
        return score

    def mc_once(self, my_cards, board_cards, discard_idx):
        discarded = my_cards[discard_idx]
        kept_cards = [c for i, c in enumerate(my_cards) if i != discard_idx]

        new_board = board_cards + [discarded]

        deck = self.remaining_deck(my_cards, board_cards)
        random.shuffle(deck)

        opp_cards = deck[:2]

        needed = 6 - len(new_board)
        future_board = deck[2:2 + needed]

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

if __name__ == '__main__':
    run_bot(Player(), parse_args())
