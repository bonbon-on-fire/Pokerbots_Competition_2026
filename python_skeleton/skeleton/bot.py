'''
This file contains the base class that you should implement for your pokerbot.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction

class Bot():
    '''
    The base class for a pokerbot.
    '''

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
        return

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
        return

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
        legal_actions=round_state.legal_actions()
        if DiscardAction in legal_actions:
            return DiscardAction(legal_actions[DiscardAction][0])  # discard the first legal card
        if RaiseAction in legal_actions:
            max_raise=round_state.raise_bounds()[1]
            return RaiseAction(legal_actions[RaiseAction][max_raise])  # minimum raise
        elif CheckAction in legal_actions:
            return CheckAction()
        return CallAction()  
