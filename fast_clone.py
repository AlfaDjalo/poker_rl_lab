from .poker_engine.game.state import GameState

def clone(self) -> "GameState":
    players_copy = [
        PlayerState(
            stack=p.stack,
            hand_mask=p.hand_mask,
            has_folded=p.has_folded,
            is_all_in=p.is_all_in,
        )
        for p in self.players
    ]

    new_state = GameState(players_copy, self.deck_mask)
    new_state.board_mask = self.board_mask
    new_state.pot = self.pot
    new_state.current_player = self.current_player
    new_state.dealer_position = self.dealer_position
    new_state.street = self.street
    new_state.bet_to_call = self.bet_to_call
    new_state.last_raise_call = self.last_raise_size

    return new_state
