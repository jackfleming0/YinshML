import unittest
from yinsh_ml.game.encoding import StateEncoder
from yinsh_ml.game.game_state import GameState
from yinsh_ml.game.moves import Move, MoveType
from yinsh_ml.game.constants import Player, Position

class TestStateEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = StateEncoder()
        self.game_state = GameState()
        # Initialize the game state as needed

    def test_move_to_index_and_back(self):
        move = Move(
            type=MoveType.MOVE_RING,
            player=Player.WHITE,
            source=Position.from_string('E5'),
            destination=Position.from_string('F5')
        )
        index = self.encoder.move_to_index(move)
        decoded_move = self.encoder.index_to_move(index, Player.WHITE)
        self.assertEqual(move.type, decoded_move.type)
        self.assertEqual(move.player, decoded_move.player)
        self.assertEqual(move.source, decoded_move.source)
        self.assertEqual(move.destination, decoded_move.destination)

    def test_invalid_move_encoding(self):
        move = Move(
            type=MoveType.REMOVE_RING,
            player=Player.BLACK,
            source=Position.from_string('Z9')  # Invalid position
        )
        with self.assertRaises(ValueError):
            self.encoder.move_to_index(move)

    # Add more tests as needed

if __name__ == '__main__':
    unittest.main()