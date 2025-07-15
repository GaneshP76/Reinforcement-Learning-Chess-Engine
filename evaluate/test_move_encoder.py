import chess
from utils import move_encoder

# Create a new board (starting position)
board = chess.Board()

# Select a legal move to test
move = chess.Move.from_uci("e2e4")

# Encode the move to an index
index = move_encoder.move_to_index(move)
print(f"Move: {move}, Index: {index}")

# Decode back from index to move
decoded_move = move_encoder.index_to_move(index, board)
print(f"Decoded Move: {decoded_move}")

# Verify it's still legal
if decoded_move in board.legal_moves:
    print("✅ Move encoding and decoding successful and legal.")
else:
    print("❌ Something went wrong.")
