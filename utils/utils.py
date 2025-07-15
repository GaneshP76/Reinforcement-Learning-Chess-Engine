import chess
import numpy as np

# 0-5: white pawn, knight, bishop, rook, queen, king
# 6-11: black pawn, knight, bishop, rook, queen, king
piece_map = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def board_to_tensor(board):
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        piece_type = piece_map[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        row = 7 - (square // 8)
        col = square % 8
        tensor[color_offset + piece_type][row][col] = 1.0

    return tensor
