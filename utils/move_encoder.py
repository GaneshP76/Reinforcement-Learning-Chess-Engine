import chess

# All possible moves in standard chess (from a1 to h8, promotions, etc.)
def get_all_possible_uci_moves():
    board = chess.Board()
    all_moves = set()

    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            for promotion in [None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                try:
                    move = chess.Move(from_square, to_square, promotion=promotion)
                    if chess.Board().is_legal(move):
                        all_moves.add(move.uci())
                except:
                    pass

    return sorted(all_moves)

# Generate mappings
ALL_UCI_MOVES = get_all_possible_uci_moves()
UCI_TO_INDEX = {uci: i for i, uci in enumerate(ALL_UCI_MOVES)}
INDEX_TO_UCI = {i: uci for i, uci in enumerate(ALL_UCI_MOVES)}

def move_to_index(move):
    """Convert a chess.Move to an action index."""
    return UCI_TO_INDEX.get(move.uci(), None)

def index_to_move(index, board):
    """Convert an index back to a legal move object, if valid."""
    uci = INDEX_TO_UCI.get(index, None)
    if uci:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return move
    return None
