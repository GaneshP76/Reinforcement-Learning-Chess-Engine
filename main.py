import chess
import chess.engine
from agents.dqn_agent import DQNAgent


def print_board(board):
    print(board.unicode(borders=True))
    print(f"FEN: {board.fen()}")
    print()

agent = DQNAgent()


def play_game():
    board = chess.Board()
    print("Welcome! You are playing as White.")
    
    while not board.is_game_over():
        print_board(board)
        
        if board.turn == chess.WHITE:
            move_uci = input("Your move (in UCI, e.g., e2e4): ")
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move. Try again.")
            except:
                print("Invalid input. Try again.")
        else:
            print("AI is thinking...")
            move = agent.choose_move(board)
            print(f"AI plays: {move}")
            board.push(move)

    print_board(board)
    result = board.result()
    print(f"Game over! Result: {result}")

if __name__ == "__main__":
    play_game()
