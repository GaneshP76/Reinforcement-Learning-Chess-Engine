import torch
import chess
import random

from agents.dqn_agent import DQNModel
from utils.utils import board_to_tensor
from utils import move_encoder

CHECKPOINT_PATH = "data/dqn_checkpoint.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = DQNModel().to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

NUM_GAMES = 500
results = {"win": 0, "loss": 0, "draw": 0}

for game in range(NUM_GAMES):
    board = chess.Board()
    dqn_color = chess.WHITE if game % 2 == 0 else chess.BLACK

    while not board.is_game_over():
        legal_moves = list(board.legal_moves)

        if board.turn == dqn_color:  # DQN's move
            state_tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                q_values = model(state_tensor)[0].cpu()

            legal_indices = []
            index_to_move = {}
            for m in legal_moves:
                idx = move_encoder.move_to_index(m)
                if idx is not None:
                    legal_indices.append(idx)
                    index_to_move[idx] = m

            if not legal_indices:
                move = random.choice(legal_moves)
            else:
                best_index = max(legal_indices, key=lambda i: q_values[i])
                move = index_to_move[best_index]
        else:
            move = random.choice(legal_moves)

        board.push(move)

    result = board.result()
    if (result == "1-0" and dqn_color == chess.WHITE) or (result == "0-1" and dqn_color == chess.BLACK):
        results["win"] += 1
    elif (result == "0-1" and dqn_color == chess.WHITE) or (result == "1-0" and dqn_color == chess.BLACK):
        results["loss"] += 1
    else:
        results["draw"] += 1

# === Summary ===
print("\n✅ Evaluation complete (DQN vs Random)")
print(f"Games Played: {NUM_GAMES}")
print(f"Wins : {results['win']}")
print(f"Losses: {results['loss']}")
print(f"Draws : {results['draw']}")
print(f"Win Rate: {results['win'] / NUM_GAMES:.2%}")
print(f"Loss Rate: {results['loss'] / NUM_GAMES:.2%}")
print(f"Draw Rate: {results['draw'] / NUM_GAMES:.2%}")
