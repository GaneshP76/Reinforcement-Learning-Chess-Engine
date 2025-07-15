import torch
import chess
from agents.dqn_agent import DQNModel
from utils.utils import board_to_tensor


# Initialize a new chess board
board = chess.Board()

# Convert the board to tensor format
input_tensor = board_to_tensor(board)  # shape (12, 8, 8)
input_tensor = torch.tensor(input_tensor).unsqueeze(0)  # add batch dim → shape (1, 12, 8, 8)

# Load the model
model = DQNModel()

# Forward pass
with torch.no_grad():
    output = model(input_tensor)  # shape: (1, 4672)

# Show output
print("Output shape:", output.shape)
print("Top 5 Q-values (logits):")
topk = torch.topk(output, 5)
print("Indices:", topk.indices)
print("Values:", topk.values)
