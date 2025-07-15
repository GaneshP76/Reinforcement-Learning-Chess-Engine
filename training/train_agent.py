import torch
import torch.nn as nn
import torch.optim as optim
import random
import chess

from agents.dqn_agent import DQNModel
from agents.replay_buffer import ReplayBuffer
from utils.utils import board_to_tensor
from utils import move_encoder

import os
import csv

CHECKPOINT_PATH = "data/dqn_checkpoint.pth"
REWARD_LOG_PATH = "data/reward_log.csv"

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]

# === Hyperparameters ===
EPISODES = 2000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Setup ===
model = DQNModel().to(DEVICE)
target_model = DQNModel().to(DEVICE)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
replay_buffer = ReplayBuffer()

def save_checkpoint(model, optimizer, episode, epsilon, path=CHECKPOINT_PATH):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episode': episode,
        'epsilon': epsilon
    }, path)

def load_checkpoint(model, optimizer, path=CHECKPOINT_PATH):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Loaded checkpoint from episode {checkpoint['episode']}")
        return checkpoint['episode'], checkpoint['epsilon']
    else:
        print("✅ No checkpoint found. Starting from Episode 0")
        return 0, EPSILON_START

def log_reward_to_csv(episode, reward, epsilon, path=REWARD_LOG_PATH):
    write_header = not os.path.exists(path)
    with open(path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["episode", "reward", "epsilon"])
        writer.writerow([episode, reward, epsilon])

# === Training Loop ===
start_episode, epsilon = load_checkpoint(model, optimizer)

for episode in range(start_episode, EPISODES):
    board = chess.Board()
    done = False
    total_reward = 0
    previous_material = sum([PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == board.turn])
    piece_move_count = {}

    while not done:
        state_tensor = board_to_tensor(board)
        state_tensor = torch.tensor(state_tensor).unsqueeze(0).to(DEVICE)

        if random.random() < epsilon:
            move = random.choice(list(board.legal_moves))
        else:
            with torch.no_grad():
                q_values = model(state_tensor)[0].cpu()
            legal_moves = list(board.legal_moves)
            legal_indices = [move_encoder.move_to_index(m) for m in legal_moves if move_encoder.move_to_index(m) is not None]
            if not legal_indices:
                move = random.choice(legal_moves)
            else:
                best_index = max(legal_indices, key=lambda i: q_values[i])
                move = move_encoder.index_to_move(best_index, board)

        action_index = move_encoder.move_to_index(move)
        from_sq = move.from_square
        to_sq = move.to_square

        # === Refactored Reward Calculation ===
        reward = 0.0

        # Center control bonus
        if to_sq in CENTER_SQUARES:
            reward += 0.01

        # Penalize moving same piece repeatedly
        piece_id = (board.piece_at(from_sq), from_sq)
        piece_move_count[piece_id] = piece_move_count.get(piece_id, 0) + 1
        if piece_move_count[piece_id] > 2:
            reward -= 0.1

        # Bonus for castling
        if board.is_castling(move):
            reward += 0.1

        board.push(move)

        # Terminal reward
        if board.is_checkmate():
            reward += 1.0 if board.turn == chess.BLACK else -1.0
            done = True
        elif board.is_stalemate() or board.is_insufficient_material():
            reward -= 0.3
            done = True
        elif board.can_claim_draw():
            reward -= 0.2
            done = True

        # Surviving move (small incentive to make progress)
        reward += 0.01

        # Capturing piece reward
        if board.is_capture(move):
            captured_piece = board.piece_at(to_sq)
            if captured_piece and captured_piece.piece_type in PIECE_VALUES:
                reward += PIECE_VALUES[captured_piece.piece_type] * 0.1

        # Penalize lost material
        new_material = sum([PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color != board.turn])
        material_diff = new_material - previous_material
        if material_diff < 0:
            reward += 0.05 * material_diff
        previous_material = new_material

        # Check / being in check
        if board.is_check():
            reward -= 0.05
        board_copy = board.copy()
        board_copy.push(chess.Move.null())
        if board_copy.is_check():
            reward += 0.05

        # Penalize repetition
        if board.can_claim_threefold_repetition():
            reward -= 0.1

        reward = max(min(reward, 1.0), -1.0)

        next_state_tensor = board_to_tensor(board)
        if action_index is not None:
            replay_buffer.push(state_tensor.cpu().numpy(), action_index, reward, next_state_tensor, done)
        total_reward += reward

        if len(replay_buffer) >= BATCH_SIZE:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states = torch.tensor(states).squeeze(1).to(DEVICE)
            actions = [a for a in actions if a is not None]
            actions = torch.tensor(actions).unsqueeze(1).to(DEVICE)
            rewards = torch.tensor(rewards).unsqueeze(1).to(DEVICE)
            next_states = torch.tensor(next_states).squeeze(1).to(DEVICE)
            dones = torch.tensor(dones).unsqueeze(1).to(DEVICE)

            q_values = model(states).gather(1, actions)
            with torch.no_grad():
                max_next_q_values = target_model(next_states).max(1)[0].unsqueeze(1)
                target = rewards + GAMMA * max_next_q_values * (1 - dones)

            loss = criterion(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())
        save_checkpoint(model, optimizer, episode, epsilon)
        print(f"💾 Checkpoint saved at Episode {episode}")

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(f"Episode {episode + 1}/{EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
    log_reward_to_csv(episode + 1, total_reward, epsilon)
