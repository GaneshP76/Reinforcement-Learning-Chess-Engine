import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np
import random
from collections import deque
import pickle
import os

class EnhancedDQNModel(nn.Module):
    """Enhanced DQN with better architecture for chess understanding"""
    
    def __init__(self):
        super(EnhancedDQNModel, self).__init__()
        
        # Enhanced convolution layers - more filters for pattern recognition
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        
        # More residual blocks for deeper chess understanding
        self.res_blocks = nn.ModuleList([
            self._make_residual_block(256) for _ in range(4)
        ])
        
        # Attention mechanism for important squares
        self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
        
        # Policy head (what moves to consider)
        self.policy_conv = nn.Conv2d(256, 128, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(128)
        
        # Value head (how good is the position)
        self.value_conv = nn.Conv2d(256, 64, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(64)
        
        # Final layers
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(64 * 8 * 8, 512)
        self.value_head = nn.Linear(512, 1)
        
        # Q-values output
        self.q_head = nn.Linear(1024 + 1, 4672)
        self.dropout = nn.Dropout(0.3)
        
    def _make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Residual blocks for pattern recognition
        for res_block in self.res_blocks:
            residual = x
            x = F.relu(res_block(x) + residual)
        
        # Apply attention to focus on important squares
        x_flat = x.view(batch_size, 256, -1).transpose(1, 2)
        attended, _ = self.attention(x_flat, x_flat, x_flat)
        x = attended.transpose(1, 2).view(batch_size, 256, 8, 8)
        
        # Policy head (move preferences)
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(batch_size, -1)
        policy = self.dropout(F.relu(self.fc1(policy)))
        
        # Value head (position evaluation)
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(batch_size, -1)
        value = torch.tanh(self.value_head(self.fc2(value)))
        
        # Combine for final Q-values
        combined = torch.cat([policy, value], dim=1)
        q_values = self.q_head(combined)
        
        return q_values, value

class ChessEvaluator:
    """Enhanced chess position evaluation"""
    
    PIECE_VALUES = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
    }
    
    # Piece-square tables for better positional play
    PAWN_TABLE = np.array([
        [0,  0,  0,  0,  0,  0,  0,  0],
        [78, 83, 86, 73, 102, 82, 85, 90],
        [7, 29, 21, 44, 40, 31, 44, 7],
        [-17, 16, -2, 15, 14, 0, 15, -13],
        [-26, 3, 10, 9, 6, 1, 0, -23],
        [-22, 9, 5, -11, -10, -2, 3, -19],
        [-31, 8, -7, -37, -36, -14, 3, -31],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    
    def __init__(self):
        self.position_history = deque(maxlen=10)
    
    def evaluate_position(self, board, move=None):
        """Calculate reward for the current position"""
        reward = 0.0
        
        # Material balance
        material_balance = self._calculate_material_balance(board)
        reward += material_balance * 0.01
        
        # Piece activity and positioning
        reward += self._evaluate_piece_activity(board) * 0.005
        
        # King safety evaluation
        reward += self._evaluate_king_safety(board) * 0.01
        
        # Center control
        reward += self._evaluate_center_control(board) * 0.003
        
        # Pawn structure
        reward += self._evaluate_pawn_structure(board) * 0.002
        
        # Development bonus in opening
        if len(board.move_stack) < 20:
            reward += self._evaluate_development(board) * 0.005
        
        # Avoid repetition
        current_fen = board.fen().split()[0]
        self.position_history.append(current_fen)
        if self.position_history.count(current_fen) >= 2:
            reward -= 0.1
        
        return np.clip(reward, -2.0, 2.0)
    
    def _calculate_material_balance(self, board):
        """Calculate material advantage"""
        white_material = sum(self.PIECE_VALUES[piece.piece_type] 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE)
        black_material = sum(self.PIECE_VALUES[piece.piece_type] 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK)
        
        balance = (white_material - black_material) / 100.0
        return balance if board.turn == chess.WHITE else -balance
    
    def _evaluate_piece_activity(self, board):
        """Evaluate how active pieces are"""
        score = 0
        for square, piece in board.piece_map().items():
            if piece.color == board.turn:
                # Positional bonus using piece-square tables
                row, col = divmod(square, 8)
                if piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        score += self.PAWN_TABLE[7-row][col] / 10
                    else:
                        score += self.PAWN_TABLE[row][col] / 10
                
                # Mobility bonus
                piece_mobility = self._count_piece_moves(board, square)
                score += piece_mobility * 2
        
        return score / 100.0
    
    def _count_piece_moves(self, board, square):
        """Count legal moves for a piece"""
        count = 0
        for move in board.legal_moves:
            if move.from_square == square:
                count += 1
        return count
    
    def _evaluate_king_safety(self, board):
        """Evaluate king safety"""
        king_square = board.king(board.turn)
        if king_square is None:
            return -10.0
        
        safety_score = 0
        
        # Penalize exposed king
        attackers = len(board.attackers(not board.turn, king_square))
        safety_score -= attackers * 20
        
        # Reward castling rights
        if board.has_kingside_castling_rights(board.turn):
            safety_score += 15
        if board.has_queenside_castling_rights(board.turn):
            safety_score += 10
        
        return safety_score / 100.0
    
    def _evaluate_center_control(self, board):
        """Evaluate control of center squares"""
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        score = 0
        
        for square in center_squares:
            # Count attackers and defenders
            our_attackers = len(board.attackers(board.turn, square))
            their_attackers = len(board.attackers(not board.turn, square))
            score += (our_attackers - their_attackers) * 10
        
        return score / 100.0
    
    def _evaluate_pawn_structure(self, board):
        """Evaluate pawn structure"""
        score = 0
        our_pawns = [sq for sq, piece in board.piece_map().items() 
                    if piece.piece_type == chess.PAWN and piece.color == board.turn]
        
        # Penalize doubled pawns
        files = [chess.square_file(sq) for sq in our_pawns]
        for file_idx in range(8):
            file_count = files.count(file_idx)
            if file_count > 1:
                score -= file_count * 15
        
        # Reward passed pawns
        for pawn_sq in our_pawns:
            if self._is_passed_pawn(board, pawn_sq, board.turn):
                score += 25
        
        return score / 100.0
    
    def _is_passed_pawn(self, board, pawn_square, color):
        """Check if pawn is passed (no enemy pawns can stop it)"""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check adjacent files and same file for enemy pawns
        check_files = [f for f in [file-1, file, file+1] if 0 <= f <= 7]
        
        for check_file in check_files:
            for check_rank in range(8):
                check_square = chess.square(check_file, check_rank)
                piece = board.piece_at(check_square)
                if (piece and piece.piece_type == chess.PAWN and 
                    piece.color != color):
                    if color == chess.WHITE and check_rank > rank:
                        return False
                    if color == chess.BLACK and check_rank < rank:
                        return False
        
        return True
    
    def _evaluate_development(self, board):
        """Evaluate piece development in opening"""
        score = 0
        
        # Count developed pieces
        development_squares = {
            chess.WHITE: [chess.B1, chess.C1, chess.F1, chess.G1],
            chess.BLACK: [chess.B8, chess.C8, chess.F8, chess.G8]
        }
        
        for square in development_squares[board.turn]:
            piece = board.piece_at(square)
            if not piece or piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                if not piece:  # Piece has moved from starting square
                    score += 10
        
        return score / 100.0

class EnhancedDQNAgent:
    """Enhanced DQN Agent with continuous learning capabilities"""
    
    def __init__(self, model_path="data/enhanced_dqn_checkpoint.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EnhancedDQNModel().to(self.device)
        self.evaluator = ChessEvaluator()
        self.model_path = model_path
        
        # Learning parameters
        self.epsilon = 0.1  # Start with some exploration
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        
        # Continuous learning buffer
        self.recent_games = deque(maxlen=100)
        self.learning_enabled = True
        
        # Load existing model if available
        self.load_model()
        
        print(f"🤖 Enhanced Chess DQN Agent initialized on {self.device}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def choose_move(self, board, temperature=1.0):
        """Choose the best move using the enhanced model"""
        from utils.utils import board_to_tensor
        from utils import move_encoder
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Convert board to tensor
        state_tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, position_value = self.model(state_tensor)
            q_values = q_values[0]
        
        # Get legal move indices
        legal_indices = []
        index_to_move = {}
        
        for move in legal_moves:
            idx = move_encoder.move_to_index(move)
            if idx is not None:
                legal_indices.append(idx)
                index_to_move[idx] = move
        
        if not legal_indices:
            return random.choice(legal_moves)
        
        # Apply temperature for move selection diversity
        if temperature > 0 and random.random() < self.epsilon:
            # Exploration: weighted random selection
            legal_q_values = [q_values[i].item() for i in legal_indices]
            # Apply temperature
            scaled_values = [v / temperature for v in legal_q_values]
            max_val = max(scaled_values)
            exp_values = [np.exp(v - max_val) for v in scaled_values]
            probs = [v / sum(exp_values) for v in exp_values]
            
            chosen_idx = np.random.choice(len(legal_indices), p=probs)
            best_index = legal_indices[chosen_idx]
        else:
            # Exploitation: best move
            best_index = max(legal_indices, key=lambda i: q_values[i])
        
        return index_to_move[best_index]
    
    def learn_from_game(self, game_moves, game_result):
        """Learn from a completed game"""
        if not self.learning_enabled:
            return
        
        # Store game for continuous learning
        self.recent_games.append({
            'moves': game_moves,
            'result': game_result,
            'timestamp': len(self.recent_games)
        })
        
        # Update exploration
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        print(f"📚 Learned from game. Result: {game_result}, Epsilon: {self.epsilon:.3f}")
    
    def save_model(self):
        """Save the current model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
            'recent_games': list(self.recent_games)
        }, self.model_path)
        print(f"💾 Enhanced model saved to {self.model_path}")
    
    def load_model(self):
        """Load existing model if available"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.epsilon = checkpoint.get('epsilon', 0.1)
                self.recent_games = deque(checkpoint.get('recent_games', []), maxlen=100)
                print(f"✅ Loaded enhanced model from {self.model_path}")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
        else:
            print("🆕 No existing model found. Starting fresh.")
    
    def get_model_info(self):
        """Get information about the current model"""
        return {
            'games_learned': len(self.recent_games),
            'epsilon': self.epsilon,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters())
        }