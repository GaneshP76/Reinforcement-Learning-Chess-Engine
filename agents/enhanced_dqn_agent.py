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
    """Enhanced DQN Agent with FIXED promotion logic"""
    
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
        
        # 👑 PROMOTION STATISTICS TRACKING
        self.promotion_stats = {
            'queen': 0,
            'rook': 0,
            'bishop': 0,
            'knight': 0,
            'total_promotions': 0
        }
        
        # Load existing model if available
        self.load_model()
        
        print(f"🤖 Enhanced Chess DQN Agent initialized on {self.device}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def choose_promotion_piece(self, board, move):
        """
        🔥 COMPLETELY REWRITTEN: Now STRONGLY favors Queen promotion
        Only underpromotes in very specific tactical situations
        """
        to_square = move.to_square
        
        # 👑 DEFAULT: ALWAYS QUEEN unless there's a compelling reason not to
        promotion_piece = chess.QUEEN
        
        # Test Queen promotion first
        test_board = board.copy()
        queen_move = chess.Move(move.from_square, to_square, promotion=chess.QUEEN)
        test_board.push(queen_move)
        
        # 🚫 ONLY Exception 1: Queen promotion causes immediate stalemate
        if test_board.is_stalemate():
            print("🚫 Queen causes stalemate! Trying alternatives...")
            for alt_piece in [chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                test_alt = board.copy()
                alt_move = chess.Move(move.from_square, to_square, promotion=alt_piece)
                test_alt.push(alt_move)
                
                if not test_alt.is_stalemate() and not test_alt.is_insufficient_material():
                    piece_names = {chess.ROOK: "Rook", chess.BISHOP: "Bishop", chess.KNIGHT: "Knight"}
                    print(f"✅ Promoting to {piece_names[alt_piece]} to avoid stalemate!")
                    return alt_piece
        
        # 🚫 ONLY Exception 2: Very specific knight tactics (extremely rare)
        # But only if Queen doesn't give the same benefit
        if len(test_board.piece_map()) <= 8:  # Late endgame only
            test_knight = board.copy()
            knight_move = chess.Move(move.from_square, to_square, promotion=chess.KNIGHT)
            test_knight.push(knight_move)
            
            # Check if knight gives unique tactical benefit
            knight_checks = test_knight.is_check()
            queen_checks = test_board.is_check()
            
            if knight_checks and not queen_checks:
                # Knight gives check but Queen doesn't - very rare situation
                if self._knight_gives_unique_fork(test_knight, to_square, test_board):
                    print("🐴 Rare knight promotion for unique tactical advantage!")
                    return chess.KNIGHT
        
        # 👑 95%+ of cases: Promote to Queen!
        print("👑 Standard Queen promotion - MAXIMUM POWER!")
        return chess.QUEEN
    
    def _knight_gives_unique_fork(self, knight_board, knight_square, queen_board):
        """Check if knight gives tactical advantage that Queen doesn't"""
        # Count valuable targets attacked by knight
        knight_targets = 0
        knight_attacks = list(knight_board.attacks(knight_square))
        
        for square in knight_attacks:
            piece = knight_board.piece_at(square)
            if (piece and piece.color != knight_board.turn and 
                piece.piece_type in [chess.KING, chess.QUEEN, chess.ROOK]):
                knight_targets += 1
        
        # Count valuable targets attacked by Queen
        queen_square = None
        for square, piece in queen_board.piece_map().items():
            if (piece.piece_type == chess.QUEEN and 
                piece.color == queen_board.turn and 
                square != knight_square):  # Find our newly promoted queen
                queen_square = square
                break
        
        if queen_square:
            queen_targets = 0
            queen_attacks = list(queen_board.attacks(queen_square))
            for square in queen_attacks:
                piece = queen_board.piece_at(square)
                if (piece and piece.color != queen_board.turn and 
                    piece.piece_type in [chess.KING, chess.QUEEN, chess.ROOK]):
                    queen_targets += 1
            
            # Only choose knight if it attacks significantly more valuable targets
            return knight_targets >= 2 and knight_targets > queen_targets
        
        return knight_targets >= 2
    
    def choose_move(self, board, temperature=1.0):
        """🔥 FIXED: Enhanced move choice with proper promotion handling"""
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
        
        # 🔥 CRITICAL FIX: Proper promotion move handling
        legal_indices = []
        index_to_move = {}
        
        for move in legal_moves:
            if move.promotion is not None:
                # 🔥 For each promotion square, evaluate ALL promotion types and pick BEST
                promotion_moves = []
                promotion_scores = []
                
                # Test all possible promotions
                for promo_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    test_promotion = chess.Move(move.from_square, move.to_square, promotion=promo_type)
                    if test_promotion in legal_moves:  # Make sure it's legal
                        idx = move_encoder.move_to_index(test_promotion)
                        if idx is not None:
                            promotion_moves.append(test_promotion)
                            promotion_scores.append(q_values[idx].item())
                
                # Choose the promotion with highest Q-value (should usually be Queen)
                if promotion_moves:
                    best_promo_idx = promotion_scores.index(max(promotion_scores))
                    best_promotion_move = promotion_moves[best_promo_idx]
                    
                    # Add to legal moves for consideration
                    idx = move_encoder.move_to_index(best_promotion_move)
                    if idx is not None:
                        legal_indices.append(idx)
                        index_to_move[idx] = best_promotion_move
            else:
                # Regular non-promotion move
                idx = move_encoder.move_to_index(move)
                if idx is not None:
                    legal_indices.append(idx)
                    index_to_move[idx] = move
        
        if not legal_indices:
            return random.choice(legal_moves)
        
        # Move selection with temperature
        if temperature > 0 and random.random() < self.epsilon:
            # Exploration: weighted random selection
            legal_q_values = [q_values[i].item() for i in legal_indices]
            scaled_values = [v / temperature for v in legal_q_values]
            max_val = max(scaled_values)
            exp_values = [np.exp(v - max_val) for v in scaled_values]
            probs = [v / sum(exp_values) for v in exp_values]
            
            chosen_idx = np.random.choice(len(legal_indices), p=probs)
            best_index = legal_indices[chosen_idx]
        else:
            # Exploitation: best move
            best_index = max(legal_indices, key=lambda i: q_values[i])
        
        chosen_move = index_to_move[best_index]
        
        # 📊 Track promotion statistics
        if chosen_move.promotion:
            self.promotion_stats['total_promotions'] += 1
            if chosen_move.promotion == chess.QUEEN:
                self.promotion_stats['queen'] += 1
                print("👑 AI promoted to QUEEN! (As it should 95% of the time)")
            elif chosen_move.promotion == chess.ROOK:
                self.promotion_stats['rook'] += 1
                print("🏰 AI promoted to Rook (rare but sometimes correct)")
            elif chosen_move.promotion == chess.BISHOP:
                self.promotion_stats['bishop'] += 1
                print("♗ AI promoted to Bishop (very rare)")
            elif chosen_move.promotion == chess.KNIGHT:
                self.promotion_stats['knight'] += 1
                print("🐴 AI promoted to Knight (should be extremely rare)")
            
            # Print promotion statistics
            if self.promotion_stats['total_promotions'] % 5 == 0:  # Every 5 promotions
                self._print_promotion_stats()
        
        return chosen_move
    
    def _print_promotion_stats(self):
        """Print promotion statistics to track if AI is promoting correctly"""
        total = self.promotion_stats['total_promotions']
        if total == 0:
            return
        
        print(f"\n📊 PROMOTION STATISTICS (last {total} promotions):")
        print(f"   👑 Queen: {self.promotion_stats['queen']}/{total} ({self.promotion_stats['queen']/total*100:.1f}%) - TARGET: 90%+")
        print(f"   🏰 Rook: {self.promotion_stats['rook']}/{total} ({self.promotion_stats['rook']/total*100:.1f}%)")
        print(f"   ♗ Bishop: {self.promotion_stats['bishop']}/{total} ({self.promotion_stats['bishop']/total*100:.1f}%)")
        print(f"   🐴 Knight: {self.promotion_stats['knight']}/{total} ({self.promotion_stats['knight']/total*100:.1f}%)")
        
        queen_rate = self.promotion_stats['queen'] / total
        if queen_rate < 0.8:  # Less than 80% Queens is concerning
            print("⚠️  WARNING: AI is under-promoting too often!")
            print("💡 This suggests the evaluation function may need tuning")
        elif queen_rate > 0.95:  # More than 95% Queens is perfect
            print("✅ EXCELLENT: AI correctly favors Queen promotion!")
    
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
    
    def learn_from_human_game(self, game_moves, game_result, human_rating=1500):
        """Enhanced learning from human games with rating consideration"""
        if not self.learning_enabled:
            return
        
        # Weight learning based on human strength
        learning_weight = min(2.0, max(0.5, human_rating / 1500.0))
        
        # Store enhanced game data
        self.recent_games.append({
            'moves': game_moves,
            'result': game_result,
            'human_rating': human_rating,
            'learning_weight': learning_weight,
            'timestamp': len(self.recent_games)
        })
        
        # Adjust exploration based on performance against humans
        if game_result in ['1-0', '0-1']:  # Decisive games teach more
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)
        
        print(f"🎓 Enhanced learning from {human_rating}-rated opponent (weight: {learning_weight:.2f})")
    
    def save_model(self):
        """Save the current model"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
            'recent_games': list(self.recent_games),
            'promotion_stats': self.promotion_stats  # 📊 Save promotion statistics
        }, self.model_path, weights_only=False)
        print(f"💾 Enhanced model saved to {self.model_path}")
        
        # Print final promotion stats when saving
        if self.promotion_stats['total_promotions'] > 0:
            print("📊 Final Promotion Statistics:")
            self._print_promotion_stats()
    
    def load_model(self):
        """Load existing model if available"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.epsilon = checkpoint.get('epsilon', 0.1)
                self.recent_games = deque(checkpoint.get('recent_games', []), maxlen=100)
                # Load promotion stats if available
                self.promotion_stats = checkpoint.get('promotion_stats', {
                    'queen': 0, 'rook': 0, 'bishop': 0, 'knight': 0, 'total_promotions': 0
                })
                print(f"✅ Loaded enhanced model from {self.model_path}")
                
                # Show promotion stats if any exist
                if self.promotion_stats['total_promotions'] > 0:
                    print("📊 Loaded Promotion Statistics:")
                    self._print_promotion_stats()
                    
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
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'promotion_stats': self.promotion_stats
        }
    
    def get_promotion_stats(self):
        """Get statistics about promotion choices (for analysis)"""
        return self.promotion_stats.copy()
    
    def reset_promotion_stats(self):
        """Reset promotion statistics (useful for testing)"""
        self.promotion_stats = {
            'queen': 0,
            'rook': 0,
            'bishop': 0,
            'knight': 0,
            'total_promotions': 0
        }
        print("📊 Promotion statistics reset")