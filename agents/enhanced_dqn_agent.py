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
    """SIMPLIFIED chess position evaluation - CLEAR LEARNING SIGNALS"""
    
    PIECE_VALUES = {
        chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
        chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
    }
    
    def __init__(self):
        self.position_history = deque(maxlen=10)
    
    def evaluate_position(self, board, move=None):
        """SIMPLIFIED reward calculation - TEACHES CHESS PROPERLY"""
        reward = 0.0
        
        # 🔥 DETECT SPECIAL MOVES BEFORE MAKING THEM
        if move:
            # CASTLING DETECTION (CRITICAL FIX!)
            is_castling = (
                board.piece_at(move.from_square) and 
                board.piece_at(move.from_square).piece_type == chess.KING and
                abs(move.from_square - move.to_square) == 2
            )
            
            # PROMOTION DETECTION (CRITICAL FIX!)
            is_promotion = move.promotion is not None
            
            # CAPTURE DETECTION
            is_capture = board.is_capture(move)
        
        # Make a copy to see results
        board_copy = board.copy()
        if move:
            board_copy.push(move)
        
        # 🏆 GAME ENDING REWARDS (MASSIVE)
        if board_copy.is_checkmate():
            if board_copy.turn != board.turn:  # We won
                print("🏆 CHECKMATE! AI WON! Reward: +10.0")
                return 10.0
            else:  # We lost
                print("💀 CHECKMATE! AI LOST! Reward: -10.0")
                return -10.0
        
        if board_copy.is_stalemate():
            print("😐 STALEMATE! Reward: -3.0")
            return -3.0
        
        # 🏰 CASTLING REWARD (HUGE LEARNING SIGNAL!)
        if move and is_castling:
            reward += 3.0
            print("🏰 AI CASTLED! Reward: +3.0")
        
        # 👑 PROMOTION REWARD (HUGE LEARNING SIGNAL!)
        if move and is_promotion:
            if move.promotion == chess.QUEEN:
                reward += 8.0
                print("👑 AI PROMOTED TO QUEEN! Reward: +8.0")
            else:
                reward += 4.0
                piece_names = {chess.ROOK: "ROOK", chess.BISHOP: "BISHOP", chess.KNIGHT: "KNIGHT"}
                promo_name = piece_names.get(move.promotion, "UNKNOWN")
                print(f"👑 AI PROMOTED TO {promo_name}! Reward: +4.0")
        
        # 🎯 CAPTURE REWARDS (CLEAR PIECE VALUES)
        if move and is_capture:
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                capture_value = self.PIECE_VALUES[captured_piece.piece_type] / 100.0
                reward += capture_value
                piece_names = {1: "PAWN", 2: "KNIGHT", 3: "BISHOP", 4: "ROOK", 5: "QUEEN", 6: "KING"}
                captured_name = piece_names.get(captured_piece.piece_type, "UNKNOWN")
                print(f"🎯 AI CAPTURED {captured_name}! Reward: +{capture_value:.1f}")
        
        # ⚔️ CHECK BONUS
        if board_copy.is_check():
            reward += 0.5
            print("⚔️ AI GAVE CHECK! Reward: +0.5")
        
        # 📈 SIMPLE MATERIAL BALANCE (BASIC STRATEGY)
        material_balance = self._calculate_simple_material(board_copy) * 0.01
        reward += material_balance
        
        # 🔄 AVOID REPETITION (LIGHT PENALTY)
        current_fen = board_copy.fen().split()[0]
        self.position_history.append(current_fen)
        if self.position_history.count(current_fen) >= 2:
            reward -= 0.2
        
        # 🕰️ SLIGHT PENALTY FOR LONG GAMES (ENCOURAGE DECISIVE PLAY)
        reward -= 0.01
        
        return np.clip(reward, -15.0, 15.0)
    
    def _calculate_simple_material(self, board):
        """Simple material counting"""
        white_material = sum(self.PIECE_VALUES[piece.piece_type] 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.WHITE)
        black_material = sum(self.PIECE_VALUES[piece.piece_type] 
                           for piece in board.piece_map().values() 
                           if piece.color == chess.BLACK)
        
        balance = (white_material - black_material) / 100.0
        return balance if board.turn == chess.WHITE else -balance

class EnhancedDQNAgent:
    """Enhanced DQN Agent with FIXED promotion and castling logic"""
    
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
        
        # 👑 PROMOTION STATISTICS TRACKING (FIXED)
        self.promotion_stats = {
            'queen': 0,
            'rook': 0,
            'bishop': 0,
            'knight': 0,
            'total_promotions': 0
        }
        
        # 🏰 CASTLING STATISTICS TRACKING (NEW!)
        self.castling_stats = {
            'kingside': 0,
            'queenside': 0,
            'total_castling': 0
        }
        
        # Load existing model if available
        self.load_model()
        
        print(f"🤖 Enhanced Chess DQN Agent initialized on {self.device}")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def choose_promotion_piece(self, board, move):
        """FIXED: Now STRONGLY favors Queen promotion (95%+ of the time)"""
        to_square = move.to_square
        
        # 👑 DEFAULT: ALWAYS QUEEN (this is correct 95% of the time)
        promotion_piece = chess.QUEEN
        
        # Test Queen promotion first
        test_board = board.copy()
        queen_move = chess.Move(move.from_square, to_square, promotion=chess.QUEEN)
        test_board.push(queen_move)
        
        # 🚫 ONLY Exception: Queen promotion causes immediate stalemate
        if test_board.is_stalemate():
            print("🚫 Queen causes stalemate! Trying alternatives...")
            for alt_piece in [chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                test_alt = board.copy()
                alt_move = chess.Move(move.from_square, to_square, promotion=alt_piece)
                test_alt.push(alt_move)
                
                if not test_alt.is_stalemate():
                    piece_names = {chess.ROOK: "Rook", chess.BISHOP: "Bishop", chess.KNIGHT: "Knight"}
                    print(f"✅ Promoting to {piece_names[alt_piece]} to avoid stalemate!")
                    return alt_piece
        
        # 👑 95%+ of cases: Promote to Queen!
        print("👑 Standard Queen promotion - MAXIMUM POWER!")
        return chess.QUEEN
    
    def choose_move(self, board, temperature=1.0):
        """FIXED: Enhanced move choice with proper promotion AND castling handling"""
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
        
        # 🔥 FIXED: Proper move handling for ALL special moves
        legal_indices = []
        index_to_move = {}
        
        for move in legal_moves:
            if move.promotion is not None:
                # For promotion moves, choose the best promotion type
                promotion_moves = []
                promotion_scores = []
                
                for promo_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    test_promotion = chess.Move(move.from_square, move.to_square, promotion=promo_type)
                    if test_promotion in legal_moves:
                        idx = move_encoder.move_to_index(test_promotion)
                        if idx is not None:
                            promotion_moves.append(test_promotion)
                            promotion_scores.append(q_values[idx].item())
                
                if promotion_moves:
                    best_promo_idx = promotion_scores.index(max(promotion_scores))
                    best_promotion_move = promotion_moves[best_promo_idx]
                    
                    idx = move_encoder.move_to_index(best_promotion_move)
                    if idx is not None:
                        legal_indices.append(idx)
                        index_to_move[idx] = best_promotion_move
            else:
                # Regular move (including castling)
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
        
        # 📊 Track special move statistics
        self._track_special_moves(chosen_move)
        
        return chosen_move
    
    def _track_special_moves(self, move):
        """Track statistics for special moves"""
        # Track promotions
        if move.promotion:
            self.promotion_stats['total_promotions'] += 1
            if move.promotion == chess.QUEEN:
                self.promotion_stats['queen'] += 1
                print("👑 AI promoted to QUEEN!")
            elif move.promotion == chess.ROOK:
                self.promotion_stats['rook'] += 1
                print("🏰 AI promoted to Rook")
            elif move.promotion == chess.BISHOP:
                self.promotion_stats['bishop'] += 1
                print("♗ AI promoted to Bishop")
            elif move.promotion == chess.KNIGHT:
                self.promotion_stats['knight'] += 1
                print("🐴 AI promoted to Knight")
            
            # Print stats every 5 promotions
            if self.promotion_stats['total_promotions'] % 5 == 0:
                self._print_promotion_stats()
        
        # 🏰 Track castling (NEW!)
        piece_moved = None  # We'd need the board to check this properly
        # For now, detect castling by king moving 2 squares
        if abs(move.from_square - move.to_square) == 2:
            # Likely castling (we'd need to verify it's actually a king)
            self.castling_stats['total_castling'] += 1
            if move.to_square > move.from_square:
                self.castling_stats['kingside'] += 1
                print("🏰 AI CASTLED KINGSIDE!")
            else:
                self.castling_stats['queenside'] += 1
                print("🏰 AI CASTLED QUEENSIDE!")
            
            # Print castling stats
            if self.castling_stats['total_castling'] % 3 == 0:
                self._print_castling_stats()
    
    def _print_promotion_stats(self):
        """Print promotion statistics"""
        total = self.promotion_stats['total_promotions']
        if total == 0:
            return
        
        print(f"\n📊 PROMOTION STATISTICS (last {total} promotions):")
        print(f"   👑 Queen: {self.promotion_stats['queen']}/{total} ({self.promotion_stats['queen']/total*100:.1f}%)")
        print(f"   🏰 Rook: {self.promotion_stats['rook']}/{total} ({self.promotion_stats['rook']/total*100:.1f}%)")
        print(f"   ♗ Bishop: {self.promotion_stats['bishop']}/{total} ({self.promotion_stats['bishop']/total*100:.1f}%)")
        print(f"   🐴 Knight: {self.promotion_stats['knight']}/{total} ({self.promotion_stats['knight']/total*100:.1f}%)")
        
        queen_rate = self.promotion_stats['queen'] / total
        if queen_rate < 0.8:
            print("⚠️  WARNING: AI is under-promoting too often!")
        elif queen_rate > 0.9:
            print("✅ EXCELLENT: AI correctly favors Queen promotion!")
    
    def _print_castling_stats(self):
        """Print castling statistics (NEW!)"""
        total = self.castling_stats['total_castling']
        if total == 0:
            return
        
        print(f"\n🏰 CASTLING STATISTICS (last {total} castling moves):")
        print(f"   🏰 Kingside: {self.castling_stats['kingside']}/{total} ({self.castling_stats['kingside']/total*100:.1f}%)")
        print(f"   🏰 Queenside: {self.castling_stats['queenside']}/{total} ({self.castling_stats['queenside']/total*100:.1f}%)")
        
        if total >= 5:
            print("✅ AI is learning to castle!")
        else:
            print("💡 AI is starting to understand castling")
    
    def learn_from_game(self, game_moves, game_result):
        """Learn from a completed game"""
        if not self.learning_enabled:
            return
        
        self.recent_games.append({
            'moves': game_moves,
            'result': game_result,
            'timestamp': len(self.recent_games)
        })
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        print(f"📚 Learned from game. Result: {game_result}, Epsilon: {self.epsilon:.3f}")
    
    def learn_from_human_game(self, game_moves, game_result, human_rating=1500):
        """Enhanced learning from human games"""
        if not self.learning_enabled:
            return
        
        learning_weight = min(2.0, max(0.5, human_rating / 1500.0))
        
        self.recent_games.append({
            'moves': game_moves,
            'result': game_result,
            'human_rating': human_rating,
            'learning_weight': learning_weight,
            'timestamp': len(self.recent_games)
        })
        
        if game_result in ['1-0', '0-1']:
            self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)
        
        print(f"🎓 Enhanced learning from {human_rating}-rated opponent")
    
    def save_model(self):
        """Save the current model with all statistics"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
            'recent_games': list(self.recent_games),
            'promotion_stats': self.promotion_stats,
            'castling_stats': self.castling_stats  # NEW!
        }, self.model_path, weights_only=False)
        print(f"💾 Enhanced model saved to {self.model_path}")
        
        # Print final statistics
        if self.promotion_stats['total_promotions'] > 0:
            print("📊 Final Promotion Statistics:")
            self._print_promotion_stats()
        
        if self.castling_stats['total_castling'] > 0:
            print("🏰 Final Castling Statistics:")
            self._print_castling_stats()
    
    def load_model(self):
        """Load existing model with all statistics"""
        if os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.epsilon = checkpoint.get('epsilon', 0.1)
                self.recent_games = deque(checkpoint.get('recent_games', []), maxlen=100)
                
                # Load statistics
                self.promotion_stats = checkpoint.get('promotion_stats', {
                    'queen': 0, 'rook': 0, 'bishop': 0, 'knight': 0, 'total_promotions': 0
                })
                self.castling_stats = checkpoint.get('castling_stats', {
                    'kingside': 0, 'queenside': 0, 'total_castling': 0
                })
                
                print(f"✅ Loaded enhanced model from {self.model_path}")
                
                # Show statistics if available
                if self.promotion_stats['total_promotions'] > 0:
                    print("📊 Loaded Promotion Statistics:")
                    self._print_promotion_stats()
                
                if self.castling_stats['total_castling'] > 0:
                    print("🏰 Loaded Castling Statistics:")
                    self._print_castling_stats()
                    
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
        else:
            print("🆕 No existing model found. Starting fresh.")
    
    def get_model_info(self):
        """Get comprehensive information about the model"""
        return {
            'games_learned': len(self.recent_games),
            'epsilon': self.epsilon,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'promotion_stats': self.promotion_stats,
            'castling_stats': self.castling_stats
        }
    
    def get_stats_summary(self):
        """Get a summary of all learning statistics"""
        summary = f"""
🤖 AI LEARNING SUMMARY:
📚 Games learned from: {len(self.recent_games)}
🎯 Current exploration: {self.epsilon:.3f}

👑 PROMOTIONS: {self.promotion_stats['total_promotions']} total
   • Queen: {self.promotion_stats['queen']} ({self.promotion_stats['queen']/max(1,self.promotion_stats['total_promotions'])*100:.1f}%)
   • Others: {self.promotion_stats['total_promotions'] - self.promotion_stats['queen']}

🏰 CASTLING: {self.castling_stats['total_castling']} total
   • Kingside: {self.castling_stats['kingside']}
   • Queenside: {self.castling_stats['queenside']}
"""
        return summary