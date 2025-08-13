import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.pgn
import random
import numpy as np
from collections import deque, defaultdict
import pickle
import os

class OpeningBook:
    """Opening book for better early game play"""
    
    def __init__(self):
        self.openings = self._create_opening_book()
        self.opening_exploration = 0.3  # 30% chance to try new openings
        self.learned_openings = set()
        
    def _create_opening_book(self):
        """Create comprehensive opening book"""
        openings = {
            # Standard starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1": [
                "e2e4", "d2d4", "g1f3", "c2c4", "b1c3"  # Main opening moves
            ],
            
            # After 1.e4
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": [
                "g1f3", "f2f4", "b1c3", "f1c4"  # King's pawn openings
            ],
            
            # Italian Game
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": [
                "f1c4", "f1b5", "d2d3"  # Italian, Spanish, King's Indian Attack
            ],
            
            # Sicilian Defense responses
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2": [
                "g1f3", "b1c3", "f2f4"  # Open Sicilian, Closed Sicilian, King's Gambit
            ],
            
            # French Defense responses
            "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2": [
                "e4e5", "d2d4", "b1c3"  # Advance, Exchange, Tarrasch
            ],
            
            # Queen's Gambit
            "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2": [
                "c2c4", "g1f3", "b1c3"  # Queen's Gambit, Queen's Pawn Game
            ],
            
            # Queen's Gambit Accepted/Declined
            "rnbqkbnr/ppp1pppp/8/8/2pP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3": [
                "g1f3", "e2e3", "b1c3"  # QGA main lines
            ],
        }
        return openings
    
    def get_opening_move(self, board):
        """Get opening move from book or explore new ones"""
        position_key = self._position_to_key(board)
        
        # Check if we have this position in our book
        if position_key in self.openings:
            available_moves = self.openings[position_key]
            
            # Filter for legal moves only
            legal_book_moves = []
            for move_uci in available_moves:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        legal_book_moves.append(move)
                except:
                    continue
            
            if legal_book_moves:
                # Sometimes explore, sometimes use book
                if random.random() < self.opening_exploration:
                    # Exploration: try a legal move not in book
                    other_moves = [m for m in board.legal_moves if m not in legal_book_moves]
                    if other_moves:
                        chosen_move = random.choice(other_moves)
                        self.learn_opening_move(position_key, chosen_move.uci())
                        return chosen_move
                
                # Use book move
                return random.choice(legal_book_moves)
        
        return None
    
    def learn_opening_move(self, position_key, move_uci):
        """Learn a new opening move"""
        if position_key not in self.openings:
            self.openings[position_key] = []
        
        if move_uci not in self.openings[position_key]:
            self.openings[position_key].append(move_uci)
            self.learned_openings.add((position_key, move_uci))
            print(f"📖 Learned new opening: {move_uci} in position {position_key[:20]}...")
    
    def _position_to_key(self, board):
        """Convert board position to key for opening book"""
        return board.fen()
    
    def save_openings(self, filepath="data/learned_openings.pkl"):
        """Save learned openings"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'openings': self.openings,
                'learned_openings': self.learned_openings
            }, f)
        print(f"💾 Saved {len(self.learned_openings)} learned openings")
    
    def load_openings(self, filepath="data/learned_openings.pkl"):
        """Load learned openings"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    self.openings.update(data.get('openings', {}))
                    self.learned_openings = data.get('learned_openings', set())
                print(f"📚 Loaded {len(self.learned_openings)} learned openings")
            except Exception as e:
                print(f"⚠️ Could not load openings: {e}")

class PrioritizedReplayBuffer:
    """Enhanced replay buffer with prioritization"""
    
    def __init__(self, capacity=50000, alpha=0.6, beta_start=0.4, beta_steps=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_steps = beta_steps
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, experience):
        """Add experience with maximum priority"""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch with prioritization"""
        if self.size < batch_size:
            return None
        
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        beta = self._compute_beta()
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities based on TD errors"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = abs(priority) + 1e-6
    
    def _compute_beta(self):
        """Compute beta for importance sampling"""
        fraction = min(self.frame / self.beta_steps, 1.0)
        beta = self.beta_start + fraction * (1.0 - self.beta_start)
        self.frame += 1
        return beta
    
    def __len__(self):
        return self.size

class ContinuousLearner:
    """Component for continuous learning from games"""
    
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.game_database = []
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=50, factor=0.5
        )
        
    def learn_from_human_game(self, moves, result, human_rating=1500):
        """Learn from a game against a human player"""
        # Weight learning based on opponent strength
        learning_weight = min(2.0, human_rating / 1500.0)
        
        # Process game moves
        board = chess.Board()
        positions = []
        
        for i, move in enumerate(moves):
            # Store position before move
            positions.append({
                'board': board.copy(),
                'move': move,
                'result': result,
                'move_number': i,
                'weight': learning_weight
            })
            board.push(move)
        
        # Add to database
        self.game_database.append({
            'positions': positions,
            'result': result,
            'opponent_rating': human_rating,
            'learning_weight': learning_weight
        })
        
        print(f"🎓 Learning from human game (rating: {human_rating}, weight: {learning_weight:.2f})")
        
        # Trigger learning if we have enough games
        if len(self.game_database) >= 5:
            self._update_from_recent_games()
    
    def _update_from_recent_games(self):
        """Update model from recent games"""
        from utils.utils import board_to_tensor
        from utils import move_encoder
        
        # Sample positions from recent games
        all_positions = []
        for game in self.game_database[-10:]:  # Use last 10 games
            all_positions.extend(game['positions'])
        
        # Randomly sample positions for training
        sample_size = min(100, len(all_positions))
        sampled_positions = random.sample(all_positions, sample_size)
        
        # Create training batch
        states = []
        targets = []
        
        for pos in sampled_positions:
            board = pos['board']
            move = pos['move']
            result = pos['result']
            weight = pos['weight']
            
            # Convert board to tensor
            state_tensor = board_to_tensor(board)
            states.append(state_tensor)
            
            # Create target based on game result
            move_index = move_encoder.move_to_index(move)
            if move_index is not None:
                # Reward based on game outcome
                if result == "1-0":  # White won
                    target_value = 1.0 if board.turn == chess.WHITE else -1.0
                elif result == "0-1":  # Black won
                    target_value = -1.0 if board.turn == chess.WHITE else 1.0
                else:  # Draw
                    target_value = 0.0
                
                targets.append((move_index, target_value * weight))
        
        if len(states) > 10:  # Only train if we have enough data
            self._train_batch(states, targets)
    
    def _train_batch(self, states, targets):
        """Train model on a batch of positions"""
        self.model.train()
        
        # Convert to tensors
        state_batch = torch.tensor(np.array(states)).to(self.device)
        
        # Forward pass
        q_values, position_values = self.model(state_batch)
        
        # Calculate losses
        total_loss = 0
        valid_targets = 0
        
        for i, (move_index, target_value) in enumerate(targets):
            if move_index is not None:
                predicted_q = q_values[i][move_index]
                target_q = torch.tensor(target_value).to(self.device)
                loss = nn.MSELoss()(predicted_q, target_q)
                total_loss += loss
                valid_targets += 1
        
        if valid_targets > 0:
            avg_loss = total_loss / valid_targets
            
            # Backward pass
            self.optimizer.zero_grad()
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            print(f"🔄 Continuous learning update - Loss: {avg_loss.item():.4f}")
        
        self.model.eval()

class AdaptiveTraining:
    """Adaptive training strategies"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=100)
        self.current_difficulty = 1.0
        self.difficulty_adjustment_rate = 0.1
        
    def adjust_difficulty(self, win_rate):
        """Adjust training difficulty based on performance"""
        self.performance_history.append(win_rate)
        
        if len(self.performance_history) >= 20:
            recent_performance = np.mean(list(self.performance_history)[-20:])
            
            # If performing too well, increase difficulty
            if recent_performance > 0.7:
                self.current_difficulty = min(2.0, self.current_difficulty + self.difficulty_adjustment_rate)
                print(f"📈 Increased training difficulty to {self.current_difficulty:.2f}")
            
            # If performing poorly, decrease difficulty
            elif recent_performance < 0.4:
                self.current_difficulty = max(0.5, self.current_difficulty - self.difficulty_adjustment_rate)
                print(f"📉 Decreased training difficulty to {self.current_difficulty:.2f}")
    
    def get_opponent_strength(self):
        """Get current opponent strength multiplier"""
        return self.current_difficulty
    
    def should_use_curriculum(self, episode):
        """Determine if curriculum learning should be used"""
        # Use curriculum for first 1000 episodes
        return episode < 1000

class GameAnalyzer:
    """Analyze games to identify weaknesses and improve"""
    
    def __init__(self):
        self.blunder_patterns = defaultdict(int)
        self.weakness_areas = {
            'opening': 0,
            'middlegame': 0,
            'endgame': 0,
            'tactics': 0,
            'positional': 0
        }
    
    def analyze_game(self, moves, result, our_color):
        """Analyze a completed game"""
        board = chess.Board()
        
        for i, move in enumerate(moves):
            # Determine game phase
            phase = self._get_game_phase(board, i)
            
            # Check for potential blunders
            if self._is_potential_blunder(board, move):
                self.blunder_patterns[phase] += 1
                if result != "1/2-1/2":  # Not a draw
                    self.weakness_areas[phase] += 1
            
            board.push(move)
        
        # Analyze final result
        self._analyze_result(result, our_color)
    
    def _get_game_phase(self, board, move_number):
        """Determine current game phase"""
        if move_number < 20:
            return 'opening'
        elif len(board.piece_map()) > 12:
            return 'middlegame'
        else:
            return 'endgame'
    
    def _is_potential_blunder(self, board, move):
        """Simple blunder detection"""
        # Check if move hangs a piece
        board_copy = board.copy()
        board_copy.push(move)
        
        # If the moved piece is immediately attacked
        if move.to_square in board_copy.attackers(not board.turn, move.to_square):
            piece = board_copy.piece_at(move.to_square)
            if piece and piece.color == board.turn:
                return True
        
        return False
    
    def _analyze_result(self, result, our_color):
        """Analyze game result"""
        if result == "1-0" and our_color == chess.BLACK:
            self.weakness_areas['overall'] = self.weakness_areas.get('overall', 0) + 1
        elif result == "0-1" and our_color == chess.WHITE:
            self.weakness_areas['overall'] = self.weakness_areas.get('overall', 0) + 1
    
    def get_training_focus(self):
        """Get recommended training focus"""
        if not self.weakness_areas:
            return 'balanced'
        
        max_weakness = max(self.weakness_areas.items(), key=lambda x: x[1])
        return max_weakness[0]
    
    def print_analysis_summary(self):
        """Print analysis summary"""
        print("🔍 Game Analysis Summary:")
        for area, count in self.weakness_areas.items():
            if count > 0:
                print(f"   {area}: {count} issues identified")
        
        focus = self.get_training_focus()
        print(f"💡 Recommended focus: {focus}")

class ModelEvaluator:
    """Evaluate model performance against different opponents"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_vs_random(self, agent, num_games=100):
        """Evaluate against random player"""
        wins = draws = losses = 0
        
        for game_num in range(num_games):
            board = chess.Board()
            agent_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
            
            while not board.is_game_over():
                if board.turn == agent_color:
                    move = agent.choose_move(board, temperature=0.1)
                else:
                    move = random.choice(list(board.legal_moves))
                
                if move:
                    board.push(move)
                else:
                    break
            
            result = board.result()
            if (result == "1-0" and agent_color == chess.WHITE) or (result == "0-1" and agent_color == chess.BLACK):
                wins += 1
            elif result == "1/2-1/2":
                draws += 1
            else:
                losses += 1
        
        win_rate = wins / num_games
        self.evaluation_history.append({
            'opponent': 'random',
            'games': num_games,
            'win_rate': win_rate,
            'wins': wins,
            'draws': draws,
            'losses': losses
        })
        
        print(f"🎯 Evaluation vs Random: {win_rate:.1%} win rate ({wins}W-{draws}D-{losses}L)")
        return win_rate
    
    def estimate_rating(self, win_rate_vs_random):
        """Estimate Elo rating based on performance"""
        # Rough estimation based on win rate against random play
        if win_rate_vs_random >= 0.95:
            estimated_rating = 1200 + (win_rate_vs_random - 0.95) * 2000
        elif win_rate_vs_random >= 0.8:
            estimated_rating = 800 + (win_rate_vs_random - 0.8) * 2667
        else:
            estimated_rating = 400 + win_rate_vs_random * 500
        
        estimated_rating = int(min(2500, estimated_rating))  # Cap at 2500
        print(f"📊 Estimated rating: ~{estimated_rating}")
        return estimated_rating
    
    def get_improvement_suggestions(self):
        """Get suggestions for improvement"""
        if not self.evaluation_history:
            return ["Run evaluations first"]
        
        latest = self.evaluation_history[-1]
        suggestions = []
        
        if latest['win_rate'] < 0.6:
            suggestions.append("Focus on basic tactics training")
            suggestions.append("Improve opening book")
        elif latest['win_rate'] < 0.8:
            suggestions.append("Work on positional understanding")
            suggestions.append("Add endgame knowledge")
        else:
            suggestions.append("Train against stronger opponents")
            suggestions.append("Fine-tune evaluation function")
        
        return suggestions