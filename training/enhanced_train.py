import torch
import torch.nn as nn
import torch.optim as optim
import chess
import random
import numpy as np
import os
import csv
from tqdm import tqdm
import time

# Import our enhanced components
from agents.enhanced_dqn_agent import EnhancedDQNModel, ChessEvaluator, EnhancedDQNAgent
from training.training_components import OpeningBook, PrioritizedReplayBuffer, ContinuousLearner, AdaptiveTraining, GameAnalyzer, ModelEvaluator
from utils.utils import board_to_tensor
from utils import move_encoder

class EnhancedChessTrainer:
    """Enhanced trainer with continuous learning capabilities"""
    
    def __init__(self):
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Enhanced Chess DQN Trainer initialized on {self.device}")
        
        # Initialize models
        self.model = EnhancedDQNModel().to(self.device)
        self.target_model = EnhancedDQNModel().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Initialize components
        self.evaluator = ChessEvaluator()
        self.opening_book = OpeningBook()
        self.replay_buffer = PrioritizedReplayBuffer()
        self.adaptive_training = AdaptiveTraining()
        self.game_analyzer = GameAnalyzer()
        self.model_evaluator = ModelEvaluator()
        
        # Training parameters
        self.learning_rate = 1e-4
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize continuous learner
        self.continuous_learner = ContinuousLearner(self.model, self.optimizer, self.device)
        
        # Training hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 32
        self.target_update_freq = 10
        
        # Paths
        self.checkpoint_path = "data/enhanced_dqn_checkpoint.pth"
        self.reward_log_path = "data/enhanced_reward_log.csv"
        
        # Load existing progress
        self.load_checkpoint()
        self.opening_book.load_openings()
        
        print("✅ All components initialized successfully!")
    
    def train(self, episodes=3000, save_freq=50, eval_freq=200):
        """Main training loop with all enhancements"""
        print(f"🎯 Starting enhanced training for {episodes} episodes")
        print(f"📊 Current model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_episode = self.load_episode_number()
        best_win_rate = 0.0
        
        # Training loop with progress bar
        for episode in tqdm(range(start_episode, episodes), desc="Training Progress"):
            # Play one training episode
            episode_reward, game_moves, game_result = self._play_training_episode(episode)
            
            # Learn from the episode
            self._learn_from_episode()
            
            # Analyze the game
            self.game_analyzer.analyze_game(game_moves, game_result, chess.WHITE)
            
            # Update target network
            if episode % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
            
            # Save progress
            if episode % save_freq == 0:
                self.save_checkpoint(episode)
                self.opening_book.save_openings()
                
            # Evaluate model
            if episode % eval_freq == 0 and episode > 0:
                win_rate = self.model_evaluator.evaluate_vs_random(
                    EnhancedDQNAgent(self.checkpoint_path), num_games=100
                )
                
                # Adjust training difficulty
                self.adaptive_training.adjust_difficulty(win_rate)
                
                # Save best model
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    self.save_best_model()
                    print(f"🌟 New best model! Win rate: {win_rate:.1%}")
                
                # Estimate rating
                estimated_rating = self.model_evaluator.estimate_rating(win_rate)
                
                # Log progress
                self._log_progress(episode, episode_reward, win_rate, estimated_rating)
            
            # Update exploration
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print periodic updates
            if episode % 50 == 0:
                focus = self.game_analyzer.get_training_focus()
                print(f"Episode {episode}: Reward={episode_reward:.2f}, ε={self.epsilon:.3f}, Focus={focus}")
        
        print("🎉 Training completed!")
        self._print_final_summary()
    
    def _play_training_episode(self, episode):
        """Play one training episode with all enhancements"""
        board = chess.Board()
        total_reward = 0.0
        moves_played = []
        episode_experiences = []
        
        move_count = 0
        max_moves = 200  # Prevent infinite games
        
        while not board.is_game_over() and move_count < max_moves:
            # Get current state
            state = board_to_tensor(board)
            state_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
            
            # Choose move
            if board.turn == chess.WHITE:
                # Our agent's turn
                move = self._choose_agent_move(board, episode)
            else:
                # Opponent's turn (adaptive difficulty)
                move = self._choose_opponent_move(board, episode)
            
            if move is None:
                break
            
            # Store the move
            moves_played.append(move)
            
            # Calculate reward
            reward = self._calculate_reward(board, move)
            
            # Make the move
            board.push(move)
            
            # Store experience for our agent's moves only
            if len(moves_played) > 1 and move_count % 2 == 0:  # Our moves
                next_state = board_to_tensor(board)
                action_index = move_encoder.move_to_index(moves_played[-1])
                
                if action_index is not None:
                    experience = (
                        state,
                        action_index,
                        reward,
                        next_state,
                        board.is_game_over()
                    )
                    self.replay_buffer.push(experience)
                    episode_experiences.append((state_tensor, reward))
            
            total_reward += reward
            move_count += 1
        
        # Determine game result
        game_result = board.result()
        
        # Add terminal reward
        if game_result == "1-0":  # We won
            total_reward += 2.0
        elif game_result == "0-1":  # We lost
            total_reward -= 2.0
        else:  # Draw
            total_reward += 0.5
        
        return total_reward, moves_played, game_result
    
    def _choose_agent_move(self, board, episode):
        """Choose move for our agent"""
        # Use opening book in early game
        if len(board.move_stack) < 12:
            opening_move = self.opening_book.get_opening_move(board)
            if opening_move:
                return opening_move
        
        # Use epsilon-greedy with model
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # Exploration vs exploitation
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
        
        # Use model to choose move
        state_tensor = torch.tensor(board_to_tensor(board)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values, _ = self.model(state_tensor)
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
        
        # Choose best move
        best_index = max(legal_indices, key=lambda i: q_values[i])
        return index_to_move[best_index]
    
    def _choose_opponent_move(self, board, episode):
        """Choose opponent move with adaptive difficulty"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        difficulty = self.adaptive_training.get_opponent_strength()
        
        # Random move probability decreases with difficulty
        random_prob = max(0.1, 1.0 - (difficulty * 0.4))
        
        if random.random() < random_prob:
            return random.choice(legal_moves)
        
        # Use simple evaluation for stronger opponent
        best_move = legal_moves[0]
        best_score = -float('inf')
        
        for move in legal_moves[:min(20, len(legal_moves))]:  # Limit search
            board_copy = board.copy()
            board_copy.push(move)
            
            # Simple evaluation
            score = self.evaluator.evaluate_position(board_copy)
            if board.turn == chess.BLACK:  # From opponent's perspective
                score = -score
            
            # Add some randomness
            score += random.uniform(-0.1, 0.1)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def _calculate_reward(self, board, move):
        """Calculate reward for a move"""
        # Use our enhanced evaluator
        board_copy = board.copy()
        board_copy.push(move)
        
        reward = self.evaluator.evaluate_position(board_copy, move)
        
        # Additional rewards
        if board_copy.is_checkmate():
            reward += 5.0 if board_copy.turn == chess.BLACK else -5.0
        elif board_copy.is_check():
            reward += 0.1
        
        # Reward captures
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                reward += self.evaluator.PIECE_VALUES[captured_piece.piece_type] / 1000.0
        
        return np.clip(reward, -5.0, 5.0)
    
    def _learn_from_episode(self):
        """Learn from experiences in replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch_data = self.replay_buffer.sample(self.batch_size)
        if batch_data is None:
            return
        
        experiences, indices, weights = batch_data
        
        # Prepare batch
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for exp in experiences:
            states.append(exp[0])
            actions.append(exp[1])
            rewards.append(exp[2])
            next_states.append(exp[3])
            dones.append(exp[4])
        
        # Convert to tensors
        states = torch.tensor(np.array(states)).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        # Current Q-values
        current_q_values, _ = self.model(states)
        current_q_values = current_q_values.gather(1, actions)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values, _ = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)
        
        # Calculate loss with importance sampling
        td_errors = current_q_values - target_q_values
        weighted_loss = (weights * td_errors.pow(2).squeeze()).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update priorities
        priorities = td_errors.abs().cpu().data.numpy().flatten()
        self.replay_buffer.update_priorities(indices, priorities)
    
    def enable_human_learning(self, human_games_log="data/human_games.csv"):
        """Enable learning from human games"""
        self.human_games_log = human_games_log
        print("🤝 Human learning enabled! Games will be logged for continuous improvement.")
    
    def learn_from_human_game(self, moves, result, human_rating=1500):
        """Learn from a game against a human"""
        self.continuous_learner.learn_from_human_game(moves, result, human_rating)
        
        # Log the game
        if hasattr(self, 'human_games_log'):
            self._log_human_game(moves, result, human_rating)
    
    def _log_human_game(self, moves, result, human_rating):
        """Log human games for analysis"""
        os.makedirs(os.path.dirname(self.human_games_log), exist_ok=True)
        
        with open(self.human_games_log, 'a', newline='') as f:
            writer = csv.writer(f)
            if os.path.getsize(self.human_games_log) == 0:  # Empty file
                writer.writerow(['moves', 'result', 'human_rating', 'timestamp'])
            
            moves_str = ' '.join([str(move) for move in moves])
            writer.writerow([moves_str, result, human_rating, time.time()])
    
    def save_checkpoint(self, episode):
        """Save training checkpoint"""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        
        torch.save({
            'episode': episode,
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'replay_buffer': self.replay_buffer,
            'adaptive_training': self.adaptive_training
        }, self.checkpoint_path)
        
        print(f"💾 Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self):
        """Load training checkpoint"""
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', 0.1)
                
                print(f"✅ Loaded checkpoint from episode {checkpoint.get('episode', 0)}")
                return checkpoint.get('episode', 0)
            except Exception as e:
                print(f"⚠️ Could not load checkpoint: {e}")
        
        return 0
    
    def load_episode_number(self):
        """Get starting episode number"""
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                return checkpoint.get('episode', 0)
            except:
                pass
        return 0
    
    def save_best_model(self):
        """Save the best performing model"""
        best_path = "data/best_enhanced_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon
        }, best_path)
    
    def _log_progress(self, episode, reward, win_rate, rating):
        """Log training progress"""
        os.makedirs(os.path.dirname(self.reward_log_path), exist_ok=True)
        
        write_header = not os.path.exists(self.reward_log_path)
        with open(self.reward_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['episode', 'reward', 'epsilon', 'win_rate', 'estimated_rating'])
            writer.writerow([episode, reward, self.epsilon, win_rate, rating])
    
    def _print_final_summary(self):
        """Print final training summary"""
        print("\n" + "="*60)
        print("🎉 ENHANCED TRAINING COMPLETED!")
        print("="*60)
        
        # Print game analysis
        self.game_analyzer.print_analysis_summary()
        
        # Print improvement suggestions
        suggestions = self.model_evaluator.get_improvement_suggestions()
        print("\n💡 Improvement Suggestions:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")
        
        print(f"\n📁 Model saved to: {self.checkpoint_path}")
        print(f"📊 Training logs: {self.reward_log_path}")
        print("\n🚀 Your enhanced chess AI is ready!")

def main():
    """Main training function"""
    print("🎯 Enhanced Chess DQN Training")
    print("="*50)
    
    # Initialize trainer
    trainer = EnhancedChessTrainer()
    
    # Enable human learning
    trainer.enable_human_learning()
    
    # Start training
    try:
        trainer.train(episodes=3000, save_freq=50, eval_freq=200)
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        trainer.save_checkpoint(trainer.load_episode_number())
        print("💾 Progress saved!")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        trainer.save_checkpoint(trainer.load_episode_number())
        print("💾 Progress saved before exit!")

if __name__ == "__main__":
    main()