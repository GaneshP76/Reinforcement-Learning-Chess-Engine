# execute with  python -m training.enhanced_train.py
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

# Import configuration and monitoring
from config import ChessDQNConfig
from utils.system_monitor import SystemMonitor

# Import our enhanced components
from agents.enhanced_dqn_agent import EnhancedDQNModel, ChessEvaluator, EnhancedDQNAgent
from training.training_components import OpeningBook, PrioritizedReplayBuffer, ContinuousLearner, AdaptiveTraining, GameAnalyzer, ModelEvaluator
from utils.utils import board_to_tensor
from utils import move_encoder

class EnhancedChessTrainer:
    """Enhanced trainer with continuous learning capabilities"""
    
    def __init__(self, config=None):
        # Use provided config or create default
        self.config = config or ChessDQNConfig()
        
        # Initialize system monitoring
        self.monitor = SystemMonitor(self.config.DATA_DIR)
        
        # Create necessary directories
        self.config.create_directories()
        
        # Print configuration summary
        self.config.print_config_summary()
        
        # Check initial system health
        print("\n🔍 Initial system check...")
        initial_health = self.monitor.print_resource_summary()
        if not initial_health['healthy']:
            print("⚠️ System resources are limited. Consider:")
            print("   • Closing other programs")
            print("   • Reducing BATCH_SIZE in config.py")
        
        input("Press Enter to continue with training, or Ctrl+C to abort...")
        
        # Setup device
        self.device = self.config.DEVICE
        print(f"🚀 Enhanced Chess DQN Trainer initialized on {self.device}")
        
        # Initialize models
        self.model = EnhancedDQNModel().to(self.device)
        self.target_model = EnhancedDQNModel().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Initialize components
        self.evaluator = ChessEvaluator()
        self.opening_book = OpeningBook()
        self.replay_buffer = PrioritizedReplayBuffer(capacity=self.config.MAX_REPLAY_BUFFER)
        self.adaptive_training = AdaptiveTraining()
        self.game_analyzer = GameAnalyzer()
        self.model_evaluator = ModelEvaluator()
        
        # Training parameters from config
        self.learning_rate = self.config.LEARNING_RATE
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Initialize continuous learner
        self.continuous_learner = ContinuousLearner(self.model, self.optimizer, self.device)
        
        # Training hyperparameters from config
        self.gamma = self.config.GAMMA
        self.epsilon = self.config.EPSILON_START
        self.epsilon_decay = self.config.EPSILON_DECAY
        self.epsilon_min = self.config.EPSILON_END
        self.batch_size = self.config.BATCH_SIZE
        self.target_update_freq = self.config.TARGET_UPDATE_FREQ
        
        # Paths from config
        self.checkpoint_path = self.config.CHECKPOINT_PATH
        self.reward_log_path = self.config.LOG_PATH
        
        # Load existing progress
        self.load_checkpoint()
        self.opening_book.load_openings()
        
        print("✅ All components initialized successfully!")
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, episodes=None, save_freq=None, eval_freq=None):
        """Main training loop with all enhancements"""
        # Use config values if not specified
        episodes = episodes or self.config.EPISODES
        save_freq = save_freq or self.config.SAVE_FREQUENCY
        eval_freq = eval_freq or self.config.EVAL_FREQUENCY
        
        print(f"\n🎯 Starting enhanced training for {episodes} episodes")
        
        start_episode = self.load_episode_number()
        best_win_rate = 0.0
        
        # Estimate training time
        time_estimate = self.config.estimate_training_time()
        print(f"⏰ Estimated completion: {time_estimate['total_hours']} hours")
        print("💡 You can stop training anytime with Ctrl+C - progress will be saved!")
        print("🎮 Open another terminal and run 'python gui/gui_app.py' to test your AI!")
        
        try:
            # Training loop with progress bar
            for episode in tqdm(range(start_episode, episodes), desc="🧠 Training Chess AI"):
                # System health check every 100 episodes
                if episode % 100 == 0 and episode > 0:
                    health = self.monitor.check_system_health()
                    if not health['healthy']:
                        print(f"\n⚠️ System resources getting low at episode {episode}!")
                        self.monitor.print_resource_summary()
                        
                        # Auto-cleanup if needed
                        if health['free_space_gb'] < 3:
                            print("🧹 Auto-cleaning old files...")
                            self.monitor.cleanup_old_files(keep_latest=3)
                        
                        response = input("Continue training? (y/n/s=show resources): ").lower()
                        if response == 's':
                            self.monitor.print_resource_summary()
                            response = input("Continue? (y/n): ").lower()
                        if response != 'y':
                            print("💾 Saving progress and exiting...")
                            self.save_checkpoint(episode)
                            return
                
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
                if episode % save_freq == 0 and episode > 0:
                    self.save_checkpoint(episode)
                    self.opening_book.save_openings()
                    
                    # Show system status periodically
                    if episode % (save_freq * 2) == 0:  # Every other save
                        print(f"\n📊 Episode {episode} Status:")
                        self.monitor.print_resource_summary()
                        print()
                
                # Evaluate model
                if episode % eval_freq == 0 and episode > 0:
                    print(f"\n🎯 Evaluating model at episode {episode}...")
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
                    print()  # Add spacing
                
                # Update exploration
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                # Print periodic updates
                if episode % self.config.PRINT_FREQUENCY == 0 and episode > 0:
                    focus = self.game_analyzer.get_training_focus()
                    storage_gb = self.monitor.get_training_storage_growth()
                    print(f"\n📈 Episode {episode}: Reward={episode_reward:.2f}, ε={self.epsilon:.3f}, Focus={focus}, Storage=+{storage_gb:.1f}GB")
            
            print("🎉 Training completed successfully!")
            
        except KeyboardInterrupt:
            print(f"\n⏸️ Training interrupted at episode {episode}")
            print("💾 Saving progress...")
            self.save_checkpoint(episode)
            self.opening_book.save_openings()
            print("✅ Progress saved! Run the same command to resume.")
            return
            
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            print("💾 Saving emergency checkpoint...")
            current_episode = episode if 'episode' in locals() else start_episode
            self.save_checkpoint(current_episode)
            import traceback
            traceback.print_exc()
            return
        
        self._print_final_summary(best_win_rate)
    
    def _play_training_episode(self, episode):
        """Play one training episode with all enhancements"""
        board = chess.Board()
        total_reward = 0.0
        moves_played = []
        episode_experiences = []
        
        move_count = 0
        max_moves = self.config.MAX_GAME_LENGTH
        
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
            'adaptive_training': self.adaptive_training,
            'game_analyzer': self.game_analyzer,
            'training_config': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'gamma': self.gamma
            }
        }, self.checkpoint_path)
        
        # Create backup every 500 episodes
        if episode % 500 == 0:
            backup_path = f"data/backups/backup_{episode}.pth"
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            torch.save({
                'episode': episode,
                'model_state_dict': self.model.state_dict(),
                'epsilon': self.epsilon
            }, backup_path)
            print(f"💾 Backup created: {backup_path}")
        
        print(f"💾 Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self):
        """Load training checkpoint"""
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epsilon = checkpoint.get('epsilon', 0.1)
                
                # Load additional components if available
                if 'replay_buffer' in checkpoint:
                    self.replay_buffer = checkpoint['replay_buffer']
                if 'adaptive_training' in checkpoint:
                    self.adaptive_training = checkpoint['adaptive_training']
                if 'game_analyzer' in checkpoint:
                    self.game_analyzer = checkpoint['game_analyzer']
                
                episode = checkpoint.get('episode', 0)
                print(f"✅ Loaded checkpoint from episode {episode}")
                return episode
            except Exception as e:
                print(f"⚠️ Could not load checkpoint: {e}")
                print("🆕 Starting fresh training...")
        else:
            print("🆕 No existing checkpoint found. Starting fresh.")
        
        return 0
    
    def load_episode_number(self):
        """Get starting episode number"""
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                return checkpoint.get('episode', 0)
            except:
                pass
        return 0
    
    def save_best_model(self):
        """Save the best performing model"""
        best_path = "data/best_enhanced_model.pth"
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon,
            'timestamp': time.time()
        }, best_path)
        print(f"🌟 Best model saved to {best_path}")
    
    def _log_progress(self, episode, reward, win_rate, rating):
        """Log training progress"""
        os.makedirs(os.path.dirname(self.reward_log_path), exist_ok=True)
        
        write_header = not os.path.exists(self.reward_log_path)
        with open(self.reward_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['episode', 'reward', 'epsilon', 'win_rate', 'estimated_rating', 'timestamp'])
            writer.writerow([episode, reward, self.epsilon, win_rate, rating, time.time()])
        
        print(f"📈 Episode {episode}: Win Rate={win_rate:.1%}, Rating≈{rating}, ε={self.epsilon:.3f}")
    
    def _print_final_summary(self, best_win_rate):
        """Print final training summary"""
        print("\n" + "="*60)
        print("🎉 ENHANCED TRAINING COMPLETED!")
        print("="*60)
        
        # Show final system state
        print("\n📊 Final System State:")
        final_health = self.monitor.print_resource_summary()
        
        # Show training growth
        storage_growth = self.monitor.get_training_storage_growth()
        print(f"\n📈 Training Statistics:")
        print(f"   💾 Total storage used: {storage_growth:.1f}GB")
        print(f"   🏆 Best win rate achieved: {best_win_rate:.1%}")
        
        # Estimate final rating
        if best_win_rate > 0:
            final_rating = self.model_evaluator.estimate_rating(best_win_rate)
            print(f"   🎯 Estimated final rating: ~{final_rating}")
        
        # Print game analysis
        print(f"\n🔍 Game Analysis:")
        self.game_analyzer.print_analysis_summary()
        
        # Print improvement suggestions
        suggestions = self.model_evaluator.get_improvement_suggestions()
        print("\n💡 Next Steps:")
        for suggestion in suggestions:
            print(f"   • {suggestion}")
        
        print(f"\n📁 Files created:")
        print(f"   🧠 Main model: {self.checkpoint_path}")
        print(f"   🌟 Best model: data/best_enhanced_model.pth")
        print(f"   📊 Training logs: {self.reward_log_path}")
        print(f"   📚 Opening book: data/learned_openings.pkl")
        
        print(f"\n🚀 Your enhanced chess AI is ready!")
        print(f"🎮 Play against it: python gui/gui_app.py")
        print(f"📈 View training graphs: python evaluate/plot_rewards.py")

def main():
    """Main training function with noob-friendly interface"""
    print("🎯 Enhanced Chess DQN Training")
    print("="*50)
    print("This will train a neural network to play chess!")
    print("Your RTX 2060 should handle this perfectly. 🚀")
    
    # Let user choose training intensity
    print("\nChoose training intensity:")
    print("1. 🧪 Test Run (200 episodes, ~30 min) - RECOMMENDED FIRST!")
    print("2. 🎯 Quick Training (1000 episodes, ~1.5 hours)")
    print("3. 🚀 Standard Training (3000 episodes, ~4 hours)")
    print("4. 💪 Intensive Training (5000 episodes, ~6 hours)")
    print("5. 🎓 Marathon Training (10000 episodes, ~12 hours)")
    print("6. 🔧 Custom (you choose)")
    
    try:
        choice = input("\nEnter your choice (1-6): ").strip()
        
        # Create config based on choice
        config = ChessDQNConfig()
        
        if choice == "1":
            config.EPISODES = 200
            print("🧪 Test run selected! Perfect for first-time setup.")
        elif choice == "2":
            config.EPISODES = 1000
            print("🎯 Quick training selected!")
        elif choice == "3":
            config.EPISODES = 3000
            print("🚀 Standard training selected!")
        elif choice == "4":
            config.EPISODES = 5000
            print("💪 Intensive training selected!")
        elif choice == "5":
            config.EPISODES = 10000
            print("🎓 Marathon training selected! This will take a while...")
        elif choice == "6":
            episodes = int(input("Enter number of episodes: "))
            config.EPISODES = episodes
            print(f"🔧 Custom training: {episodes} episodes")
        else:
            print("Invalid choice, using test run (200 episodes)")
            config.EPISODES = 200
        
    except (ValueError, KeyboardInterrupt):
        print("Using default: test run (200 episodes)")
        config = ChessDQNConfig()
        config.EPISODES = 200
    
    print(f"\n🎯 Training will run for {config.EPISODES} episodes")
    print("💡 Tips:")
    print("   • You can stop anytime with Ctrl+C (progress is saved!)")
    print("   • Open another terminal and run 'python gui/gui_app.py' to test your AI")
    print("   • Check the 'data/' folder for checkpoints and logs")
    
    # Initialize trainer
    trainer = EnhancedChessTrainer(config)
    
    # Enable human learning
    trainer.enable_human_learning()
    
    # Start training
    try:
        trainer.train()
        print("\n✅ Training completed successfully!")
        print("🎮 Now run: python gui/gui_app.py")
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        print("✅ Progress has been saved! Run the same command to resume.")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print("💾 Emergency save completed!")
        print("🔧 Try reducing BATCH_SIZE in config.py if you got memory errors")

if __name__ == "__main__":
    main()