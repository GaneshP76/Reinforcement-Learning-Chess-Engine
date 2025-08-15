# Create this file: training/chess_database_trainer.py

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import requests
import os
import gzip
import io
from tqdm import tqdm
from collections import deque
import random
import zipfile
from datetime import datetime

class ChessDataDownloader:
    """Download high-quality chess games from various sources"""
    
    def __init__(self, data_dir="data/chess_databases"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_lichess_database(self, year=2023, month=12, rating_threshold=2400):
        """
        Download Lichess database of high-rated games
        These are games from players rated 2400+ (master level)
        """
        print(f"📥 Downloading Lichess database for {year}-{month:02d}...")
        
        # Lichess provides monthly databases
        filename = f"lichess_db_standard_rated_{year}-{month:02d}.pgn.bz2"
        url = f"https://database.lichess.org/standard/{filename}"
        
        local_path = os.path.join(self.data_dir, filename)
        
        if os.path.exists(local_path):
            print(f"✅ Database already exists: {local_path}")
            return local_path
        
        try:
            print(f"🌐 Downloading from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f, tqdm(
                desc="Downloading", 
                total=total_size, 
                unit='B', 
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Downloaded: {local_path}")
            return local_path
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("💡 Try downloading manually from: https://database.lichess.org/")
            return None
    
    def download_fics_games(self):
        """Download FICS (Free Internet Chess Server) games"""
        print("📥 Downloading FICS games database...")
        
        # FICS games are available from various sources
        urls = [
            "https://www.ficsgames.org/download/ficsgames-2023.pgn.bz2",
            "https://www.ficsgames.org/download/ficsgames-2022.pgn.bz2"
        ]
        
        downloaded_files = []
        for url in urls:
            try:
                filename = url.split('/')[-1]
                local_path = os.path.join(self.data_dir, filename)
                
                if not os.path.exists(local_path):
                    print(f"🌐 Downloading {filename}...")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                
                downloaded_files.append(local_path)
                print(f"✅ Downloaded: {local_path}")
                
            except Exception as e:
                print(f"⚠️ Failed to download {url}: {e}")
        
        return downloaded_files
    
    def download_chess_com_games(self):
        """Download Chess.com games (if available)"""
        print("📥 Downloading Chess.com master games...")
        
        # Chess.com provides some databases
        # You might need to manually download from chess.com/games/archive
        print("💡 For Chess.com games, visit: https://www.chess.com/games/archive")
        print("💡 Download master-level games and place them in:", self.data_dir)

class ChessGameProcessor:
    """Process PGN files and extract training data"""
    
    def __init__(self, min_rating=2200, max_games_per_file=50000):
        self.min_rating = min_rating  # Only use games from strong players
        self.max_games_per_file = max_games_per_file
        self.processed_positions = []
        
    def process_pgn_file(self, pgn_path, output_prefix="training_data"):
        """
        Process a PGN file and extract positions with outcomes
        Returns training data suitable for the DQN model
        """
        print(f"🔄 Processing {pgn_path}...")
        
        # Handle compressed files
        if pgn_path.endswith('.bz2'):
            import bz2
            pgn_file = bz2.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
        elif pgn_path.endswith('.gz'):
            import gzip
            pgn_file = gzip.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            pgn_file = open(pgn_path, 'r', encoding='utf-8', errors='ignore')
        
        training_positions = []
        game_count = 0
        processed_count = 0
        
        try:
            while game_count < self.max_games_per_file:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                game_count += 1
                
                # Filter by rating (only use high-rated games)
                white_rating = self._get_rating(game, 'White')
                black_rating = self._get_rating(game, 'Black')
                
                if (white_rating >= self.min_rating and black_rating >= self.min_rating):
                    positions = self._extract_positions_from_game(game)
                    training_positions.extend(positions)
                    processed_count += 1
                
                if game_count % 10000 == 0:
                    print(f"   📊 Processed {game_count} games, kept {processed_count} high-rated games")
        
        finally:
            pgn_file.close()
        
        print(f"✅ Extracted {len(training_positions)} positions from {processed_count} games")
        
        # Save processed data
        if training_positions:
            self._save_training_data(training_positions, f"{output_prefix}_{datetime.now().strftime('%Y%m%d')}")
        
        return training_positions
    
    def _get_rating(self, game, color):
        """Extract player rating from game headers"""
        try:
            rating_str = game.headers.get(f'{color}Elo', '0')
            return int(rating_str) if rating_str.isdigit() else 0
        except:
            return 0
    
    def _extract_positions_from_game(self, game):
        """Extract positions and their evaluations from a game"""
        positions = []
        board = game.board()
        
        # Get game result
        result = game.headers.get('Result', '*')
        if result == '*':  # Unfinished game
            return positions
        
        # Convert result to numerical value
        if result == '1-0':
            white_score = 1.0
            black_score = -1.0
        elif result == '0-1':
            white_score = -1.0
            black_score = 1.0
        else:  # Draw
            white_score = 0.0
            black_score = 0.0
        
        move_number = 0
        for move in game.mainline_moves():
            # Store position before the move
            current_score = white_score if board.turn == chess.WHITE else black_score
            
            # Weight later moves more heavily (they're more decisive)
            move_weight = min(1.0, 0.3 + (move_number / 100.0))
            final_score = current_score * move_weight
            
            position_data = {
                'board': board.copy(),
                'move': move,
                'score': final_score,
                'move_number': move_number,
                'game_result': result
            }
            positions.append(position_data)
            
            board.push(move)
            move_number += 1
        
        return positions
    
    def _save_training_data(self, positions, filename):
        """Save training data for later use"""
        import pickle
        
        output_path = os.path.join("data", f"{filename}.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(positions, f)
        
        print(f"💾 Saved training data: {output_path}")

class MasterGamesTrainer:
    """Train the DQN model on master-level games"""
    
    def __init__(self, agent, learning_rate=0.0001):
        self.agent = agent
        self.device = agent.device
        self.model = agent.model
        
        # Use lower learning rate for fine-tuning
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        print(f"🎓 Master Games Trainer initialized")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Device: {self.device}")
    
    def train_on_master_games(self, training_positions, epochs=5, batch_size=32):
        """
        Train the model on positions from master games
        This is supervised learning using game outcomes
        """
        print(f"🎓 Training on {len(training_positions)} master positions...")
        
        # Prepare training data
        states, targets, move_indices = self._prepare_training_data(training_positions)
        
        if len(states) == 0:
            print("❌ No valid training data prepared!")
            return
        
        # Convert to tensors
        states_tensor = torch.tensor(np.array(states)).to(self.device)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).to(self.device)
        move_indices_tensor = torch.tensor(move_indices).to(self.device)
        
        # Training loop
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle data
            indices = torch.randperm(len(states_tensor))
            
            for i in tqdm(range(0, len(indices), batch_size), desc=f"Epoch {epoch+1}/{epochs}"):
                batch_indices = indices[i:i+batch_size]
                
                batch_states = states_tensor[batch_indices]
                batch_targets = targets_tensor[batch_indices]
                batch_moves = move_indices_tensor[batch_indices]
                
                # Forward pass
                q_values, position_values = self.model(batch_states)
                
                # Calculate loss using the master game outcomes
                # We want the Q-value of the chosen move to match the game outcome
                chosen_q_values = q_values.gather(1, batch_moves.unsqueeze(1)).squeeze(1)
                loss = self.criterion(chosen_q_values, batch_targets)
                
                # Also train the position evaluation head
                position_loss = self.criterion(position_values.squeeze(1), batch_targets)
                total_loss_batch = loss + 0.1 * position_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss_batch.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            total_loss += avg_epoch_loss
            
            print(f"   📈 Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")
        
        avg_total_loss = total_loss / epochs
        print(f"✅ Master games training completed!")
        print(f"   📊 Final average loss: {avg_total_loss:.4f}")
        
        self.model.eval()
        return avg_total_loss
    
    def _prepare_training_data(self, positions):
        """Convert game positions to training tensors"""
        from utils.utils import board_to_tensor
        from utils import move_encoder
        
        states = []
        targets = []
        move_indices = []
        
        print("🔄 Preparing training data from master games...")
        
        for pos_data in tqdm(positions[:100000], desc="Processing positions"):  # Limit to 100k positions
            try:
                board = pos_data['board']
                move = pos_data['move']
                score = pos_data['score']
                
                # Convert board to tensor
                state = board_to_tensor(board)
                
                # Get move index
                move_idx = move_encoder.move_to_index(move)
                
                if move_idx is not None:
                    states.append(state)
                    targets.append(score)
                    move_indices.append(move_idx)
                    
            except Exception as e:
                continue  # Skip problematic positions
        
        print(f"✅ Prepared {len(states)} valid training examples")
        return states, targets, move_indices

def main():
    """Main function to download and train on chess databases"""
    print("🏆 CHESS MASTER DATABASE TRAINER")
    print("=" * 60)
    print("This will download games from 2200+ rated players")
    print("and train your AI to play like a master!")
    print()
    
    # Step 1: Download databases
    downloader = ChessDataDownloader()
    
    print("📥 STEP 1: Download master games")
    print("Choose data source:")
    print("1. 🏆 Lichess Database (Recommended - Free)")
    print("2. 🎯 FICS Games (Alternative)")
    print("3. 📂 Use local PGN file")
    print("4. 🚀 Download All Sources")
    
    choice = input("Enter choice (1-4): ").strip()
    
    pgn_files = []
    
    if choice == "1":
        # Download recent Lichess database
        lichess_file = downloader.download_lichess_database(2023, 12)
        if lichess_file:
            pgn_files.append(lichess_file)
    
    elif choice == "2":
        # Download FICS games
        fics_files = downloader.download_fics_games()
        pgn_files.extend(fics_files)
    
    elif choice == "3":
        # Use local file
        pgn_path = input("Enter path to your PGN file: ").strip()
        if os.path.exists(pgn_path):
            pgn_files.append(pgn_path)
        else:
            print("❌ File not found!")
            return
    
    elif choice == "4":
        # Download everything
        lichess_file = downloader.download_lichess_database(2023, 12)
        if lichess_file:
            pgn_files.append(lichess_file)
        
        fics_files = downloader.download_fics_games()
        pgn_files.extend(fics_files)
    
    if not pgn_files:
        print("❌ No PGN files available for training!")
        return
    
    # Step 2: Process games
    print("\n🔄 STEP 2: Process master games")
    processor = ChessGameProcessor(min_rating=2200)
    
    all_positions = []
    for pgn_file in pgn_files:
        if os.path.exists(pgn_file):
            positions = processor.process_pgn_file(pgn_file)
            all_positions.extend(positions)
        else:
            print(f"⚠️ File not found: {pgn_file}")
    
    if not all_positions:
        print("❌ No positions extracted from games!")
        return
    
    # Step 3: Load your trained model
    print("\n🤖 STEP 3: Load your existing model")
    
    # Import your agent
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.enhanced_dqn_agent import EnhancedDQNAgent
    
    # Load your existing model
    checkpoint_paths = [
        "data/enhanced_dqn_checkpoint.pth",
        "data/best_enhanced_model.pth",
        "data/dqn_checkpoint.pth"
    ]
    
    model_loaded = False
    for path in checkpoint_paths:
        if os.path.exists(path):
            print(f"✅ Loading model from: {path}")
            agent = EnhancedDQNAgent(path)
            model_loaded = True
            break
    
    if not model_loaded:
        print("❌ No trained model found!")
        print("💡 Train your model first with basic episodes, then run this script")
        return
    
    # Step 4: Fine-tune on master games
    print(f"\n🎓 STEP 4: Fine-tune on {len(all_positions)} master positions")
    
    trainer = MasterGamesTrainer(agent, learning_rate=0.0001)  # Lower learning rate for fine-tuning
    
    # Train in smaller chunks to avoid memory issues
    chunk_size = 50000
    total_chunks = (len(all_positions) + chunk_size - 1) // chunk_size
    
    print(f"🔄 Training in {total_chunks} chunks of {chunk_size} positions each")
    
    for chunk_num in range(total_chunks):
        start_idx = chunk_num * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_positions))
        chunk_positions = all_positions[start_idx:end_idx]
        
        print(f"\n📊 Training chunk {chunk_num + 1}/{total_chunks}")
        trainer.train_on_master_games(chunk_positions, epochs=3, batch_size=32)
        
        # Save progress after each chunk
        agent.save_model()
    
    print("\n🎉 MASTER GAMES TRAINING COMPLETED!")
    print("=" * 60)
    print("Your AI has now learned from master-level games!")
    print("Expected improvements:")
    print("  • Better opening knowledge")
    print("  • Improved tactical patterns")
    print("  • Stronger endgame play")
    print("  • More human-like positional understanding")
    print()
    print("🎮 Test your improved AI: python gui/gui_app.py")
    print("📈 The AI should now play significantly stronger!")

if __name__ == "__main__":
    main()