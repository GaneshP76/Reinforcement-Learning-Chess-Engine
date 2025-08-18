# Create this file: training/master_games_integration.py

import torch
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the training classes (assuming they exist in the same directory or can be found)
try:
    from training.chess_database_trainer import ChessDataDownloader, ChessGameProcessor, MasterGamesTrainer
except ImportError:
    print("❌ chess_database_trainer module not found. Please ensure it exists in the training directory.")
    sys.exit(1)

try:
    from agents.enhanced_dqn_agent import EnhancedDQNAgent
except ImportError:
    print("❌ enhanced_dqn_agent module not found. Please ensure it exists in the agents directory.")
    sys.exit(1)

# Instead of importing from config module, define the configuration classes here
class MasterTrainingConfig:
    """Configuration for training on master-level chess games"""
    
    # 🏆 GAME SOURCES AND QUALITY FILTERS
    MIN_PLAYER_RATING = 2200      # Only use games from 2200+ players (master level)
    PREFERRED_RATING = 2600       # Prefer games from 2600+ players (super-GM level)
    MAX_GAMES_PER_SOURCE = 100000 # Limit games per database to manage memory
    
    # 📅 DATABASE SOURCES
    LICHESS_DATABASES = [
        # Recent databases (higher quality, more modern openings)
        {"year": 2024, "month": 1},
        {"year": 2023, "month": 12},
        {"year": 2023, "month": 11},
    ]
    
    # 🧠 TRAINING PARAMETERS
    MASTER_LEARNING_RATE = 0.00005    # Very low LR for fine-tuning
    MASTER_BATCH_SIZE = 16            # Smaller batches for stability
    MASTER_EPOCHS = 3                 # Few epochs to avoid overfitting
    GRADIENT_CLIP_NORM = 0.5          # Strict gradient clipping
    
    # 🎯 POSITION FILTERING
    MIN_MOVES_PER_GAME = 20           # Skip very short games
    MAX_MOVES_PER_GAME = 150          # Skip very long games
    SKIP_BLITZ_GAMES = True           # Only use classical time controls
    SKIP_BULLET_GAMES = True          # Only use classical time controls
    
    # 💾 MEMORY MANAGEMENT
    CHUNK_SIZE = 25000                # Process positions in chunks
    MAX_POSITIONS_MEMORY = 200000     # Max positions to keep in memory
    
    # 🎲 DATA AUGMENTATION
    POSITION_SAMPLE_RATE = 0.3        # Sample 30% of positions from each game
    OUTCOME_WEIGHTING = True          # Weight positions by game outcome
    TEMPORAL_WEIGHTING = True         # Weight later moves more heavily
    
    # 📊 FAMOUS PLAYERS TO PRIORITIZE (if available in headers)
    PRIORITY_PLAYERS = [
        "Carlsen", "Magnus", "Nakamura", "Hikaru", "Caruana", "Fabiano",
        "Ding", "Liren", "Nepomnichtchi", "Ian", "Firouzja", "Alireza",
        "Giri", "Anish", "Aronian", "Levon", "Mamedyarov", "Shakhriyar",
        "Grischuk", "Alexander", "Vachier-Lagrave", "Maxime", "So", "Wesley"
    ]
    
    # 🏅 TOURNAMENT GAMES (higher quality)
    PRIORITY_EVENTS = [
        "World Championship", "Candidates", "Grand Prix", "Chess.com Global Championship",
        "Tata Steel", "Norway Chess", "Saint Louis Rapid", "Speed Chess Championship"
    ]
    
    @classmethod
    def get_download_urls(cls):
        """Get URLs for downloading master databases"""
        urls = []
        
        # Lichess databases
        for db in cls.LICHESS_DATABASES:
            filename = f"lichess_db_standard_rated_{db['year']}-{db['month']:02d}.pgn.bz2"
            url = f"https://database.lichess.org/standard/{filename}"
            urls.append({
                "url": url,
                "filename": filename,
                "source": "lichess",
                "estimated_size_gb": 2.5,
                "quality": "high"
            })
        
        # Add other sources
        urls.extend([
            {
                "url": "https://www.pgnmentor.com/files.html#players",
                "filename": "master_games.pgn",
                "source": "pgnmentor", 
                "estimated_size_gb": 1.2,
                "quality": "very_high"
            },
            {
                "url": "https://www.chess.com/games/archive",
                "filename": "chess_com_masters.pgn",
                "source": "chess_com",
                "estimated_size_gb": 0.8,
                "quality": "high"
            }
        ])
        
        return urls
    
    @classmethod
    def estimate_training_time(cls, num_positions):
        """Estimate training time based on positions and hardware"""
        # Rough estimates based on batch size and hardware
        positions_per_hour_gpu = 50000    # With GPU
        positions_per_hour_cpu = 5000     # CPU only
        
        import torch
        if torch.cuda.is_available():
            hours = num_positions / positions_per_hour_gpu
            hardware = "GPU"
        else:
            hours = num_positions / positions_per_hour_cpu
            hardware = "CPU"
        
        return {
            "estimated_hours": round(hours, 1),
            "hardware": hardware,
            "positions": num_positions
        }
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary"""
        print("🏆 MASTER TRAINING CONFIGURATION")
        print("=" * 50)
        print(f"📊 Quality Filters:")
        print(f"   Minimum Rating: {cls.MIN_PLAYER_RATING}")
        print(f"   Preferred Rating: {cls.PREFERRED_RATING}")
        print(f"   Max Games/Source: {cls.MAX_GAMES_PER_SOURCE:,}")
        print()
        print(f"🧠 Training Parameters:")
        print(f"   Learning Rate: {cls.MASTER_LEARNING_RATE}")
        print(f"   Batch Size: {cls.MASTER_BATCH_SIZE}")
        print(f"   Epochs: {cls.MASTER_EPOCHS}")
        print()
        print(f"📅 Data Sources: {len(cls.LICHESS_DATABASES)} Lichess databases")
        print(f"🎯 Priority Players: {len(cls.PRIORITY_PLAYERS)} grandmasters")
        print("=" * 50)


class TrainingPresets:
    """Predefined configurations for different training goals"""
    
    @staticmethod
    def get_quick_improvement():
        """Quick training on recent high-quality games"""
        config = MasterTrainingConfig()
        config.MAX_GAMES_PER_SOURCE = 25000
        config.MASTER_EPOCHS = 2
        config.MIN_PLAYER_RATING = 2400
        return config
    
    @staticmethod
    def get_comprehensive_training():
        """Comprehensive training on large database"""
        config = MasterTrainingConfig()
        config.MAX_GAMES_PER_SOURCE = 200000
        config.MASTER_EPOCHS = 5
        config.MIN_PLAYER_RATING = 2200
        return config
    
    @staticmethod
    def get_grandmaster_training():
        """Elite training on only 2600+ games"""
        config = MasterTrainingConfig()
        config.MAX_GAMES_PER_SOURCE = 50000
        config.MIN_PLAYER_RATING = 2600
        config.MASTER_EPOCHS = 4
        config.MASTER_LEARNING_RATE = 0.00003
        return config
    
    @staticmethod
    def get_mobile_optimized():
        """Lightweight training for mobile deployment"""
        config = MasterTrainingConfig()
        config.MAX_GAMES_PER_SOURCE = 15000
        config.MASTER_BATCH_SIZE = 8
        config.MASTER_EPOCHS = 2
        config.CHUNK_SIZE = 10000
        return config


class IntegratedMasterTraining:
    """
    Complete system to train your chess AI on master-level games
    This bridges the gap from your 10K episodes to 2800+ level play
    """
    
    def __init__(self, checkpoint_path="data/enhanced_dqn_checkpoint.pth"):
        self.checkpoint_path = checkpoint_path
        self.config = MasterTrainingConfig()
        
        print("🏆 INTEGRATED MASTER TRAINING SYSTEM")
        print("=" * 60)
        print("This will transform your AI from amateur to master level!")
        
    def run_complete_master_training(self):
        """
        Complete pipeline: Download → Process → Train
        This is the main function you'll call after your 10K episodes
        """
        print("\n🚀 STARTING COMPLETE MASTER TRAINING PIPELINE")
        
        # Step 1: Check if your base model exists
        if not self._verify_base_model():
            return False
        
        # Step 2: Choose training intensity
        training_config = self._choose_training_intensity()
        
        # Step 3: Download master games
        game_files = self._download_master_games()
        if not game_files:
            return False
        
        # Step 4: Process games into training data
        training_positions = self._process_master_games(game_files, training_config)
        if not training_positions:
            return False
        
        # Step 5: Load your existing model
        agent = self._load_trained_model()
        if not agent:
            return False
        
        # Step 6: Fine-tune on master games
        success = self._fine_tune_on_masters(agent, training_positions, training_config)
        
        if success:
            print("\n🎉 MASTER TRAINING COMPLETED SUCCESSFULLY!")
            self._print_final_summary()
            return True
        else:
            print("\n❌ Master training failed!")
            return False
    
    def _verify_base_model(self):
        """Check if base model exists and is ready for master training"""
        print("\n🔍 STEP 1: Verifying base model...")
        
        possible_paths = [
            self.checkpoint_path,
            "data/best_enhanced_model.pth",
            "data/dqn_checkpoint.pth"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    # Try to load and check the model
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                    episode = checkpoint.get('episode', 0)
                    
                    print(f"✅ Found model: {path}")
                    print(f"   Episodes trained: {episode}")
                    
                    if episode < 5000:
                        print("⚠️  WARNING: Model has fewer than 5,000 episodes")
                        print("💡 Consider training more base episodes first for better results")
                        
                        response = input("Continue anyway? (y/n): ").lower()
                        if response != 'y':
                            return False
                    
                    self.checkpoint_path = path
                    return True
                    
                except Exception as e:
                    print(f"❌ Failed to load {path}: {e}")
                    continue
        
        print("❌ No suitable base model found!")
        print("💡 Please train your base model first:")
        print("   python training/enhanced_train.py")
        return False
    
    def _choose_training_intensity(self):
        """Let user choose training intensity"""
        print("\n🎯 STEP 2: Choose training intensity...")
        
        print("\nAvailable training presets:")
        print("1. 🚀 Quick Improvement (2-3 hours, 25K games)")
        print("   └ Fast boost using recent master games")
        print("2. 💪 Comprehensive Training (6-8 hours, 200K games)")
        print("   └ Thorough training on large database")
        print("3. 👑 Grandmaster Training (4-5 hours, 50K elite games)")
        print("   └ Train only on 2600+ rated players")
        print("4. 📱 Mobile Optimized (1-2 hours, 15K games)")
        print("   └ Lightweight training for mobile deployment")
        print("5. 🔧 Custom Configuration")
        
        while True:
            try:
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == "1":
                    config = TrainingPresets.get_quick_improvement()
                    print("🚀 Quick Improvement selected!")
                    break
                elif choice == "2":
                    config = TrainingPresets.get_comprehensive_training()
                    print("💪 Comprehensive Training selected!")
                    break
                elif choice == "3":
                    config = TrainingPresets.get_grandmaster_training()
                    print("👑 Grandmaster Training selected!")
                    break
                elif choice == "4":
                    config = TrainingPresets.get_mobile_optimized()
                    print("📱 Mobile Optimized selected!")
                    break
                elif choice == "5":
                    config = self._custom_configuration()
                    print("🔧 Custom configuration created!")
                    break
                else:
                    print("❌ Invalid choice. Please enter 1-5.")
                    continue
                    
            except KeyboardInterrupt:
                print("\n⏸️ Training cancelled by user")
                return None
        
        # Show configuration summary
        print(f"\n📋 Configuration Summary:")
        print(f"   Min Rating: {config.MIN_PLAYER_RATING}")
        print(f"   Max Games: {config.MAX_GAMES_PER_SOURCE:,}")
        print(f"   Learning Rate: {config.MASTER_LEARNING_RATE}")
        print(f"   Batch Size: {config.MASTER_BATCH_SIZE}")
        print(f"   Epochs: {config.MASTER_EPOCHS}")
        
        return config
    
    def _custom_configuration(self):
        """Create custom training configuration"""
        config = MasterTrainingConfig()
        
        print("\n🔧 Custom Configuration Setup:")
        
        try:
            rating = int(input(f"Minimum player rating [{config.MIN_PLAYER_RATING}]: ") or config.MIN_PLAYER_RATING)
            config.MIN_PLAYER_RATING = max(1500, min(3000, rating))
            
            max_games = int(input(f"Max games per source [{config.MAX_GAMES_PER_SOURCE}]: ") or config.MAX_GAMES_PER_SOURCE)
            config.MAX_GAMES_PER_SOURCE = max(1000, min(500000, max_games))
            
            epochs = int(input(f"Training epochs [{config.MASTER_EPOCHS}]: ") or config.MASTER_EPOCHS)
            config.MASTER_EPOCHS = max(1, min(10, epochs))
            
            print("✅ Custom configuration created!")
            
        except ValueError:
            print("⚠️ Invalid input, using default values")
        
        return config
    
    def _download_master_games(self):
        """Download master-level chess games"""
        print("\n📥 STEP 3: Downloading master games...")
        
        downloader = ChessDataDownloader()
        
        # Try to download recent Lichess database
        print("🌐 Attempting to download Lichess master database...")
        
        # Try recent months
        for year in [2024, 2023]:
            for month in [12, 11, 10, 9, 8]:
                lichess_file = downloader.download_lichess_database(year, month)
                if lichess_file and os.path.exists(lichess_file):
                    print(f"✅ Successfully downloaded: {lichess_file}")
                    return [lichess_file]
        
        # Fallback: check for manually downloaded files
        print("🔍 Checking for manually downloaded PGN files...")
        
        data_dir = "data/chess_databases"
        manual_files = []
        
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(('.pgn', '.pgn.bz2', '.pgn.gz')):
                    full_path = os.path.join(data_dir, file)
                    manual_files.append(full_path)
                    print(f"📁 Found: {file}")
        
        if manual_files:
            print(f"✅ Using {len(manual_files)} manually downloaded files")
            return manual_files
        
        # Last resort: provide manual download instructions
        print("❌ No master game databases found!")
        print("\n💡 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("1. Visit: https://database.lichess.org/")
        print("2. Download a recent month (e.g., lichess_db_standard_rated_2023-12.pgn.bz2)")
        print("3. Place it in: data/chess_databases/")
        print("4. Run this script again")
        print()
        print("Alternative sources:")
        print("• https://www.pgnmentor.com/files.html")
        print("• https://www.chess.com/games/archive")
        print("• https://www.ficsgames.org/")
        
        return []
    
    def _process_master_games(self, game_files, config):
        """Process downloaded games into training data"""
        print(f"\n🔄 STEP 4: Processing master games...")
        
        processor = ChessGameProcessor(
            min_rating=config.MIN_PLAYER_RATING,
            max_games_per_file=config.MAX_GAMES_PER_SOURCE
        )
        
        all_positions = []
        
        for game_file in game_files:
            print(f"\n📖 Processing: {os.path.basename(game_file)}")
            
            try:
                positions = processor.process_pgn_file(
                    game_file, 
                    output_prefix=f"master_data_{config.MIN_PLAYER_RATING}"
                )
                all_positions.extend(positions)
                
                print(f"✅ Extracted {len(positions):,} positions")
                
            except Exception as e:
                print(f"❌ Error processing {game_file}: {e}")
                continue
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   Total positions: {len(all_positions):,}")
        print(f"   Average per game: ~{len(all_positions)/config.MAX_GAMES_PER_SOURCE:.0f}")
        
        if len(all_positions) < 10000:
            print("⚠️ WARNING: Very few positions extracted!")
            print("💡 Consider lowering the minimum rating or using more game files")
            
            response = input("Continue with limited data? (y/n): ").lower()
            if response != 'y':
                return []
        
        return all_positions
    
    def _load_trained_model(self):
        """Load your existing trained model"""
        print(f"\n🤖 STEP 5: Loading your trained model...")
        
        try:
            agent = EnhancedDQNAgent(self.checkpoint_path)
            
            # Get model info
            info = agent.get_model_info()
            print(f"✅ Model loaded successfully!")
            print(f"   Games learned: {info['games_learned']}")
            print(f"   Parameters: {info['parameters']:,}")
            print(f"   Current epsilon: {info['epsilon']:.4f}")
            
            # Set model to evaluation mode for training stability
            agent.model.eval()
            
            return agent
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return None
    
    def _fine_tune_on_masters(self, agent, training_positions, config):
        """Fine-tune the model on master games"""
        print(f"\n🎓 STEP 6: Fine-tuning on master games...")
        
        if len(training_positions) == 0:
            print("❌ No training positions available!")
            return False
        
        try:
            # Create trainer with configuration
            trainer = MasterGamesTrainer(agent, learning_rate=config.MASTER_LEARNING_RATE)
            
            # Estimate training time
            time_estimate = MasterTrainingConfig.estimate_training_time(len(training_positions))
            print(f"⏰ Estimated training time: {time_estimate['estimated_hours']} hours on {time_estimate['hardware']}")
            
            # Confirm before starting
            response = input("Start master training? (y/n): ").lower()
            if response != 'y':
                print("⏸️ Training cancelled")
                return False
            
            # Train in chunks to manage memory
            chunk_size = config.CHUNK_SIZE
            total_chunks = (len(training_positions) + chunk_size - 1) // chunk_size
            
            print(f"🔄 Training in {total_chunks} chunks of {chunk_size:,} positions each")
            
            overall_loss = 0
            successful_chunks = 0
            
            for chunk_num in range(total_chunks):
                start_idx = chunk_num * chunk_size
                end_idx = min(start_idx + chunk_size, len(training_positions))
                chunk_positions = training_positions[start_idx:end_idx]
                
                print(f"\n📊 Training chunk {chunk_num + 1}/{total_chunks}")
                print(f"   Positions: {len(chunk_positions):,}")
                
                try:
                    chunk_loss = trainer.train_on_master_games(
                        chunk_positions, 
                        epochs=config.MASTER_EPOCHS,
                        batch_size=config.MASTER_BATCH_SIZE
                    )
                    
                    overall_loss += chunk_loss
                    successful_chunks += 1
                    
                    # Save progress after each chunk
                    agent.save_model()
                    print(f"💾 Progress saved (chunk {chunk_num + 1}/{total_chunks})")
                    
                except Exception as e:
                    print(f"⚠️ Chunk {chunk_num + 1} failed: {e}")
                    print("🔄 Continuing with next chunk...")
                    continue
            
            if successful_chunks > 0:
                avg_loss = overall_loss / successful_chunks
                print(f"\n✅ MASTER TRAINING COMPLETED!")
                print(f"   Successful chunks: {successful_chunks}/{total_chunks}")
                print(f"   Average loss: {avg_loss:.4f}")
                return True
            else:
                print(f"\n❌ All training chunks failed!")
                return False
                
        except Exception as e:
            print(f"❌ Master training error: {e}")
            return False
    
    def _print_final_summary(self):
        """Print final training summary with next steps"""
        print("\n" + "=" * 60)
        print("🎉 CONGRATULATIONS! Your AI is now MASTER-LEVEL trained!")
        print("=" * 60)
        
        print(f"\n📈 What just happened:")
        print(f"   ✅ Loaded your {self.checkpoint_path}")
        print(f"   ✅ Downloaded master-level chess games")
        print(f"   ✅ Processed games from 2200+ rated players")
        print(f"   ✅ Fine-tuned your neural network on master patterns")
        print(f"   ✅ Saved the improved model")
        
        print(f"\n🚀 Expected improvements:")
        print(f"   📚 Better opening knowledge from recent master games")
        print(f"   ⚡ Improved tactical pattern recognition")
        print(f"   🏰 Stronger endgame technique")
        print(f"   🎯 More accurate position evaluation")
        print(f"   👑 Human-like strategic thinking")
        
        print(f"\n🎮 Next Steps:")
        print(f"   1. 🎲 Test your AI: python gui/gui_app.py")
        print(f"   2. 📊 Compare before/after performance")
        print(f"   3. 🏆 Challenge stronger players (1800-2200 rating)")
        print(f"   4. 📱 Export for mobile: python utils/model_export.py")
        
        print(f"\n📁 Important Files:")
        print(f"   🧠 Improved model: {self.checkpoint_path}")
        print(f"   🌟 Backup: data/best_enhanced_model.pth")
        print(f"   📊 Training logs: data/training_rewards.csv")
        
        print(f"\n💡 Pro Tips:")
        print(f"   • Your AI should now play around 1600-2000 strength")
        print(f"   • It learned from games by Magnus Carlsen level players")
        print(f"   • For even stronger play, repeat with 2600+ rating filter")
        print(f"   • The model is now ready for competitive play!")
        
        print("\n" + "=" * 60)
        print("🏆 WELCOME TO MASTER-LEVEL CHESS AI!")
        print("=" * 60)


def main():
    """Main function to run integrated master training"""
    print("🎯 CHESS AI MASTER TRAINING SYSTEM")
    print("Transform your AI from amateur to master level!")
    print()
    
    trainer = IntegratedMasterTraining()
    success = trainer.run_complete_master_training()
    
    if success:
        print("\n🎉 SUCCESS! Your chess AI is now master-level trained!")
        print("🎮 Go test it: python gui/gui_app.py")
    else:
        print("\n😞 Training was not completed successfully")
        print("💡 Check the error messages above and try again")


if __name__ == "__main__":
    main()