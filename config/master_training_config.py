# Create this file: config/master_training_config.py

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

# Easy presets for different training intensities
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