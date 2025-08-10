# config.py - Central configuration for your Chess DQN
"""
Enhanced Chess DQN Configuration
Centralized settings for easy modification
"""

import torch
import os

class ChessDQNConfig:
    """Central configuration for all training parameters"""
    
    # ===============================
    # TRAINING EPISODES
    # ===============================
    EPISODES = 5000  # 🔄 MODIFY THIS FOR TRAINING LENGTH
    QUICK_TEST_EPISODES = 200  # For testing setup
    EVALUATION_GAMES = 100  # For performance evaluation
    
    # ===============================
    # HARDWARE OPTIMIZATION
    # ===============================
    # Your RTX 2060 settings (optimized)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 64  # Optimal for 6GB VRAM
    MAX_REPLAY_BUFFER = 30000  # ~1GB memory usage
    NUM_WORKERS = 4  # For your 6-core CPU (with hyperthreading)
    
    # ===============================
    # CHECKPOINT SETTINGS  
    # ===============================
    SAVE_FREQUENCY = 50  # Save every N episodes
    EVAL_FREQUENCY = 200  # Evaluate every N episodes
    BACKUP_FREQUENCY = 500  # Create backup every N episodes
    AUTO_RESUME = True  # Resume from checkpoint automatically
    
    # ===============================
    # STORAGE OPTIMIZATION
    # ===============================
    # Paths
    DATA_DIR = "data"
    CHECKPOINT_PATH = f"{DATA_DIR}/enhanced_dqn_checkpoint.pth"
    BACKUP_DIR = f"{DATA_DIR}/backups"
    LOG_PATH = f"{DATA_DIR}/training_log.csv"
    
    # Storage limits (to prevent disk overflow)
    MAX_CHECKPOINTS = 5  # Keep only 5 recent checkpoints
    MAX_BACKUPS = 3  # Keep only 3 backup sets
    LOG_ROTATION_SIZE = 10  # MB before rotating logs
    
    # ===============================
    # TRAINING HYPERPARAMETERS
    # ===============================
    LEARNING_RATE = 1e-4
    GAMMA = 0.99  # Discount factor
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    TARGET_UPDATE_FREQ = 10
    
    # ===============================
    # MODEL ARCHITECTURE
    # ===============================
    CONV_FILTERS = [128, 256]  # Optimized for RTX 2060
    RESIDUAL_BLOCKS = 4
    ATTENTION_HEADS = 8
    DROPOUT_RATE = 0.3
    
    # ===============================
    # PERFORMANCE MONITORING
    # ===============================
    PRINT_FREQUENCY = 25  # Print progress every N episodes
    DETAILED_LOGGING = True
    PLOT_GRAPHS = True  # Auto-generate training graphs
    
    # ===============================
    # SAFETY FEATURES
    # ===============================
    MAX_GAME_LENGTH = 200  # Prevent infinite games
    MEMORY_CHECK_FREQ = 100  # Check memory usage every N episodes
    AUTO_CLEANUP = True  # Clean old files automatically
    
    @classmethod
    def get_device_info(cls):
        """Get detailed device information"""
        if torch.cuda.is_available():
            return {
                'device': 'CUDA',
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory // 1024**3,
                'cuda_version': torch.version.cuda,
                'optimal_batch_size': cls.BATCH_SIZE
            }
        else:
            return {
                'device': 'CPU',
                'threads': torch.get_num_threads(),
                'optimal_batch_size': cls.BATCH_SIZE // 2
            }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        dirs = [cls.DATA_DIR, cls.BACKUP_DIR]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directories: {', '.join(dirs)}")
    
    @classmethod  
    def estimate_training_time(cls):
        """Estimate training time based on episodes and hardware"""
        if torch.cuda.is_available():
            episodes_per_hour = 800  # RTX 2060 estimate
        else:
            episodes_per_hour = 200  # CPU estimate
        
        hours = cls.EPISODES / episodes_per_hour
        return {
            'total_hours': round(hours, 1),
            'episodes_per_hour': episodes_per_hour,
            'estimated_storage_gb': round(cls.EPISODES / 1000 * 0.5, 1)
        }
    
    @classmethod
    def print_config_summary(cls):
        """Print configuration summary"""
        device_info = cls.get_device_info()
        time_estimate = cls.estimate_training_time()
        
        print("🚀 CHESS DQN CONFIGURATION")
        print("=" * 50)
        print(f"📊 Episodes: {cls.EPISODES:,}")
        print(f"💻 Device: {device_info['device']}")
        if 'gpu_name' in device_info:
            print(f"🎮 GPU: {device_info['gpu_name']}")
            print(f"💾 GPU Memory: {device_info['gpu_memory']}GB")
        print(f"⚡ Batch Size: {cls.BATCH_SIZE}")
        print(f"🕐 Estimated Time: {time_estimate['total_hours']} hours")
        print(f"💽 Estimated Storage: {time_estimate['estimated_storage_gb']}GB")
        print(f"💾 Checkpoint Every: {cls.SAVE_FREQUENCY} episodes")
        print("=" * 50)

# Usage examples:
if __name__ == "__main__":
    # Test configuration
    config = ChessDQNConfig()
    config.create_directories()
    config.print_config_summary()
    
    # Modify episodes for different training phases:
    # config.EPISODES = 1000      # Quick training
    # config.EPISODES = 5000      # Standard training  
    # config.EPISODES = 15000     # Extensive training