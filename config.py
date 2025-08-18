import torch
import os
import psutil
import time

class ChessDQNConfig:
    """FIXED Configuration for Chess DQN Training - CLEAR LEARNING SIGNALS"""
    
    def __init__(self):
        # 🚀 DEVICE SETUP
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._print_device_info()
        
        # 🧠 TRAINING PARAMETERS - OPTIMIZED FOR LEARNING
        self.EPISODES = 30000           # More episodes needed for chess mastery
        self.BATCH_SIZE = 64            # Reduced for RTX 2060 memory
        self.LEARNING_RATE = 0.0003     # Slower learning for stability
        self.GAMMA = 0.99               # Discount factor
        
        # 🎯 EXPLORATION - FIXED SCHEDULE
        self.EPSILON_START = 1.0        # Start with full exploration
        self.EPSILON_END = 0.05         # End with minimal exploration
        self.EPSILON_DECAY = 0.9995     # Very slow decay (important!)
        
        # 💾 MEMORY AND UPDATES - OPTIMIZED
        self.MAX_REPLAY_BUFFER = 100000 # Large buffer for diverse experiences
        self.TARGET_UPDATE_FREQ = 500   # Update target network less frequently
        self.MAX_GAME_LENGTH = 200      # Prevent infinite games
        
        # 📁 PATHS
        self.DATA_DIR = "data"
        self.CHECKPOINT_PATH = "data/fixed_enhanced_dqn_checkpoint.pth"
        self.LOG_PATH = "data/fixed_training_log.csv"
        
        # 📈 TRAINING FREQUENCIES
        self.SAVE_FREQUENCY = 200       # Save every 200 episodes
        self.EVAL_FREQUENCY = 1000      # Evaluate every 1000 episodes  
        self.PRINT_FREQUENCY = 100      # Print progress every 100 episodes
        
        # 🔧 ADVANCED PARAMETERS
        self.MIN_REPLAY_SIZE = 1000     # Minimum experiences before training
        self.PRIORITY_ALPHA = 0.6       # Prioritized replay parameter
        self.PRIORITY_BETA = 0.4        # Importance sampling parameter
        
        # Create necessary directories
        self.create_directories()
        
    def _print_device_info(self):
        """Print device and system information"""
        print(f"🚀 Device: {self.DEVICE}")
        
        if self.DEVICE.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f}GB")
            
            # Check if it's RTX 2060 (give specific advice)
            if "2060" in gpu_name:
                print("   ✅ RTX 2060 detected - perfect for chess training!")
                print("   💡 Batch size optimized for your GPU")
        else:
            print("   ⚠️ Using CPU - training will be slower")
            print("   💡 Consider reducing BATCH_SIZE to 32 if memory issues")
        
        # System RAM info
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f"   RAM: {ram_gb:.1f}GB")
        
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.DATA_DIR,
            "data/backups",
            "data/logs",
            "data/models"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"📁 Created directories: {', '.join(directories)}")
    
    def print_config_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("🔧 FIXED CHESS DQN CONFIGURATION")
        print("="*60)
        print(f"🎯 Episodes: {self.EPISODES:,}")
        print(f"🧠 Batch Size: {self.BATCH_SIZE}")
        print(f"📚 Learning Rate: {self.LEARNING_RATE}")
        print(f"🎲 Epsilon: {self.EPSILON_START} → {self.EPSILON_END}")
        print(f"💾 Replay Buffer: {self.MAX_REPLAY_BUFFER:,}")
        print(f"🚀 Device: {self.DEVICE}")
        
        print(f"\n📈 Training Schedule:")
        print(f"   💾 Save every: {self.SAVE_FREQUENCY} episodes")
        print(f"   🎯 Evaluate every: {self.EVAL_FREQUENCY} episodes") 
        print(f"   📊 Print every: {self.PRINT_FREQUENCY} episodes")
        
        print(f"\n📁 Key Files:")
        print(f"   🧠 Model: {self.CHECKPOINT_PATH}")
        print(f"   📊 Logs: {self.LOG_PATH}")
        
        print("="*60)
    
    def estimate_training_time(self):
        """Estimate training time based on system"""
        # Base estimates (in seconds per episode)
        if self.DEVICE.type == "cuda":
            # GPU estimates
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "2060" in gpu_name:
                base_time = 3.0  # RTX 2060 estimate
            elif "2070" in gpu_name or "2080" in gpu_name:
                base_time = 2.5  # Faster RTX cards
            elif "1060" in gpu_name or "1070" in gpu_name:
                base_time = 4.0  # Older GTX cards
            else:
                base_time = 3.5  # Generic GPU estimate
        else:
            base_time = 8.0  # CPU is much slower
        
        # Calculate total time
        total_seconds = self.EPISODES * base_time
        total_hours = total_seconds / 3600
        
        return {
            'seconds_per_episode': base_time,
            'total_seconds': total_seconds,
            'total_hours': total_hours,
            'estimated_completion': f"{total_hours:.1f} hours ({total_hours/24:.1f} days)"
        }
    
    def get_memory_requirements(self):
        """Get memory requirements"""
        # Model memory (approximate)
        model_params = 20_000_000  # ~20M parameters
        model_memory_gb = (model_params * 4 * 2) / 1e9  # 4 bytes per param, 2 models
        
        # Replay buffer memory
        replay_memory_gb = (self.MAX_REPLAY_BUFFER * 0.001)  # Rough estimate
        
        # Total GPU memory needed
        total_gpu_gb = model_memory_gb + replay_memory_gb + 1.0  # +1GB overhead
        
        # System RAM needed
        system_ram_gb = 4.0  # Base system needs
        
        return {
            'model_memory_gb': model_memory_gb,
            'replay_memory_gb': replay_memory_gb,
            'total_gpu_gb': total_gpu_gb,
            'system_ram_gb': system_ram_gb
        }
    
    def check_system_compatibility(self):
        """Check if system can handle training"""
        memory_req = self.get_memory_requirements()
        issues = []
        
        # Check GPU memory
        if self.DEVICE.type == "cuda":
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory_gb < memory_req['total_gpu_gb']:
                issues.append(f"GPU memory low: {gpu_memory_gb:.1f}GB < {memory_req['total_gpu_gb']:.1f}GB needed")
                issues.append("Solution: Reduce BATCH_SIZE to 32 or 16")
        
        # Check system RAM
        system_ram_gb = psutil.virtual_memory().total / 1e9
        if system_ram_gb < memory_req['system_ram_gb']:
            issues.append(f"System RAM low: {system_ram_gb:.1f}GB < {memory_req['system_ram_gb']:.1f}GB needed")
        
        # Check disk space
        free_space = psutil.disk_usage('.').free / 1e9
        if free_space < 5.0:  # Need at least 5GB
            issues.append(f"Disk space low: {free_space:.1f}GB free, need 5GB+")
        
        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'memory_requirements': memory_req
        }
    
    def optimize_for_system(self):
        """Auto-optimize settings based on system capabilities"""
        compat = self.check_system_compatibility()
        
        if not compat['compatible']:
            print("⚠️ System optimization needed!")
            
            for issue in compat['issues']:
                print(f"   • {issue}")
                
                # Auto-fix common issues
                if "GPU memory low" in issue:
                    old_batch = self.BATCH_SIZE
                    self.BATCH_SIZE = max(16, self.BATCH_SIZE // 2)
                    print(f"   🔧 Auto-reduced batch size: {old_batch} → {self.BATCH_SIZE}")
                
                elif "System RAM low" in issue:
                    old_buffer = self.MAX_REPLAY_BUFFER
                    self.MAX_REPLAY_BUFFER = max(50000, self.MAX_REPLAY_BUFFER // 2)
                    print(f"   🔧 Auto-reduced replay buffer: {old_buffer:,} → {self.MAX_REPLAY_BUFFER:,}")
        
        else:
            print("✅ System is compatible with current settings!")
    
    def save_config(self, filepath=None):
        """Save current configuration"""
        if filepath is None:
            filepath = os.path.join(self.DATA_DIR, "training_config.txt")
        
        with open(filepath, 'w') as f:
            f.write("FIXED CHESS DQN CONFIGURATION\n")
            f.write("="*50 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("TRAINING PARAMETERS:\n")
            f.write(f"Episodes: {self.EPISODES:,}\n")
            f.write(f"Batch Size: {self.BATCH_SIZE}\n") 
            f.write(f"Learning Rate: {self.LEARNING_RATE}\n")
            f.write(f"Gamma: {self.GAMMA}\n")
            f.write(f"Device: {self.DEVICE}\n\n")
            
            f.write("EXPLORATION:\n")
            f.write(f"Epsilon Start: {self.EPSILON_START}\n")
            f.write(f"Epsilon End: {self.EPSILON_END}\n")
            f.write(f"Epsilon Decay: {self.EPSILON_DECAY}\n\n")
            
            f.write("MEMORY:\n")
            f.write(f"Replay Buffer: {self.MAX_REPLAY_BUFFER:,}\n")
            f.write(f"Target Update Freq: {self.TARGET_UPDATE_FREQ}\n\n")
            
            f.write("PATHS:\n")
            f.write(f"Checkpoint: {self.CHECKPOINT_PATH}\n")
            f.write(f"Logs: {self.LOG_PATH}\n")
        
        print(f"💾 Configuration saved to {filepath}")

# SIMPLIFIED REWARD CONFIG
class RewardConfig:
    """FIXED reward configuration - clear learning signals"""
    
    # 🏆 GAME ENDING REWARDS (MASSIVE)
    CHECKMATE_REWARD = 10.0
    STALEMATE_PENALTY = -3.0
    
    # 🏰 SPECIAL MOVE REWARDS (BIG)
    CASTLING_REWARD = 3.0
    QUEEN_PROMOTION_REWARD = 8.0
    OTHER_PROMOTION_REWARD = 4.0
    
    # 🎯 TACTICAL REWARDS (MEDIUM)
    PIECE_VALUES = {
        1: 1.0,   # Pawn
        2: 3.0,   # Knight
        3: 3.0,   # Bishop  
        4: 5.0,   # Rook
        5: 9.0,   # Queen
        6: 0.0    # King
    }
    
    # ⚔️ SMALL BONUSES
    CHECK_BONUS = 0.5
    
    # 🔄 PENALTIES
    REPETITION_PENALTY = -0.2
    TIME_PENALTY = -0.01  # Encourage decisive play
    
    @classmethod
    def print_reward_structure(cls):
        """Print reward structure"""
        print("\n🎯 FIXED REWARD STRUCTURE:")
        print("="*40)
        print(f"🏆 Checkmate: +{cls.CHECKMATE_REWARD}")
        print(f"😐 Stalemate: {cls.STALEMATE_PENALTY}")
        print(f"🏰 Castling: +{cls.CASTLING_REWARD}")
        print(f"👑 Queen Promotion: +{cls.QUEEN_PROMOTION_REWARD}")
        print(f"⚔️ Check: +{cls.CHECK_BONUS}")
        print("\n🎯 Piece Values:")
        for piece_type, value in cls.PIECE_VALUES.items():
            piece_names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
            if piece_type != 6:  # Skip king
                print(f"   {piece_names[piece_type]}: +{value}")
        print("="*40)

def main():
    """Test configuration"""
    print("🧪 Testing FIXED configuration...")
    
    config = ChessDQNConfig()
    config.print_config_summary()
    
    # Check system compatibility
    print("\n🔍 System Compatibility Check:")
    compat = config.check_system_compatibility()
    
    if compat['compatible']:
        print("✅ System is ready for training!")
    else:
        print("⚠️ Issues found:")
        for issue in compat['issues']:
            print(f"   • {issue}")
        
        print("\n🔧 Applying automatic optimizations...")
        config.optimize_for_system()
    
    # Show reward structure
    RewardConfig.print_reward_structure()
    
    # Estimate training time
    time_est = config.estimate_training_time()
    print(f"\n⏰ Training Time Estimate:")
    print(f"   Per episode: ~{time_est['seconds_per_episode']:.1f} seconds")
    print(f"   Total time: {time_est['estimated_completion']}")
    
    # Save config
    config.save_config()
    
    print("\n🚀 Configuration ready!")

if __name__ == "__main__":
    main()