# utils/system_monitor.py
"""
System monitoring utilities for Chess DQN training
Monitors memory, storage, and GPU usage
"""

import psutil
import torch
import os
import shutil
from pathlib import Path

class SystemMonitor:
    """Monitor system resources during training"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.initial_disk_usage = self.get_disk_usage()
        
    def get_memory_usage(self):
        """Get current memory usage"""
        memory = psutil.virtual_memory()
        return {
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3),
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3)
        }
    
    def get_gpu_usage(self):
        """Get GPU memory usage if available"""
        if not torch.cuda.is_available():
            return None
        
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'total_gb': total,
            'percent': (allocated / total) * 100
        }
    
    def get_disk_usage(self):
        """Get disk usage for data directory"""
        if not self.data_dir.exists():
            return {'used_gb': 0, 'files': 0}
        
        total_size = 0
        file_count = 0
        
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            'used_gb': total_size / (1024**3),
            'files': file_count
        }
    
    def get_training_storage_growth(self):
        """Calculate storage growth since training started"""
        current = self.get_disk_usage()
        growth_gb = current['used_gb'] - self.initial_disk_usage['used_gb']
        return max(0, growth_gb)
    
    def check_system_health(self):
        """Check if system can continue training safely"""
        memory = self.get_memory_usage()
        gpu = self.get_gpu_usage()
        disk = self.get_disk_usage()
        
        warnings = []
        
        # Memory checks
        if memory['percent'] > 90:
            warnings.append("⚠️ High RAM usage (>90%)")
        
        # GPU checks
        if gpu and gpu['percent'] > 95:
            warnings.append("⚠️ High GPU memory usage (>95%)")
        
        # Disk checks
        free_space_gb = shutil.disk_usage(self.data_dir.parent).free / (1024**3)
        if free_space_gb < 2:
            warnings.append("⚠️ Low disk space (<2GB free)")
        
        return {
            'healthy': len(warnings) == 0,
            'warnings': warnings,
            'memory': memory,
            'gpu': gpu,
            'disk': disk,
            'free_space_gb': free_space_gb
        }
    
    def print_resource_summary(self):
        """Print current resource usage"""
        health = self.check_system_health()
        memory = health['memory']
        gpu = health['gpu']
        disk = health['disk']
        
        print("\n📊 SYSTEM RESOURCES")
        print("-" * 30)
        print(f"🧠 RAM: {memory['used_gb']:.1f}/{memory['total_gb']:.1f}GB ({memory['percent']:.1f}%)")
        
        if gpu:
            print(f"🎮 GPU: {gpu['allocated_gb']:.1f}/{gpu['total_gb']:.1f}GB ({gpu['percent']:.1f}%)")
        else:
            print("🎮 GPU: Not available")
        
        print(f"💽 Data: {disk['used_gb']:.1f}GB ({disk['files']} files)")
        print(f"📈 Growth: +{self.get_training_storage_growth():.1f}GB since start")
        print(f"💾 Free: {health['free_space_gb']:.1f}GB available")
        
        # Print warnings
        for warning in health['warnings']:
            print(warning)
        
        if health['healthy']:
            print("✅ System healthy for continued training")
        
        return health
    
    def cleanup_old_files(self, keep_latest=5):
        """Clean up old checkpoint files to save space"""
        checkpoint_pattern = self.data_dir.glob("*checkpoint*.pth")
        backup_pattern = self.data_dir.glob("backup_*.pth")
        
        all_files = list(checkpoint_pattern) + list(backup_pattern)
        if len(all_files) <= keep_latest:
            return 0
        
        # Sort by modification time (newest first)
        all_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        files_to_delete = all_files[keep_latest:]
        deleted_size = 0
        
        for file_path in files_to_delete:
            deleted_size += file_path.stat().st_size
            file_path.unlink()
        
        deleted_gb = deleted_size / (1024**3)
        print(f"🧹 Cleaned {len(files_to_delete)} old files, freed {deleted_gb:.1f}GB")
        return deleted_gb

# Example usage in training loop:
def add_monitoring_to_trainer(total_episodes=1000):
    """Example of how to integrate monitoring into your trainer"""
    
    monitor = SystemMonitor()
    
    # Example training loop with monitoring
    for episode in range(total_episodes):
        # ... your training code would go here ...
        
        # Check system every 100 episodes
        if episode % 100 == 0:
            health = monitor.print_resource_summary()
            
            # Auto-cleanup if low on disk space
            if health['free_space_gb'] < 5:
                monitor.cleanup_old_files()
            
            # Pause if system is struggling
            if not health['healthy']:
                input("⚠️ System resources low. Press Enter to continue or Ctrl+C to stop...")
                
    print(f"✅ Example training completed for {total_episodes} episodes")

if __name__ == "__main__":
    # Test monitoring
    monitor = SystemMonitor()
    monitor.print_resource_summary()