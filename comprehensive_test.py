#!/usr/bin/env python3
"""
Comprehensive Chess DQN System Test
Tests all components with your RTX 2060 setup
"""

import sys
import os
import time
import torch
import chess

def print_header(title):
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)

def print_step(step, description):
    print(f"\n{step}. {description}")

def test_imports():
    """Test all imports and dependencies"""
    print_step("1", "Testing imports and dependencies...")
    
    try:
        # Core dependencies
        import torch
        import chess
        import numpy as np
        import matplotlib.pyplot as plt
        import PIL
        import cairosvg
        import psutil
        print("✅ Core dependencies loaded")
        
        # Our modules
        from config import ChessDQNConfig
        from utils.system_monitor import SystemMonitor
        from agents.enhanced_dqn_agent import EnhancedDQNAgent, EnhancedDQNModel, ChessEvaluator
        from training.training_components import OpeningBook, PrioritizedReplayBuffer
        from utils.utils import board_to_tensor
        from utils import move_encoder
        print("✅ Custom modules loaded")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_hardware():
    """Test hardware capabilities"""
    print_step("2", "Testing hardware capabilities...")
    
    try:
        # Test CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"✅ CUDA GPU detected: {gpu_name}")
            print(f"✅ GPU Memory: {gpu_memory:.1f}GB")
            
            # Test GPU memory allocation
            test_tensor = torch.randn(1000, 1000).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("✅ GPU memory allocation test passed")
        else:
            print("⚠️ CUDA not available, will use CPU")
        
        # Test CPU
        cpu_count = torch.get_num_threads()
        print(f"✅ CPU threads: {cpu_count}")
        
        # Test memory
        import psutil
        memory = psutil.virtual_memory()
        print(f"✅ System RAM: {memory.total / (1024**3):.1f}GB")
        
        return True
    except Exception as e:
        print(f"❌ Hardware test failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print_step("3", "Testing configuration system...")
    
    try:
        from config import ChessDQNConfig
        
        config = ChessDQNConfig()
        print(f"✅ Default episodes: {config.EPISODES}")
        print(f"✅ Device: {config.DEVICE}")
        print(f"✅ Batch size: {config.BATCH_SIZE}")
        
        # Test device info
        device_info = config.get_device_info()
        print(f"✅ Device info: {device_info['device']}")
        
        # Test time estimation
        time_estimate = config.estimate_training_time()
        print(f"✅ Estimated training time: {time_estimate['total_hours']} hours")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_model_creation():
    """Test model creation and forward pass"""
    print_step("4", "Testing model creation...")
    
    try:
        from agents.enhanced_dqn_agent import EnhancedDQNModel
        from utils.utils import board_to_tensor
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EnhancedDQNModel().to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created with {param_count:,} parameters")
        
        # Test forward pass
        board = chess.Board()
        tensor = board_to_tensor(board)
        input_tensor = torch.tensor(tensor).unsqueeze(0).to(device)
        
        with torch.no_grad():
            q_values, position_value = model(input_tensor)
        
        print(f"✅ Forward pass successful")
        print(f"   Q-values shape: {q_values.shape}")
        print(f"   Position value shape: {position_value.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_move_encoding():
    """Test move encoding system"""
    print_step("5", "Testing move encoding...")
    
    try:
        from utils import move_encoder
        
        # Test basic move encoding
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        
        index = move_encoder.move_to_index(move)
        decoded_move = move_encoder.index_to_move(index, board)
        
        print(f"✅ Move encoding: {move} -> {index} -> {decoded_move}")
        
        if move == decoded_move:
            print("✅ Move encoding/decoding successful")
        else:
            print("❌ Move encoding/decoding mismatch")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Move encoding test failed: {e}")
        return False

def test_evaluator():
    """Test chess position evaluator"""
    print_step("6", "Testing position evaluator...")
    
    try:
        from agents.enhanced_dqn_agent import ChessEvaluator
        
        evaluator = ChessEvaluator()
        
        # Test starting position
        board = chess.Board()
        score = evaluator.evaluate_position(board)
        print(f"✅ Starting position score: {score:.3f}")
        
        # Test after e4
        board.push(chess.Move.from_uci("e2e4"))
        score = evaluator.evaluate_position(board)
        print(f"✅ After e4 score: {score:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Evaluator test failed: {e}")
        return False

def test_opening_book():
    """Test opening book"""
    print_step("7", "Testing opening book...")
    
    try:
        from training.training_components import OpeningBook
        
        opening_book = OpeningBook()
        board = chess.Board()
        
        opening_move = opening_book.get_opening_move(board)
        print(f"✅ Opening book suggestion: {opening_move}")
        
        return True
    except Exception as e:
        print(f"❌ Opening book test failed: {e}")
        return False

def test_system_monitor():
    """Test system monitoring"""
    print_step("8", "Testing system monitoring...")
    
    try:
        from utils.system_monitor import SystemMonitor
        
        monitor = SystemMonitor()
        health = monitor.check_system_health()
        
        print(f"✅ System health check completed")
        print(f"   Healthy: {health['healthy']}")
        print(f"   Warnings: {len(health['warnings'])}")
        
        # Print summary
        monitor.print_resource_summary()
        
        return True
    except Exception as e:
        print(f"❌ System monitor test failed: {e}")
        return False

def test_agent_creation():
    """Test enhanced agent creation"""
    print_step("9", "Testing enhanced agent...")
    
    try:
        from agents.enhanced_dqn_agent import EnhancedDQNAgent
        
        # Create agent (will create new model if no checkpoint exists)
        agent = EnhancedDQNAgent()
        
        # Test agent info
        info = agent.get_model_info()
        print(f"✅ Agent created successfully")
        print(f"   Games learned: {info['games_learned']}")
        print(f"   Exploration rate: {info['epsilon']:.3f}")
        print(f"   Device: {info['device']}")
        
        # Test move selection
        board = chess.Board()
        move = agent.choose_move(board)
        print(f"✅ Agent chose move: {move}")
        
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_mini_training():
    """Test a mini training session"""
    print_step("10", "Testing mini training session...")
    
    try:
        print("⚠️ This will run 5 training episodes to test the training loop...")
        response = input("Proceed with mini training test? (y/n): ").lower()
        
        if response != 'y':
            print("⏭️ Skipping mini training test")
            return True
        
        from training.enhanced_train import EnhancedChessTrainer
        from config import ChessDQNConfig
        
        # Create minimal config
        config = ChessDQNConfig()
        config.EPISODES = 5  # Just 5 episodes
        config.SAVE_FREQUENCY = 2
        config.PRINT_FREQUENCY = 1
        
        trainer = EnhancedChessTrainer(config)
        
        print("🏋️ Starting mini training...")
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✅ Mini training completed in {duration:.1f} seconds")
        print(f"✅ Average time per episode: {duration/5:.1f} seconds")
        
        return True
    except Exception as e:
        print(f"❌ Mini training failed: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmark"""
    print_step("11", "Running performance benchmark...")
    
    try:
        from agents.enhanced_dqn_agent import EnhancedDQNModel
        from utils.utils import board_to_tensor
        import time
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EnhancedDQNModel().to(device)
        
        # Benchmark forward passes
        board = chess.Board()
        tensor = board_to_tensor(board)
        input_tensor = torch.tensor(tensor).unsqueeze(0).to(device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(input_tensor)
        
        # Benchmark
        num_iterations = 100
        start_time = time.time()
        
        for _ in range(num_iterations):
            with torch.no_grad():
                model(input_tensor)
        
        end_time = time.time()
        duration = end_time - start_time
        
        fps = num_iterations / duration
        print(f"✅ Performance benchmark completed")
        print(f"   Forward passes per second: {fps:.1f}")
        print(f"   Time per forward pass: {duration/num_iterations*1000:.2f} ms")
        
        # Estimate training performance
        estimated_episodes_per_hour = fps * 3600 / 50  # Rough estimate
        print(f"   Estimated episodes/hour: {estimated_episodes_per_hour:.0f}")
        
        return True
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 COMPREHENSIVE CHESS DQN SYSTEM TEST")
    print("Testing your RTX 2060 setup...")
    print("This will verify all components are working correctly.")
    
    tests = [
        test_imports,
        test_hardware,
        test_configuration,
        test_model_creation,
        test_move_encoding,
        test_evaluator,
        test_opening_book,
        test_system_monitor,
        test_agent_creation,
        test_mini_training,
        run_performance_benchmark
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print_header("TEST RESULTS")
    print(f"✅ Tests passed: {passed}")
    print(f"❌ Tests failed: {failed}")
    print(f"📊 Success rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Your chess DQN system is ready for training!")
        print("\n📋 Next steps:")
        print("   1. Run: python training/enhanced_train.py")
        print("   2. Choose option 1 (Test Run) for first training")
        print("   3. After training: python gui/gui_app.py")
    else:
        print(f"\n⚠️ {failed} test(s) failed!")
        print("🔧 Please fix the issues before proceeding with training.")
        
        if failed > passed:
            print("\n💡 Common fixes:")
            print("   • pip install psutil")
            print("   • Check CUDA installation")
            print("   • Verify all files are created correctly")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()