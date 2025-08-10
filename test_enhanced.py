#!/usr/bin/env python3
"""
Test script for enhanced chess DQN
Run this to make sure everything is working correctly
"""

import chess
import torch
import sys
import os

def test_enhanced_model():
    """Test the enhanced chess model"""
    print("🧪 Testing Enhanced Chess DQN Model")
    print("=" * 50)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from agents.enhanced_dqn_agent import EnhancedDQNAgent, EnhancedDQNModel, ChessEvaluator
        from training.training_components import OpeningBook, PrioritizedReplayBuffer
        from utils.utils import board_to_tensor
        from utils import move_encoder
        print("✅ All imports successful!")
        
        # Test model creation
        print("\n🧠 Testing model creation...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EnhancedDQNModel().to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created successfully!")
        print(f"   Device: {device}")
        print(f"   Parameters: {param_count:,}")
        
        # Test board conversion
        print("\n♟️  Testing board representation...")
        board = chess.Board()
        tensor = board_to_tensor(board)
        print(f"✅ Board tensor shape: {tensor.shape}")
        
        # Test model forward pass
        print("\n🔄 Testing model forward pass...")
        input_tensor = torch.tensor(tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values, position_value = model(input_tensor)
        print(f"✅ Model output shapes:")
        print(f"   Q-values: {q_values.shape}")
        print(f"   Position value: {position_value.shape}")
        
        # Test move encoding
        print("\n🎯 Testing move encoding...")
        move = chess.Move.from_uci("e2e4")
        move_index = move_encoder.move_to_index(move)
        decoded_move = move_encoder.index_to_move(move_index, board)
        print(f"✅ Move encoding/decoding:")
        print(f"   Original: {move}")
        print(f"   Index: {move_index}")
        print(f"   Decoded: {decoded_move}")
        
        # Test agent creation
        print("\n🤖 Testing enhanced agent...")
        agent = EnhancedDQNAgent()
        agent_info = agent.get_model_info()
        print(f"✅ Enhanced agent created!")
        print(f"   Games learned: {agent_info['games_learned']}")
        print(f"   Exploration rate: {agent_info['epsilon']:.3f}")
        
        # Test move selection
        print("\n🎲 Testing move selection...")
        move = agent.choose_move(board)
        print(f"✅ Agent chose move: {move}")
        
        # Test evaluator
        print("\n📊 Testing position evaluator...")
        evaluator = ChessEvaluator()
        eval_score = evaluator.evaluate_position(board)
        print(f"✅ Position evaluation: {eval_score:.3f}")
        
        # Test opening book
        print("\n📚 Testing opening book...")
        opening_book = OpeningBook()
        opening_move = opening_book.get_opening_move(board)
        print(f"✅ Opening book move: {opening_move}")
        
        # Test replay buffer
        print("\n💾 Testing prioritized replay buffer...")
        replay_buffer = PrioritizedReplayBuffer()
        test_experience = (tensor, 0, 0.1, tensor, False)
        replay_buffer.push(test_experience)
        print(f"✅ Replay buffer size: {len(replay_buffer)}")
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your enhanced chess DQN is ready to train!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"📍 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """Test if training can start"""
    print("\n🏋️ Testing training setup...")
    
    try:
        from training.enhanced_train import EnhancedChessTrainer
        
        # Just test initialization (don't start training)
        trainer = EnhancedChessTrainer()
        print("✅ Training setup successful!")
        print(f"   Checkpoint path: {trainer.checkpoint_path}")
        print(f"   Current epsilon: {trainer.epsilon}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training setup failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced Chess DQN Test Suite")
    print("Testing all components before training...")
    
    # Run tests
    model_test = test_enhanced_model()
    training_test = test_training_setup()
    
    if model_test and training_test:
        print("\n🎯 READY TO START TRAINING!")
        print("Run: python training/enhanced_train.py")
        print("\n🎮 READY TO PLAY!")
        print("Run: python gui/gui_app.py")
    else:
        print("\n❌ Some tests failed. Please fix the issues before proceeding.")
        sys.exit(1)