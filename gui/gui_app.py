# gui/gui_app.py - ENHANCED VERSION
# Add these imports at the top of your existing gui_app.py:

import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import chess
import chess.svg
import io
import random
import cairosvg
import torch
import os

# Import enhanced components
from agents.enhanced_dqn_agent import EnhancedDQNAgent
from utils.utils import board_to_tensor
from utils import move_encoder
from config import ChessDQNConfig  # NEW
from utils.system_monitor import SystemMonitor  # NEW

class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Chess DQN vs Human")
        self.board = chess.Board()
        self.selected_square = None
        self.game_moves = []  # Track moves for learning
        
        # Load configuration
        self.config = ChessDQNConfig()
        self.monitor = SystemMonitor(self.config.DATA_DIR)
        
        # Initialize enhanced agent with config
        checkpoint_path = self.config.CHECKPOINT_PATH
        if not os.path.exists(checkpoint_path):
            # Try alternative paths
            alt_paths = [
                "data/best_enhanced_model.pth",
                "data/enhanced_dqn_checkpoint.pth",
                "data/dqn_checkpoint.pth"
            ]
            for path in alt_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
        
        self.agent = EnhancedDQNAgent(checkpoint_path)
        
        # GUI setup
        self.setup_gui()
        self.update_board()
        
        print("🎯 Enhanced Chess GUI loaded!")
        print(f"🤖 Agent info: {self.agent.get_model_info()}")
        print(f"💾 Using model: {checkpoint_path}")

    def setup_gui(self):
        """Setup the GUI components with enhancements"""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Chess board canvas
        self.canvas = tk.Canvas(main_frame, width=480, height=480, bg='white')
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Control panel
        control_frame = tk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Game info
        info_frame = tk.LabelFrame(control_frame, text="Game Info", padx=5, pady=5)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.turn_label = tk.Label(info_frame, text="White to move", font=('Arial', 12, 'bold'))
        self.turn_label.pack()
        
        self.status_label = tk.Label(info_frame, text="Game in progress", font=('Arial', 10))
        self.status_label.pack()
        
        # Move counter
        self.move_counter = tk.Label(info_frame, text="Move: 1", font=('Arial', 9))
        self.move_counter.pack()
        
        # Agent info (Enhanced)
        agent_frame = tk.LabelFrame(control_frame, text="AI Agent", padx=5, pady=5)
        agent_frame.pack(fill=tk.X, pady=(0, 10))
        
        agent_info = self.agent.get_model_info()
        self.games_learned_label = tk.Label(agent_frame, text=f"Games learned: {agent_info['games_learned']}", font=('Arial', 9))
        self.games_learned_label.pack()
        
        self.epsilon_label = tk.Label(agent_frame, text=f"Exploration: {agent_info['epsilon']:.3f}", font=('Arial', 9))
        self.epsilon_label.pack()
        
        tk.Label(agent_frame, text=f"Parameters: {agent_info['parameters']:,}", font=('Arial', 9)).pack()
        tk.Label(agent_frame, text=f"Device: {agent_info['device']}", font=('Arial', 9)).pack()
        
        # System monitoring (NEW)
        system_frame = tk.LabelFrame(control_frame, text="System Status", padx=5, pady=5)
        system_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.update_system_info(system_frame)
        
        # Controls
        controls_frame = tk.LabelFrame(control_frame, text="Controls", padx=5, pady=5)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(controls_frame, text="New Game", command=self.new_game, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 5))
        tk.Button(controls_frame, text="Undo Move", command=self.undo_move, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 5))
        tk.Button(controls_frame, text="Get Hint", command=self.get_hint, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 5))
        
        # NEW: Analysis button
        tk.Button(controls_frame, text="Analyze Position", command=self.analyze_position, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 5))
        
        # Rating input (Enhanced)
        rating_frame = tk.LabelFrame(control_frame, text="Your Rating", padx=5, pady=5)
        rating_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.rating_var = tk.StringVar(value="1500")
        tk.Label(rating_frame, text="Enter your chess rating:", font=('Arial', 9)).pack()
        self.rating_entry = tk.Entry(rating_frame, textvariable=self.rating_var, width=10)
        self.rating_entry.pack(pady=5)
        
        # Game history (NEW)
        history_frame = tk.LabelFrame(control_frame, text="Recent Games", padx=5, pady=5)
        history_frame.pack(fill=tk.X)
        
        self.history_text = tk.Text(history_frame, height=4, width=25, font=('Arial', 8))
        self.history_text.pack(pady=5)
        self.update_game_history()

    def update_system_info(self, parent_frame):
        """Update system information display"""
        try:
            health = self.monitor.check_system_health()
            memory = health['memory']
            gpu = health['gpu']
            
            # Memory info
            tk.Label(parent_frame, text=f"RAM: {memory['used_gb']:.1f}/{memory['total_gb']:.1f}GB", 
                    font=('Arial', 9)).pack()
            
            # GPU info
            if gpu:
                tk.Label(parent_frame, text=f"GPU: {gpu['allocated_gb']:.1f}/{gpu['total_gb']:.1f}GB", 
                        font=('Arial', 9)).pack()
            else:
                tk.Label(parent_frame, text="GPU: Not available", font=('Arial', 9)).pack()
            
            # Health status
            status_color = "green" if health['healthy'] else "orange"
            status_text = "Healthy" if health['healthy'] else "Limited"
            status_label = tk.Label(parent_frame, text=f"Status: {status_text}", 
                                  font=('Arial', 9), fg=status_color)
            status_label.pack()
            
        except Exception as e:
            tk.Label(parent_frame, text="System info unavailable", font=('Arial', 9)).pack()

    def update_game_history(self):
        """Update game history display"""
        try:
            # Read recent games from log
            history_text = "Recent performance:\n"
            
            # Simple placeholder - you could enhance this
            agent_info = self.agent.get_model_info()
            games_count = agent_info['games_learned']
            
            if games_count > 0:
                history_text += f"• {games_count} games learned\n"
                history_text += f"• Exploration: {agent_info['epsilon']:.1%}\n"
            else:
                history_text += "• No games played yet\n"
                history_text += "• Ready for first game!\n"
            
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(tk.END, history_text)
            
        except Exception as e:
            self.history_text.delete(1.0, tk.END)
            self.history_text.insert(tk.END, "History unavailable")

    def analyze_position(self):
        """NEW: Analyze current position"""
        try:
            from agents.enhanced_dqn_agent import ChessEvaluator
            
            evaluator = ChessEvaluator()
            score = evaluator.evaluate_position(self.board)
            
            # Simple analysis
            if score > 0.5:
                analysis = "White has a significant advantage"
            elif score > 0.1:
                analysis = "White is slightly better"
            elif score > -0.1:
                analysis = "Position is roughly equal"
            elif score > -0.5:
                analysis = "Black is slightly better"
            else:
                analysis = "Black has a significant advantage"
            
            # Additional info
            legal_moves = len(list(self.board.legal_moves))
            game_phase = "Opening" if len(self.board.move_stack) < 20 else "Middlegame" if len(self.board.piece_map()) > 12 else "Endgame"
            
            analysis_text = f"Position Analysis:\n\n{analysis}\n\nScore: {score:.3f}\nLegal moves: {legal_moves}\nGame phase: {game_phase}"
            
            messagebox.showinfo("Position Analysis", analysis_text)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Could not analyze position: {e}")

    def update_board(self):
        """Update the chess board display with enhancements"""
        svg_data = chess.svg.board(
            self.board, 
            size=480,
            lastmove=self.board.peek() if self.board.move_stack else None
        )
        png_bytes = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
        img_data = Image.open(io.BytesIO(png_bytes))
        self.photo = ImageTk.PhotoImage(img_data)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Update turn indicator
        if not self.board.is_game_over():
            turn_text = "White to move" if self.board.turn == chess.WHITE else "Black (AI) thinking..."
            self.turn_label.config(text=turn_text)
        
        # Update move counter
        move_num = (len(self.board.move_stack) // 2) + 1
        self.move_counter.config(text=f"Move: {move_num}")
        
        # Update agent info
        try:
            agent_info = self.agent.get_model_info()
            self.games_learned_label.config(text=f"Games learned: {agent_info['games_learned']}")
            self.epsilon_label.config(text=f"Exploration: {agent_info['epsilon']:.3f}")
        except:
            pass
        
        self.root.update()

    # ... (rest of your existing methods remain the same)
    
    def check_game_over(self):
        """Check if game is over and handle the result with enhanced learning"""
        if self.board.is_game_over():
            result = self.board.result()
            
            # Determine result message
            if result == '1-0':
                msg = "Congratulations! You won! 🎉"
                game_result = "1-0"
            elif result == '0-1':
                msg = "AI won this time! 🤖"
                game_result = "0-1"
            else:
                msg = "It's a draw! 🤝"
                game_result = "1/2-1/2"
            
            self.status_label.config(text="Game Over")
            messagebox.showinfo("Game Over", msg)
            
            # Enhanced learning from human games
            try:
                human_rating = int(self.rating_var.get())
            except:
                human_rating = 1500
            
            # Let AI learn from the game
            self.agent.learn_from_game(self.game_moves, game_result)
            
            # Additional learning if available
            if hasattr(self.agent, 'learn_from_human_game'):
                self.agent.learn_from_human_game(self.game_moves, game_result, human_rating)
                print(f"🎓 AI learning from game against {human_rating}-rated player")
            
            # Update history display
            self.update_game_history()

    def on_closing(self):
        """Handle window closing with enhanced cleanup"""
        # Save agent's learning progress
        try:
            self.agent.save_model()
            print("💾 Agent progress saved!")
        except Exception as e:
            print(f"⚠️ Could not save agent progress: {e}")
        
        # Show final system status
        try:
            print("\n📊 Final System Status:")
            self.monitor.print_resource_summary()
        except:
            pass
        
        self.root.destroy()

def main():
    """Main function to run the enhanced chess GUI with config"""
    print("🎮 Starting Enhanced Chess GUI...")
    
    # Check if any trained model exists
    config = ChessDQNConfig()
    model_paths = [
        config.CHECKPOINT_PATH,
        "data/best_enhanced_model.pth", 
        "data/enhanced_dqn_checkpoint.pth",
        "data/dqn_checkpoint.pth"
    ]
    
    model_found = any(os.path.exists(path) for path in model_paths)
    
    if not model_found:
        print("⚠️ No trained model found!")
        print("🏋️ Run training first: python training/enhanced_train.py")
        print("🤖 Or the AI will play with random initialization")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            return
    
    root = tk.Tk()
    app = ChessApp(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    print("🎯 Chess GUI loaded successfully!")
    print("💡 Tips:")
    print("   • Click pieces to move them")
    print("   • Use 'Get Hint' for move suggestions") 
    print("   • Try 'Analyze Position' for evaluation")
    print("   • Your games help train the AI!")
    
    root.mainloop()

if __name__ == "__main__":
    main()