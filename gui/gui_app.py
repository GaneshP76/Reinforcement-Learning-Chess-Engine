import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import chess
import chess.svg
import io
import random
import cairosvg
import torch

# Import enhanced agent
from agents.enhanced_dqn_agent import EnhancedDQNAgent
from utils.utils import board_to_tensor
from utils import move_encoder

class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Chess DQN vs Human")
        self.board = chess.Board()
        self.selected_square = None
        self.game_moves = []  # Track moves for learning
        
        # Initialize enhanced agent
        self.agent = EnhancedDQNAgent()
        
        # GUI setup
        self.setup_gui()
        self.update_board()
        
        print("🎯 Enhanced Chess GUI loaded!")
        print(f"🤖 Agent info: {self.agent.get_model_info()}")

    def setup_gui(self):
        """Setup the GUI components"""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Chess board canvas
        self.canvas = tk.Canvas(main_frame, width=480, height=480, bg='white')
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Control panel
        control_frame = tk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Game info
        info_frame = tk.LabelFrame(control_frame, text="Game Info", padx=5, pady=5)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.turn_label = tk.Label(info_frame, text="White to move", font=('Arial', 12, 'bold'))
        self.turn_label.pack()
        
        self.status_label = tk.Label(info_frame, text="Game in progress", font=('Arial', 10))
        self.status_label.pack()
        
        # Agent info
        agent_frame = tk.LabelFrame(control_frame, text="AI Agent", padx=5, pady=5)
        agent_frame.pack(fill=tk.X, pady=(0, 10))
        
        agent_info = self.agent.get_model_info()
        tk.Label(agent_frame, text=f"Games learned: {agent_info['games_learned']}", font=('Arial', 9)).pack()
        tk.Label(agent_frame, text=f"Exploration: {agent_info['epsilon']:.3f}", font=('Arial', 9)).pack()
        tk.Label(agent_frame, text=f"Parameters: {agent_info['parameters']:,}", font=('Arial', 9)).pack()
        
        # Controls
        controls_frame = tk.LabelFrame(control_frame, text="Controls", padx=5, pady=5)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(controls_frame, text="New Game", command=self.new_game, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 5))
        tk.Button(controls_frame, text="Undo Move", command=self.undo_move, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 5))
        tk.Button(controls_frame, text="Get Hint", command=self.get_hint, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 5))
        
        # Rating input
        rating_frame = tk.LabelFrame(control_frame, text="Your Rating", padx=5, pady=5)
        rating_frame.pack(fill=tk.X)
        
        self.rating_var = tk.StringVar(value="1500")
        tk.Label(rating_frame, text="Enter your chess rating:", font=('Arial', 9)).pack()
        self.rating_entry = tk.Entry(rating_frame, textvariable=self.rating_var, width=10)
        self.rating_entry.pack(pady=5)

    def update_board(self):
        """Update the chess board display"""
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
        
        self.root.update()

    def ask_promotion_piece(self):
        """Ask user for pawn promotion piece"""
        piece_map = {
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT
        }
        piece = simpledialog.askstring(
            "Pawn Promotion", 
            "Promote pawn to (Queen/Rook/Bishop/Knight):",
            initialvalue="Queen"
        )
        if piece and piece.capitalize() in piece_map:
            return piece_map[piece.capitalize()]
        return chess.QUEEN

    def on_click(self, event):
        """Handle mouse clicks on the chess board"""
        if self.board.is_game_over() or self.board.turn == chess.BLACK:
            return

        row = event.y // 60
        col = event.x // 60
        square = chess.square(col, 7 - row)

        if self.selected_square is None:
            # Select a piece
            piece = self.board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = square
        else:
            # Try to make a move
            move = chess.Move(self.selected_square, square)

            # Handle pawn promotion
            piece = self.board.piece_at(self.selected_square)
            if (piece and piece.piece_type == chess.PAWN and 
                chess.square_rank(square) in [0, 7]):
                move.promotion = self.ask_promotion_piece()

            if move in self.board.legal_moves:
                # Make the move
                self.board.push(move)
                self.game_moves.append(move)
                self.selected_square = None
                self.update_board()
                self.check_game_over()
                
                # Let AI move after a short delay
                if not self.board.is_game_over():
                    self.root.after(500, self.make_ai_move)
            else:
                self.selected_square = None

    def make_ai_move(self):
        """Make AI move using enhanced agent"""
        if self.board.is_game_over():
            return

        self.status_label.config(text="AI is thinking...")
        self.root.update()

        try:
            # Get AI move with some thinking time simulation
            move = self.agent.choose_move(self.board, temperature=0.2)
            
            if move and move in self.board.legal_moves:
                self.board.push(move)
                self.game_moves.append(move)
                print(f"🤖 AI played: {move}")
            else:
                # Fallback to random move
                legal_moves = list(self.board.legal_moves)
                if legal_moves:
                    move = random.choice(legal_moves)
                    self.board.push(move)
                    self.game_moves.append(move)
                    print(f"🤖 AI played (random): {move}")

        except Exception as e:
            print(f"⚠️ AI move error: {e}")
            # Fallback to random move
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                move = random.choice(legal_moves)
                self.board.push(move)
                self.game_moves.append(move)

        self.status_label.config(text="Game in progress")
        self.update_board()
        self.check_game_over()

    def check_game_over(self):
        """Check if game is over and handle the result"""
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
            
            # Let AI learn from the game
            try:
                human_rating = int(self.rating_var.get())
            except:
                human_rating = 1500
            
            self.agent.learn_from_game(self.game_moves, game_result)
            
            # If it's a human vs AI game, enable continuous learning
            if hasattr(self.agent, 'continuous_learner'):
                print(f"🎓 AI learning from game against {human_rating}-rated player")

    def new_game(self):
        """Start a new game"""
        # Save current agent state
        self.agent.save_model()
        
        # Reset board
        self.board = chess.Board()
        self.selected_square = None
        self.game_moves = []
        
        self.status_label.config(text="New game started")
        self.update_board()

    def undo_move(self):
        """Undo the last move(s)"""
        if len(self.board.move_stack) >= 2:  # Undo both human and AI moves
            self.board.pop()  # AI move
            self.board.pop()  # Human move
            if len(self.game_moves) >= 2:
                self.game_moves.pop()
                self.game_moves.pop()
        elif len(self.board.move_stack) == 1:
            self.board.pop()
            if self.game_moves:
                self.game_moves.pop()
        
        self.selected_square = None
        self.update_board()

    def get_hint(self):
        """Get a move hint from the AI"""
        if self.board.turn == chess.WHITE and not self.board.is_game_over():
            try:
                hint_move = self.agent.choose_move(self.board, temperature=0.0)  # Best move
                if hint_move:
                    from_square = chess.square_name(hint_move.from_square)
                    to_square = chess.square_name(hint_move.to_square)
                    hint_text = f"Suggested move: {from_square} to {to_square}"
                    if hint_move.promotion:
                        piece_names = {
                            chess.QUEEN: "Queen",
                            chess.ROOK: "Rook", 
                            chess.BISHOP: "Bishop",
                            chess.KNIGHT: "Knight"
                        }
                        hint_text += f" (promote to {piece_names[hint_move.promotion]})"
                    
                    messagebox.showinfo("Move Hint", hint_text)
                else:
                    messagebox.showinfo("Move Hint", "No move suggestion available")
            except Exception as e:
                print(f"Hint error: {e}")
                messagebox.showinfo("Move Hint", "Unable to generate hint")

    def on_closing(self):
        """Handle window closing"""
        # Save agent's learning progress
        self.agent.save_model()
        print("💾 Agent progress saved!")
        self.root.destroy()

def main():
    """Main function to run the enhanced chess GUI"""
    root = tk.Tk()
    app = ChessApp(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()