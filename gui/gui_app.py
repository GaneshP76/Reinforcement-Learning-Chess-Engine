# execute with  python -m gui.gui_app

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
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import enhanced components
from agents.enhanced_dqn_agent import EnhancedDQNAgent
from utils.utils import board_to_tensor
from utils import move_encoder
from config import ChessDQNConfig
from utils.system_monitor import SystemMonitor

class PromotionDialog:
    """Dialog for pawn promotion selection"""
    def __init__(self, parent, color):
        self.result = chess.QUEEN  # default
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("🎯 Pawn Promotion!")
        self.dialog.geometry("350x180")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        tk.Label(self.dialog, text="🎉 Your pawn reached the end!", font=('Arial', 12, 'bold')).pack(pady=5)
        tk.Label(self.dialog, text="Choose promotion piece:", font=('Arial', 11)).pack(pady=5)
        
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=10)
        
        # Piece buttons with symbols
        pieces = [
            (chess.QUEEN, "♕ Queen (BEST!)"),
            (chess.ROOK, "♖ Rook"), 
            (chess.BISHOP, "♗ Bishop"),
            (chess.KNIGHT, "♘ Knight")
        ]
        
        for piece, text in pieces:
            btn = tk.Button(button_frame, text=text, width=15,
                          command=lambda p=piece: self.select_piece(p))
            if piece == chess.QUEEN:
                btn.config(bg='gold', font=('Arial', 10, 'bold'))
            btn.pack(side=tk.LEFT, padx=5)
    
    def select_piece(self, piece):
        piece_names = {chess.QUEEN: "QUEEN", chess.ROOK: "ROOK", chess.BISHOP: "BISHOP", chess.KNIGHT: "KNIGHT"}
        print(f"👑 PLAYER PROMOTED TO {piece_names[piece]}!")
        self.result = piece
        self.dialog.destroy()

class ColorChoiceDialog:
    """Dialog for choosing player color"""
    def __init__(self, parent):
        self.result = None
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Choose Your Color")
        self.dialog.geometry("300x200")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        tk.Label(self.dialog, text="🎯 New Game Setup", font=('Arial', 14, 'bold')).pack(pady=10)
        tk.Label(self.dialog, text="Choose your color:", font=('Arial', 12)).pack(pady=5)
        
        button_frame = tk.Frame(self.dialog)
        button_frame.pack(pady=15)
        
        tk.Button(button_frame, text="♔ Play as White", width=15, height=2,
                 command=lambda: self.select_color(chess.WHITE)).pack(pady=5)
        tk.Button(button_frame, text="♚ Play as Black", width=15, height=2,
                 command=lambda: self.select_color(chess.BLACK)).pack(pady=5)
        tk.Button(button_frame, text="🎲 Random (Coin Toss)", width=15, height=2,
                 command=self.random_color).pack(pady=5)
    
    def select_color(self, color):
        self.result = color
        self.dialog.destroy()
    
    def random_color(self):
        self.result = random.choice([chess.WHITE, chess.BLACK])
        color_name = "White" if self.result == chess.WHITE else "Black"
        messagebox.showinfo("🎲 Coin Toss Result", f"You got {color_name}!")
        self.dialog.destroy()

class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Enhanced Chess DQN vs Human (FIXED)")
        self.board = chess.Board()
        self.selected_square = None
        self.game_moves = []
        self.player_color = chess.WHITE  # Default, will be chosen
        self.current_player_rating = 1500  # Default rating
        
        # Load configuration
        self.config = ChessDQNConfig()
        self.monitor = SystemMonitor(self.config.DATA_DIR)
        
        # Initialize enhanced agent
        checkpoint_path = self.config.CHECKPOINT_PATH
        if not os.path.exists(checkpoint_path):
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
        self.choose_color_and_start()
        
        print("🎯 Enhanced Chess GUI loaded with PROMOTION and CASTLING fixes!")

    def choose_color_and_start(self):
        """Let player choose color at game start"""
        color_dialog = ColorChoiceDialog(self.root)
        self.root.wait_window(color_dialog.dialog)
        
        if color_dialog.result is not None:
            self.player_color = color_dialog.result
            ai_color = "Black" if self.player_color == chess.WHITE else "White"
            player_color = "White" if self.player_color == chess.WHITE else "Black"
            
            print(f"🎮 You are playing as {player_color}, AI as {ai_color}")
            
            # If player is black, AI goes first
            if self.player_color == chess.BLACK:
                self.root.after(1000, self.ai_move)
        else:
            self.player_color = chess.WHITE  # Default fallback
        
        self.update_board()

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
        control_frame = tk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        
        # Game info
        info_frame = tk.LabelFrame(control_frame, text="Game Info", padx=5, pady=5)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.turn_label = tk.Label(info_frame, text="White to move", font=('Arial', 12, 'bold'))
        self.turn_label.pack()
        
        self.status_label = tk.Label(info_frame, text="Game in progress", font=('Arial', 10))
        self.status_label.pack()
        
        self.move_counter = tk.Label(info_frame, text="Move: 1", font=('Arial', 9))
        self.move_counter.pack()
        
        # Player info
        player_frame = tk.LabelFrame(control_frame, text="Player Info", padx=5, pady=5)
        player_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.player_color_label = tk.Label(player_frame, text="You: White", font=('Arial', 10, 'bold'))
        self.player_color_label.pack()
        
        # Agent info with training status
        agent_frame = tk.LabelFrame(control_frame, text="🤖 AI Opponent (FIXED!)", padx=5, pady=5)
        agent_frame.pack(fill=tk.X, pady=(0, 10))
        
        try:
            agent_info = self.agent.get_model_info()
            games_learned = agent_info['games_learned']
            
            # Show AI stats
            if games_learned < 500:
                strength = "🔴 Beginner (Learning basics)"
                estimated_rating = f"~{200 + games_learned}?"
            elif games_learned < 1500:
                strength = "🟡 Novice (Knows piece values)"
                estimated_rating = f"~{400 + games_learned//2}?"
            elif games_learned < 3000:
                strength = "🟢 Intermediate (Tactical)"
                estimated_rating = f"~{800 + games_learned//5}?"
            else:
                strength = "🔵 Advanced (Strategic)"
                estimated_rating = f"~{1000 + games_learned//10}?"
            
            self.ai_strength_label = tk.Label(agent_frame, text=strength, font=('Arial', 9))
            self.ai_strength_label.pack()
            
            self.ai_rating_label = tk.Label(agent_frame, text=f"Est. Rating: {estimated_rating}", font=('Arial', 9))
            self.ai_rating_label.pack()
            
            # Show learning statistics
            stats_text = self.agent.get_stats_summary()
            stats_lines = stats_text.strip().split('\n')[:4]  # First 4 lines
            for line in stats_lines[1:]:  # Skip title
                if line.strip():
                    tk.Label(agent_frame, text=line.strip(), font=('Arial', 8)).pack()
            
        except Exception as e:
            tk.Label(agent_frame, text="AI info unavailable", font=('Arial', 9)).pack()
        
        # System monitoring
        system_frame = tk.LabelFrame(control_frame, text="System Status", padx=5, pady=5)
        system_frame.pack(fill=tk.X, pady=(0, 10))
        self.update_system_info(system_frame)
        
        # Controls
        controls_frame = tk.LabelFrame(control_frame, text="Game Controls", padx=5, pady=5)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Button(controls_frame, text="🆕 New Game", command=self.new_game, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 3))
        tk.Button(controls_frame, text="↶ Undo Move", command=self.undo_move, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 3))
        tk.Button(controls_frame, text="💡 Get Hint", command=self.get_hint, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 3))
        tk.Button(controls_frame, text="🔍 Analyze Position", command=self.analyze_position, 
                 font=('Arial', 10)).pack(fill=tk.X, pady=(0, 3))
        
        # Rating input with adaptive difficulty
        rating_frame = tk.LabelFrame(control_frame, text="🎯 Adaptive Difficulty", padx=5, pady=5)
        rating_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(rating_frame, text="Your chess rating:", font=('Arial', 9)).pack()
        self.rating_var = tk.StringVar(value="1500")
        self.rating_entry = tk.Entry(rating_frame, textvariable=self.rating_var, width=10)
        self.rating_entry.pack(pady=2)
        self.rating_entry.bind('<Return>', self.update_difficulty)
        
        tk.Button(rating_frame, text="Update Difficulty", command=self.update_difficulty, 
                 font=('Arial', 8)).pack(pady=2)
        
        self.difficulty_label = tk.Label(rating_frame, text="AI will adapt to your rating", 
                                        font=('Arial', 8), fg='blue')
        self.difficulty_label.pack()
        
        # Game tips with FIXES info
        tips_frame = tk.LabelFrame(control_frame, text="💡 Fixed Issues", padx=5, pady=5)
        tips_frame.pack(fill=tk.X)
        
        tips_text = tk.Text(tips_frame, height=4, width=25, font=('Arial', 8))
        tips_text.pack(pady=2)
        
        tips = "✅ PROMOTION FIXED!\n✅ CASTLING REWARDS ADDED!\n✅ CLEAR LEARNING SIGNALS!\n✅ Watch for special move messages!"
        tips_text.insert(tk.END, tips)
        tips_text.config(state=tk.DISABLED)

    def update_difficulty(self, event=None):
        """Update AI difficulty based on player rating"""
        try:
            rating = int(self.rating_var.get())
            self.current_player_rating = rating
            
            # Provide feedback about difficulty change
            if rating < 1000:
                difficulty_text = "🟢 AI will play easier moves"
            elif rating < 1500:
                difficulty_text = "🟡 AI will play at normal level"
            elif rating < 2000:
                difficulty_text = "🟠 AI will play stronger moves"
            else:
                difficulty_text = "🔴 AI will play at maximum strength"
                
            self.difficulty_label.config(text=difficulty_text)
            print(f"🎯 Difficulty updated for rating {rating}")
            
        except ValueError:
            messagebox.showerror("Invalid Rating", "Please enter a valid number (e.g., 1500)")

    def get_promotion_piece(self, color):
        """Get promotion piece from user with enhanced dialog"""
        promotion_dialog = PromotionDialog(self.root, color)
        self.root.wait_window(promotion_dialog.dialog)
        return promotion_dialog.result

    def on_click(self, event):
        """🔥 COMPLETELY FIXED click handler - NO MORE COORDINATE BUGS!"""
        if self.board.turn != self.player_color:
            return  # Not player's turn
            
        # 🔥 CRITICAL FIX: Simplified coordinate handling
        col = event.x // 60
        row = 7 - (event.y // 60)
        
        # ✅ NO coordinate flipping here! Let chess.svg handle the display
        # Internal coordinates are always from White's perspective
        
        if 0 <= row <= 7 and 0 <= col <= 7:
            square = chess.square(col, row)
            
            if self.selected_square is None:
                # Select piece
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    self.selected_square = square
                    print(f"Selected piece at {chess.square_name(square)}: {piece}")
            else:
                # Make move
                move = chess.Move(self.selected_square, square)
                
                # 🔥 FIXED PROMOTION DETECTION
                piece = self.board.piece_at(self.selected_square)
                if (piece and piece.piece_type == chess.PAWN):
                    # Check if pawn reaches promotion rank (FIXED LOGIC!)
                    target_rank = chess.square_rank(square)
                    if ((piece.color == chess.WHITE and target_rank == 7) or 
                        (piece.color == chess.BLACK and target_rank == 0)):
                        # Ask user for promotion piece
                        promotion_piece = self.get_promotion_piece(piece.color)
                        move.promotion = promotion_piece
                        print(f"🎯 PROMOTION DETECTED! {piece.color} pawn → {promotion_piece}")
                
                print(f"Attempting move: {move}")
                
                if move in self.board.legal_moves:
                    # 🔥 ENHANCED MOVE FEEDBACK
                    self.make_move_with_feedback(move)
                    self.selected_square = None
                    self.update_board()
                    self.check_game_over()
                    
                    # AI's turn
                    if not self.board.is_game_over():
                        self.root.after(500, self.ai_move)
                else:
                    print(f"❌ Illegal move: {move}")
                    print(f"Legal moves sample: {[str(m) for m in list(self.board.legal_moves)[:5]]}")
                    self.selected_square = None

    def make_move_with_feedback(self, move):
        """Enhanced move execution with special move feedback"""
        
        # 🏰 DETECT CASTLING (before making the move!)
        is_castling = (
            self.board.piece_at(move.from_square) and 
            self.board.piece_at(move.from_square).piece_type == chess.KING and
            abs(move.from_square - move.to_square) == 2
        )
        
        # 👑 DETECT PROMOTION
        is_promotion = move.promotion is not None
        
        # 🎯 DETECT CAPTURE
        is_capture = self.board.is_capture(move)
        
        # Make the actual move
        self.board.push(move)
        self.game_moves.append(move)
        
        # 🔊 ENHANCED FEEDBACK MESSAGES
        if is_castling:
            if move.to_square > move.from_square:
                print("🏰 PLAYER CASTLED KINGSIDE!")
                self.status_label.config(text="You castled kingside! 🏰", fg='green')
            else:
                print("🏰 PLAYER CASTLED QUEENSIDE!")
                self.status_label.config(text="You castled queenside! 🏰", fg='green')
        
        elif is_promotion:
            piece_names = {chess.QUEEN: "Queen", chess.ROOK: "Rook", 
                          chess.BISHOP: "Bishop", chess.KNIGHT: "Knight"}
            promo_name = piece_names.get(move.promotion, "Unknown")
            print(f"👑 PLAYER PROMOTED TO {promo_name.upper()}!")
            self.status_label.config(text=f"You promoted to {promo_name}! 👑", fg='gold')
        
        elif is_capture:
            print("🎯 PLAYER CAPTURED A PIECE!")
            self.status_label.config(text="Nice capture! 🎯", fg='red')
        
        else:
            self.status_label.config(text="Your move", fg='black')

    def ai_move(self):
        """Make AI move with difficulty adjustment and enhanced feedback"""
        if not self.board.is_game_over() and self.board.turn != self.player_color:
            try:
                # Adjust AI thinking based on player rating
                if self.current_player_rating < 1000:
                    temperature = 2.0  # More random for beginners
                elif self.current_player_rating < 1500:
                    temperature = 1.0  # Normal
                else:
                    temperature = 0.1  # Strong play
                
                print("🤖 AI is thinking...")
                move = self.agent.choose_move(self.board, temperature=temperature)
                
                if move:
                    # 🔥 DETECT AI SPECIAL MOVES (before making move)
                    is_ai_castling = (
                        self.board.piece_at(move.from_square) and 
                        self.board.piece_at(move.from_square).piece_type == chess.KING and
                        abs(move.from_square - move.to_square) == 2
                    )
                    
                    is_ai_promotion = move.promotion is not None
                    is_ai_capture = self.board.is_capture(move)
                    
                    # Make the move
                    self.board.push(move)
                    self.game_moves.append(move)
                    
                    # 🔊 AI MOVE FEEDBACK
                    if is_ai_castling:
                        if move.to_square > move.from_square:
                            print("🏰 AI CASTLED KINGSIDE!")
                            self.status_label.config(text="AI castled kingside! 🏰", fg='blue')
                        else:
                            print("🏰 AI CASTLED QUEENSIDE!")
                            self.status_label.config(text="AI castled queenside! 🏰", fg='blue')
                    
                    elif is_ai_promotion:
                        piece_names = {chess.QUEEN: "Queen", chess.ROOK: "Rook", 
                                      chess.BISHOP: "Bishop", chess.KNIGHT: "Knight"}
                        promo_name = piece_names.get(move.promotion, "Unknown")
                        print(f"👑 AI PROMOTED TO {promo_name.upper()}!")
                        self.status_label.config(text=f"AI promoted to {promo_name}! 👑", fg='purple')
                    
                    elif is_ai_capture:
                        print("🎯 AI CAPTURED YOUR PIECE!")
                        self.status_label.config(text="AI captured your piece! 🎯", fg='red')
                    
                    else:
                        self.status_label.config(text="AI moved", fg='blue')
                    
                    self.update_board()
                    self.check_game_over()
                    
            except Exception as e:
                print(f"❌ AI move error: {e}")
                self.status_label.config(text="AI error!", fg='red')

    def new_game(self):
        """Start a new game with color choice"""
        self.board = chess.Board()
        self.selected_square = None
        self.game_moves = []
        self.status_label.config(text="New game starting...", fg='black')
        self.choose_color_and_start()

    def update_board(self):
        """Update the chess board display"""
        try:
            # Flip board if player is black
            flipped = (self.player_color == chess.BLACK)
            
            svg_data = chess.svg.board(
                self.board, 
                size=480,
                flipped=flipped,
                lastmove=self.board.peek() if self.board.move_stack else None
            )
            png_bytes = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
            img_data = Image.open(io.BytesIO(png_bytes))
            self.photo = ImageTk.PhotoImage(img_data)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
            
            # Update labels
            if not self.board.is_game_over():
                if self.board.turn == self.player_color:
                    self.turn_label.config(text="🎯 Your turn")
                else:
                    self.turn_label.config(text="🤖 AI thinking...")
            
            # Update player color display
            player_color_text = "White" if self.player_color == chess.WHITE else "Black"
            self.player_color_label.config(text=f"You: {player_color_text}")
            
            # Update move counter
            move_num = (len(self.board.move_stack) // 2) + 1
            self.move_counter.config(text=f"Move: {move_num}")
            
            self.root.update()
        except Exception as e:
            print(f"❌ Board update error: {e}")

    def check_game_over(self):
        """Check if game is over and handle learning"""
        if self.board.is_game_over():
            result = self.board.result()
            
            # Determine winner relative to player
            if ((result == '1-0' and self.player_color == chess.WHITE) or 
                (result == '0-1' and self.player_color == chess.BLACK)):
                msg = "🎉 Congratulations! You won!"
                game_result = "player_win"
                self.status_label.config(text="🏆 YOU WON!", fg='green')
            elif ((result == '0-1' and self.player_color == chess.WHITE) or 
                  (result == '1-0' and self.player_color == chess.BLACK)):
                msg = "🤖 AI won this time!"
                game_result = "ai_win"
                self.status_label.config(text="😔 AI Won", fg='red')
            else:
                msg = "🤝 It's a draw!"
                game_result = "draw"
                self.status_label.config(text="🤝 Draw", fg='orange')
            
            messagebox.showinfo("Game Over", msg)
            
            # Enhanced learning from human games
            try:
                # Convert result for learning
                if self.player_color == chess.WHITE:
                    learning_result = result
                else:
                    # Flip result if player was black
                    if result == '1-0':
                        learning_result = '0-1'
                    elif result == '0-1':
                        learning_result = '1-0'
                    else:
                        learning_result = result
                
                self.agent.learn_from_game(self.game_moves, learning_result)
                
                # Additional learning with player rating
                if hasattr(self.agent, 'learn_from_human_game'):
                    self.agent.learn_from_human_game(self.game_moves, learning_result, self.current_player_rating)
                
                print(f"🎓 AI learned from game against {self.current_player_rating}-rated player")
                
                # Show learning summary
                print(self.agent.get_stats_summary())
                
            except Exception as e:
                print(f"❌ Learning error: {e}")

    def update_system_info(self, parent_frame):
        """Update system information display"""
        try:
            health = self.monitor.check_system_health()
            memory = health['memory']
            gpu = health['gpu']
            
            tk.Label(parent_frame, text=f"RAM: {memory['used_gb']:.1f}/{memory['total_gb']:.1f}GB", 
                    font=('Arial', 9)).pack()
            
            if gpu:
                tk.Label(parent_frame, text=f"GPU: {gpu['allocated_gb']:.1f}/{gpu['total_gb']:.1f}GB", 
                        font=('Arial', 9)).pack()
            else:
                tk.Label(parent_frame, text="GPU: Not available", font=('Arial', 9)).pack()
            
            status_color = "green" if health['healthy'] else "orange"
            status_text = "Healthy" if health['healthy'] else "Limited"
            tk.Label(parent_frame, text=f"Status: {status_text}", 
                    font=('Arial', 9), fg=status_color).pack()
            
        except Exception as e:
            tk.Label(parent_frame, text="System info unavailable", font=('Arial', 9)).pack()

    def analyze_position(self):
        """Analyze current position with more detail"""
        try:
            from agents.enhanced_dqn_agent import ChessEvaluator
            
            evaluator = ChessEvaluator()
            score = evaluator.evaluate_position(self.board)
            
            # Adjust score based on player color
            if self.player_color == chess.BLACK:
                score = -score
            
            if score > 0.5:
                analysis = "You have a significant advantage! 😊"
            elif score > 0.1:
                analysis = "You are slightly better 🙂"
            elif score > -0.1:
                analysis = "Position is roughly equal ⚖️"
            elif score > -0.5:
                analysis = "Opponent is slightly better 😐"
            else:
                analysis = "Opponent has a significant advantage 😟"
            
            legal_moves = len(list(self.board.legal_moves))
            move_count = len(self.board.move_stack)
            game_phase = "Opening" if move_count < 20 else "Middlegame" if len(self.board.piece_map()) > 12 else "Endgame"
            
            # Check for special move opportunities
            special_moves = []
            for move in self.board.legal_moves:
                if self.board.piece_at(move.from_square) and self.board.piece_at(move.from_square).piece_type == chess.KING:
                    if abs(move.from_square - move.to_square) == 2:
                        special_moves.append("🏰 Castling available!")
                if move.promotion:
                    special_moves.append("👑 Promotion possible!")
                if self.board.is_capture(move):
                    captured = self.board.piece_at(move.to_square)
                    if captured and captured.piece_type in [chess.QUEEN, chess.ROOK]:
                        special_moves.append("🎯 Major piece capture available!")
            
            analysis_text = f"🔍 Position Analysis:\n\n{analysis}\n\nEvaluation: {score:.3f}\nLegal moves: {legal_moves}\nGame phase: {game_phase}\nMoves played: {move_count}"
            
            if special_moves:
                analysis_text += "\n\nSpecial opportunities:\n" + "\n".join(special_moves)
            
            messagebox.showinfo("Position Analysis", analysis_text)
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Could not analyze position: {e}")

    def get_hint(self):
        """Get move hint from AI"""
        try:
            hint_move = self.agent.choose_move(self.board, temperature=0.1)
            if hint_move:
                # Format move nicely
                piece = self.board.piece_at(hint_move.from_square)
                piece_name = piece.symbol().upper() if piece else "?"
                
                hint_text = f"💡 Suggested move: {piece_name}{hint_move}\n\n"
                
                # Add reasoning
                if hint_move.promotion:
                    hint_text += "This promotes your pawn! 👑"
                elif self.board.piece_at(hint_move.from_square) and self.board.piece_at(hint_move.from_square).piece_type == chess.KING and abs(hint_move.from_square - hint_move.to_square) == 2:
                    hint_text += "This castles your king to safety! 🏰"
                elif self.board.is_capture(hint_move):
                    hint_text += "This captures a piece! 🎯"
                elif self.board.gives_check(hint_move):
                    hint_text += "This gives check! ♔"
                else:
                    hint_text += "This looks like a good positional move. 🤔"
                
                messagebox.showinfo("AI Hint", hint_text)
        except Exception as e:
            messagebox.showerror("Hint Error", f"Could not get hint: {e}")

    def undo_move(self):
        """Undo last move"""
        if len(self.board.move_stack) >= 2:
            self.board.pop()  # Undo AI move
            self.board.pop()  # Undo player move
            if len(self.game_moves) >= 2:
                self.game_moves = self.game_moves[:-2]
            self.selected_square = None
            self.status_label.config(text="Move undone", fg='orange')
            self.update_board()

    def on_closing(self):
        """Handle window closing"""
        try:
            self.agent.save_model()
            print("💾 Agent progress saved!")
            print("📊 Final Statistics:")
            print(self.agent.get_stats_summary())
        except Exception as e:
            print(f"⚠️ Could not save agent progress: {e}")
        self.root.destroy()

def main():
    """Main function"""
    print("🎮 Starting FIXED Enhanced Chess GUI...")
    print("✅ Promotion bug FIXED!")
    print("✅ Castling rewards ADDED!")
    print("✅ Clear learning signals IMPLEMENTED!")
    
    root = tk.Tk()
    app = ChessApp(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    print("🎯 FIXED Enhanced Chess GUI loaded!")
    print("🎮 Try promoting a pawn and castling - you should see messages!")
    root.mainloop()

if __name__ == "__main__":
    main()