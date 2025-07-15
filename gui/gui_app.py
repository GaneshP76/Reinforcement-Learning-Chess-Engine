import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import chess
import chess.svg
import io
import random
import cairosvg
from agents.dqn_agent import DQNModel
from utils.utils import board_to_tensor
from utils import move_encoder
import torch

# === Setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DQNModel().to(DEVICE)
checkpoint = torch.load("data/dqn_checkpoint.pth", map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess vs DQN Agent")
        self.board = chess.Board()
        self.selected_square = None

        self.canvas = tk.Canvas(self.root, width=480, height=480)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        self.update_board()

    def update_board(self):
        svg_data = chess.svg.board(self.board, size=480)
        png_bytes = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
        img_data = Image.open(io.BytesIO(png_bytes))
        self.photo = ImageTk.PhotoImage(img_data)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def ask_promotion_piece(self):
        piece_map = {
            "Queen": chess.QUEEN,
            "Rook": chess.ROOK,
            "Bishop": chess.BISHOP,
            "Knight": chess.KNIGHT
        }
        piece = simpledialog.askstring("Promotion", "Promote to (Queen, Rook, Bishop, Knight):")
        if piece:
            return piece_map.get(piece.capitalize(), chess.QUEEN)
        return chess.QUEEN

    def on_click(self, event):
        if self.board.is_game_over():
            return

        row = event.y // 60
        col = event.x // 60
        square = chess.square(col, 7 - row)

        if self.selected_square is None:
            if self.board.piece_at(square) and self.board.piece_at(square).color == chess.WHITE:
                self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)

            # Handle promotion
            if self.board.piece_at(self.selected_square).piece_type == chess.PAWN and chess.square_rank(square) in [0, 7]:
                move.promotion = self.ask_promotion_piece()

            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_square = None
                self.update_board()
                self.check_game_over()
                self.root.after(500, self.make_ai_move)
            else:
                self.selected_square = None

    def make_ai_move(self):
        if self.board.is_game_over():
            return

        legal_moves = list(self.board.legal_moves)
        state_tensor = torch.tensor(board_to_tensor(self.board)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            q_values = model(state_tensor)[0].cpu()

        legal_indices = []
        index_to_move = {}
        for m in legal_moves:
            idx = move_encoder.move_to_index(m)
            if idx is not None:
                legal_indices.append(idx)
                index_to_move[idx] = m

        if not legal_indices:
            move = random.choice(legal_moves)
        else:
            best_index = max(legal_indices, key=lambda i: q_values[i])
            move = index_to_move[best_index]

        self.board.push(move)
        self.update_board()
        self.check_game_over()

    def check_game_over(self):
        if self.board.is_game_over():
            result = self.board.result()
            if result == '1-0':
                msg = "You won!"
            elif result == '0-1':
                msg = "DQN won!"
            else:
                msg = "Draw!"
            messagebox.showinfo("Game Over", msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessApp(root)
    root.mainloop()
