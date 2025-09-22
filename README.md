# ♟️ Reinforcement Learning Chess Engine 

A next-generation reinforcement learning system that learns to play chess through self-play.  
This project combines an **enhanced Deep Q-Network (DQN)** with **residual blocks, multi-head attention, and joint policy/value heads**, wrapped in a resource-aware training workflow, Tkinter GUI, and advanced pipelines for master-game fine-tuning and NNUE experimentation.  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue" />
  <img src="https://img.shields.io/badge/PyTorch-DQN-red" />
  <img src="https://img.shields.io/badge/GUI-Tkinter-green" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

---

## 📑 Table of Contents  
- [Overview](#overview)  
- [Deep Learning Backbone](#deep-learning-backbone)  
- [Features](#features)  
- [Project Structure](#project-structure)  
- [Getting Started](#getting-started)  
- [Training & Evaluation](#training--evaluation)  
- [Advanced Workflows](#advanced-workflows)  
- [GUI Experience](#gui-experience)  
- [Data & Configuration](#data--configuration)  
- [Diagnostics & QA](#diagnostics--qa)  
- [Command Reference](#command-reference)  
- [Next Steps](#next-steps)  
- [License](#license)  

---

## 🔍 Overview  
The trainer bootstraps everything in one place: configuration, monitoring, models, gameplay, learning, evaluation, and checkpointing.  
- **Hardware-aware config** prepares data directories, estimates runtime, and adjusts batch sizes automatically.  
- **Continuous resource tracking** prevents crashes by monitoring RAM, GPU, and disk.  
- **Safe checkpointing** ensures long runs survive interruptions, with stale files purged automatically.  

---

## 🧠 Deep Learning Backbone  
At the core, this project is a **deep reinforcement learning system** powered by **Deep Q-Networks (DQN)** and extended with:  
- **Residual blocks** for stable deep architectures.  
- **Multi-head attention** to capture global chessboard context.  
- **Joint policy & value heads** to balance move selection with long-term game outcomes.  

This deep learning foundation allows the agent to not only approximate raw move values but also to generalize across diverse game states and playstyles.  

---

## 🚀 Features  
- **Enhanced DQN** with deep convolutions, residual blocks, attention, and 4,672-action Q-outputs.  
- **Reward shaping** for special moves, repetition tracking, and human-game fine-tuning.  
- **Adaptive pipeline**: opening-book guidance, prioritized replay, curriculum learning, and eval-driven difficulty scaling.  
- **System-aware orchestration**: GPU checks, cleanup, safe checkpointing.  
- **Desktop GUI**: promotion dialogs, color choice, AI status panel, and live monitoring.  
- **Diagnostics suite**: validates imports, encoders, replay buffers, hardware, and configs before big runs.  
- **Advanced data flows**: master-game PGN ingestion, NNUE workflows, rating-weighted fine-tuning.  

---

## 📂 Project Structure  
```
agents/        # DQN model, evaluator, reward shaping
training/      # Trainer, NNUE workflows, master-game integration
gui/           # Tkinter GUI for human-vs-AI play
evaluate/      # Benchmark scripts vs random / baseline opponents
utils/         # System monitoring & cleanup utilities
config.py      # Central configuration & resource estimator
test_enhanced.py / comprehensive_test.py  # Diagnostics
requirements.txt  # Dependencies
```

---

## ⚡ Getting Started  

1. **Set up environment**  
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. **Run smoke test**  
```bash
python test_enhanced.py
```

3. **Full checklist (hardware/config validation)**  
```bash
python comprehensive_test.py
```

4. **Train the agent** (interrupt-safe)  
```bash
python -m training.enhanced_train
```

5. **Play via GUI**  
```bash
python -m gui.gui_app
```

6. **Evaluate performance**  
```bash
python -m evaluate.test_vs_random
```

---

## 🏋️ Training & Evaluation  
- **Replay buffer**: prioritized, continuous learner.  
- **Curriculum**: dynamic difficulty scaling.  
- **Evaluation cycles**: agent vs random play, rating estimation, checkpoint promotion.  
- **Logs**: rewards, blunders, memory usage, disk growth.  

---

## 🧠 Advanced Workflows  
- **Master-game integration**: curated PGN ingestion, rating/outcome weighting, quick-improvement vs GM presets.  
- **NNUE workflow**: Stockfish-style evaluator training from self-play games, with reproducible test scripts.  

---

## 🎨 GUI Experience  
- Tkinter-based interface with:  
  - Promotion dialogs (Queen, Rook, Bishop, Knight)  
  - Random coin-toss color assignment  
  - Move counters and timers  
  - AI status + resource monitoring  
  - Auto-checkpoint detection  

---

## ⚙️ Data & Configuration  
- **`ChessDQNConfig`** centralizes hyperparameters, schedules, replay sizes, checkpoints, and logs.  
- Prints device info, estimates runtimes, checks GPU RAM, and adjusts automatically.  
- Auto-creates directories: `/data`, `/models`, `/logs`, `/backups`.  

---

## ✅ Diagnostics & QA  
- **Smoke test** (`test_enhanced.py`) → ensures models & encoders build correctly.  
- **Comprehensive test** (`comprehensive_test.py`) → hardware checks, config interrogation, forward passes, and move encoders.  

---

## 📜 Command Reference  
```bash
python -m training.enhanced_train           # Train agent
python -m gui.gui_app                       # GUI play
python -m evaluate.test_vs_random           # Benchmark vs random
python -m training.master_games_integration # Master-game fine-tuning
python -m training.nnue_train               # NNUE workflow
python test_enhanced.py                     # Smoke test
python comprehensive_test.py                # Full-system test
```

---

## 🔮 Next Steps  
Planned improvements and research directions include:  
- **NNUE Integration**: Training and deploying a Stockfish-inspired NNUE engine to combine evaluation precision with reinforcement learning exploration.  
- **Multi-playstyle Models**: Creating specialized agents (aggressive, defensive, positional) and allowing users to select opponent style.  
- **Cloud Training Pipelines**: Scaling experiments to distributed environments for faster convergence.  
- **Hybrid Evaluation**: Merging policy/value heads with NNUE nets for a best-of-both-worlds engine.  

---

## 📄 License  
MIT License – free to use, modify, and share with attribution.  
