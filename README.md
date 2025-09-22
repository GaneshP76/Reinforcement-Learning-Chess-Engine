GitHub README
Self-Learning Chess AI
Self-Learning Chess AI pairs an enhanced Deep Q-Network architecture featuring residual blocks, multi-head attention, and joint policy/value heads with a training workflow that monitors system health, manages prioritized replay, and continuously evaluates performance. A Tkinter desktop GUI, evaluation utilities, and advanced data pipelines for master-game fine-tuning and NNUE experimentation round out the toolkit for both learning and play.

Table of Contents
Overview

Features

Project Structure

Getting Started

Training & Evaluation

Advanced Workflows

GUI Experience

Data & Configuration

Diagnostics & QA

Command Reference

Overview
The enhanced trainer bootstraps configuration, instantiates system monitoring, initializes neural models, and orchestrates gameplay, learning, evaluation, and checkpointing in one place. Central configuration detects hardware, prepares data directories, prints schedules, and estimates resource needs so you can size experiments appropriately. Continuous resource tracking warns about RAM, GPU, or disk pressure and can purge stale checkpoints automatically to keep long runs stable.

Features
Enhanced DQN architecture with deeper convolutions, residual blocks, multi-head attention, and 4,672-action Q-value outputs for rich chess understanding.

Reward shaping and continuous learning that detect special moves, track repetition, and fine-tune from recent human games to reinforce high-signal experiences.

Adaptive training pipeline combining opening-book guidance, prioritized replay, curriculum logic, game analysis, and evaluation-driven difficulty adjustments.

Resource-aware orchestration with periodic system checks, disk cleanup, and checkpoint management during the training loop.

Desktop GUI featuring promotion dialogs, color selection, adaptive hints, and live status readouts powered by the enhanced agent.

Diagnostics and smoke tests to validate imports, hardware, model creation, move encoding, and trainer setup before long experiments.

Advanced data pipelines for master-game ingestion, preset training plans, NNUE self-play workflows, and evaluation against baseline opponents.

Project Structure
agents/ – Enhanced DQN model, evaluator, and agent logic with reward shaping fixes.

training/ – Enhanced trainer, NNUE workflow, master-game integration, and shared training components.

gui/ – Tkinter application for human-vs-AI play with promotion and castling fixes.

evaluate/ – Scripts for benchmarking agents against random play and summarizing win rates.

utils/ – System monitoring utilities for RAM, GPU, disk tracking, and cleanup helpers.

config.py – Hardware-aware configuration with directory setup, schedules, and resource estimators.

test_enhanced.py, comprehensive_test.py – Modular smoke tests and full-system validation steps.

requirements.txt – Python dependency list for training, GUI, and tooling.

Getting Started
Set up your Python environment. Create a virtual environment and install the pinned dependencies from requirements.txt:

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

Run the enhanced smoke test to confirm models, encoders, and the trainer instantiate correctly:

python test_enhanced.py

Optionally run the comprehensive checklist for hardware probing, configuration inspection, and forward-pass tests:

python comprehensive_test.py

Start training the enhanced agent (interrupt-safe with automatic checkpointing):

python -m training.enhanced_train

Play against the agent using the Tkinter GUI, which loads the latest checkpoint and shows live stats:

python -m gui.gui_app

Evaluate progress by playing hundreds of games versus a random opponent:

python -m evaluate.test_vs_random

Explore advanced workflows such as master-game fine-tuning or NNUE training when you’re ready:

python -m training.master_games_integration
python -m training.nnue_train

Training & Evaluation
The enhanced trainer seeds an opening book, prioritized replay buffer, continuous learner, adaptive difficulty controller, and model evaluator before entering the main loop. Each episode collects experience, updates the network, logs rewards, analyzes blunders, and periodically saves checkpoints while tracking storage growth. Evaluation cycles pit the agent against random play, adjust curriculum difficulty, estimate rating, and preserve the best-performing weights. Dedicated evaluation scripts provide reproducible win/draw/loss summaries over large game batches.

Advanced Workflows
Master-game integration downloads or prioritizes curated PGNs, filters by rating, applies temporal/outcome weighting, and offers presets tailored to different goals (quick improvement, grandmaster focus, mobile deployment). The NNUE workflow guides you through preparing training data, initializing a Stockfish-style evaluator, and testing its strength after self-play learning.

GUI Experience
The Tkinter interface provides promotion dialogs, color selection (including random coin toss), move counters, AI status panels, and system resource readouts tied into the shared configuration and monitoring utilities. It automatically locates the best available checkpoint and can trigger AI moves when the human chooses black.

Data & Configuration
ChessDQNConfig centralizes hyperparameters, exploration schedules, replay sizes, checkpoint paths, and logging destinations while printing device information, training schedules, and estimated runtimes. It also reports memory requirements, checks system compatibility, and can auto-adjust batch size when GPU memory is limited. All required directories (data, backups, logs, models) are created on initialization.

Diagnostics & QA
test_enhanced.py validates imports, model creation, tensor shapes, move encoders, replay buffers, and trainer setup for quick assurance before long runs. comprehensive_test.py extends this with hardware detection, configuration interrogation, forward passes, and move encoding checks, offering an interactive checklist for new environments.

Command Reference
python -m training.enhanced_train           # Train the enhanced DQN agent
python -m gui.gui_app                       # Launch the Tkinter GUI
python -m evaluate.test_vs_random           # Benchmark vs random play
python -m training.master_games_integration # Guided master-game fine-tuning
python -m training.nnue_train               # NNUE training workflow
python test_enhanced.py                     # Enhanced DQN smoke test
python comprehensive_test.py                # Full-system validation

Testing
⚠️ python test_enhanced.py (not run; repository review only)

⚠️ python comprehensive_test.py (not run; repository review only)

⚠️ python -m evaluate.test_vs_random (not run; repository review only)
