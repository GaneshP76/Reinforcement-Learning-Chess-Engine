import csv
import matplotlib.pyplot as plt

REWARD_LOG_PATH = "data/reward_log.csv"

episodes = []
rewards = []
epsilons = []

with open(REWARD_LOG_PATH, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        episodes.append(int(row["episode"]))
        rewards.append(float(row["reward"]))
        epsilons.append(float(row["epsilon"]))

# === Plot Total Reward Per Episode ===
plt.figure(figsize=(12, 6))
plt.plot(episodes, rewards, label="Total Reward per Episode", color="blue", alpha=0.7)
plt.axhline(0, color="black", linestyle="--", linewidth=0.5)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Chess DQN Agent - Reward per Episode")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Optional: Epsilon Decay ===
plt.figure(figsize=(12, 4))
plt.plot(episodes, epsilons, label="Epsilon", color="green")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay over Training")
plt.grid(True)
plt.tight_layout()
plt.show()
