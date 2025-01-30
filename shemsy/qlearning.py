import random

# Initialize Q-table (5 devices, 2 actions: ON/OFF)
Q = np.zeros((5, 2))

# Reward function (customized for the problem)
def get_reward(device, action):
    # Example: reward = - energy usage for turning on the device, 0 for turning off
    return - device_usage[device][0] if action == 1 else 0

# Q-Learning parameters
learning_rate = 0.1
discount_factor = 0.9
epochs = 1000

# Q-learning loop
for _ in range(epochs):
    for device in range(5):
        action = random.choice([0, 1])  # Randomly choose action (ON/OFF)
        reward = get_reward(device, action)
        Q[device, action] = Q[device, action] + learning_rate * (reward + discount_factor * np.max(Q[device]) - Q[device, action])

# Display Q-table
print("Q-table after learning:", Q)
