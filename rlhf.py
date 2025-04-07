import torch

from reward_model import RewardModel




reward_model = RewardModel()

reward_model.load_state_dict(torch.load("reward_model.pt", weights_only=True))
print(reward_model)

# 3. PPO Helper Functions (Generalized Advantage Estimation)
def compute_advantages(rewards, values, next_values, gamma=0.99, lambda_=0.95):
    deltas = rewards + gamma * next_values - values
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lambda_ * gae
        advantages[t] = gae
    return advantages

# 4. PPO Objective with Clipping
def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.1):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss

# 5. PPO Training Loop
def ppo_train(policy_model, reward_model, optimizer, num_epochs=10, batch_size=8):
    for epoch in range(num_epochs):
        # Simulate an environment loop
        rewards = []
        log_probs = []
        values = []
        next_values = []
        advantages = []

        # Collect data: here we generate some random examples for simplicity
        for _ in range(batch_size):
            prompt = "What is the capital of France?"
            generated_text = policy_model.generate(prompt)

            # Compute the reward for generated text
            reward = reward_model(generated_text)

            # Simulate value function (in practice, you'd use a trained value network)
            value = torch.tensor(np.random.random(), dtype=torch.float32)  # Placeholder value function
            next_value = torch.tensor(np.random.random(), dtype=torch.float32)  # Placeholder for next state value

            rewards.append(reward)
            log_probs.append(torch.log(torch.tensor(np.random.rand(1))))  # Simulated log probabilities
            values.append(value)
            next_values.append(next_value)

        rewards = torch.stack(rewards)
        values = torch.stack(values)
        next_values = torch.stack(next_values)

        # Compute advantages
        advantages = compute_advantages(rewards, values, next_values)

        # Calculate loss using PPO objective with the clipped advantage
        loss = ppo_loss(torch.cat(log_probs), torch.cat(log_probs), advantages)

        # Backpropagate and update policy model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 6. Initialize models and optimizer
policy_model = PolicyModel()
reward_model = RewardModel()
optimizer = optim.Adam(policy_model.parameters(), lr=1e-5)

# Train the model
ppo_train(policy_model, reward_model, optimizer)

