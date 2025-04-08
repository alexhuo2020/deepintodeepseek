import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SimpleLLM, SimpleLLMConfig
from reward_model import RewardModel
from post_train import resize_token_embeddings, Tokenizer
from generate import generate
import numpy as np

# whether to use the supervised fine tuned model or not
sft = True

# 1. Define the Policy Model (pretrained LLM w/o fine tune)
model = SimpleLLM(SimpleLLMConfig)
if not sft:
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.embedding = resize_token_embeddings(model.embedding, 2)
    model.output_proj = nn.Linear(SimpleLLMConfig.embed_dim, SimpleLLMConfig.vocab_size + 2, bias=False)
    model.output_proj.weight = model.embedding.weight  # re-tie after replacing the embedding
else:
    model.embedding = resize_token_embeddings(model.embedding, 2)
    model.output_proj = nn.Linear(SimpleLLMConfig.embed_dim, SimpleLLMConfig.vocab_size + 2, bias=False)
    model.output_proj.weight = model.embedding.weight  # re-tie after replacing the embedding
    model.load_state_dict(torch.load("model_sft.pt", weights_only=True))
tokenizer = Tokenizer()

class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.model = model
        self.tokenizer = Tokenizer()
        self.critic_head = nn.Linear(SimpleLLMConfig.embed_dim, 1)


    def forward(self, input_ids):
        logits = self.model(input_ids)
        h = self.model(input_ids, last_hidden_state=True)
        value = self.critic_head(h)
        return logits, value

    def generate(self, prompt, max_length=1):
        input_ids = torch.tensor([self.tokenizer.encode(prompt)])
        output_ids = generate(self.model, input_ids, max_length=max_length)
        generated_text = self.tokenizer.decode(output_ids[0])
        return generated_text


# 2. Define the Reward Model, see reward_model.py for how to train this model 
reward_model = RewardModel()
reward_model.load_state_dict(torch.load("reward_model.pt", weights_only=True))

# 3. PPO Helper Functions (Generalized Advantage Estimation)
def compute_advantages(rewards, values, next_values, gamma=0.99, lambda_=0.95):
    deltas = rewards + gamma * next_values - values
    advantages = torch.zeros_like(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        gae = deltas[t] + gamma * lambda_ * gae
        advantages[t] = gae
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

def get_logprob(logits, input_ids, response_mask):
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (batch, seq_len)
    response_token_log_probs = token_log_probs * response_mask  # (batch, seq_len)
    log_probs_sum = response_token_log_probs.sum(dim=-1)  # (batch,)

    return log_probs_sum

# 4. PPO Objective with Clipping
def ppo_loss(old_log_probs, new_log_probs, advantages, epsilon=0.1):
    ratio = torch.exp(new_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss


# 5. PPO Training Loop
def ppo_train(policy_model, reward_model, ref_model, optimizer, num_epochs=10, batch_size=2):
    for epoch in range(num_epochs):
        # Simulate an environment loop
        rewards = []
        log_probs = []
        values = [torch.tensor(0.)]
        next_values = []
        advantages = []
        old_log_probs = []
        # Collect data: here we generate some random examples for simplicity
        for _ in range(batch_size):
            prompt = "human do I like coffee . system"
            generated_text = policy_model.generate(prompt)
    
            # Compute the reward for generated text
            input_ids = torch.tensor([tokenizer.encode(prompt + ' ' + generated_text)])
            with torch.no_grad():
                reward = reward_model.score(input_ids)

            logits_old, _ = ref_model(input_ids)
            response_mask = torch.zeros_like(input_ids)
            response_mask[:, len(prompt):] = 1  
            
            logprob_old = get_logprob(logits_old, input_ids, response_mask)

            logits_new, value = policy_model(input_ids)
            
            logprob_new = get_logprob(logits_new, input_ids, response_mask)
            value = value[0,-1].squeeze()


            rewards.append(reward)
            old_log_probs.append(logprob_old)  # Simulated log probabilities
            log_probs.append(logprob_new)
            values.append(value)
            next_values.append(value)
        values.pop()

        rewards = torch.stack(rewards)
        values = torch.stack(values)
        next_values = torch.stack(next_values)

        # Compute advantages
        advantages, returns = compute_advantages(rewards, values, next_values)

        # Calculate loss using PPO objective with the clipped advantage
        loss = ppo_loss(torch.cat(old_log_probs), torch.cat(log_probs), advantages)

        # Backpropagate and update policy model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# 6. Initialize models and optimizer
policy_model = PolicyModel()
reward_model = RewardModel()
ref_model = PolicyModel()
ref_model.eval()
reward_model.eval()
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)

# Train the model
ppo_train(policy_model, reward_model, ref_model, optimizer)

