import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SimpleLLM, SimpleLLMConfig
from reward_model import RewardModel
from post_train import resize_token_embeddings, Tokenizer
from generate import generate
import numpy as np
torch.autograd.set_detect_anomaly(True)
# whether to use the supervised fine tuned model or not
sft = False #True

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
def compute_advantages(rewards, values, gamma=0.99, lambda_=0.95):
    advantages, gae = [], 0
    values = torch.cat((values, torch.tensor([0])))
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
    returns = torch.stack([adv + val for adv, val in zip(advantages, values[:-1])])
    return torch.stack(advantages), returns

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
def ppo_train(policy_model, reward_model, ref_model, optimizer, num_iters=4, num_epochs=100, batch_size=2):
    for iter in range(num_iters):
        # Simulate an environment loop
        rewards = []
        log_probs = []
        values = []
        advantages = []
        old_log_probs = []
        responses = []
        # Collect data: here we generate some random examples for simplicity
        for _ in range(batch_size):
            prompt = "human do I like coffee . system"
            with torch.no_grad():
                generated_text = policy_model.generate(prompt)
    
            # Compute the reward for generated text
            input_ids = torch.tensor([tokenizer.encode(prompt + ' ' + generated_text)])
            responses.append(input_ids)
            with torch.no_grad():
                reward = reward_model.score(input_ids)

                logits_old, _ = ref_model(input_ids)
            response_mask = torch.zeros_like(input_ids)
            response_mask[:, len(prompt):] = 1  
            logprob_old = get_logprob(logits_old, input_ids, response_mask)
            with torch.no_grad():
                _, value = policy_model(input_ids)
            
            value = value[0,-1].squeeze()


            rewards.append(reward)
            old_log_probs.append(logprob_old)  # Simulated log probabilities
            values.append(value.detach())
        

        rewards = torch.stack(rewards)
        values = torch.stack(values)

        # Compute advantages
        advantages, returns = compute_advantages(rewards, values)

        for epoch in range(num_epochs):
            log_probs = []
            value_preds = []
            for i in range(len(responses)):
                input_ids = responses[i]
                logits_new, value = policy_model(input_ids)
                response_mask = torch.zeros_like(input_ids)
                response_mask[:, len(prompt):] = 1  
                logprob_new = get_logprob(logits_new, input_ids, response_mask)
                log_probs.append(logprob_new)
                # value = value[0,-1].squeeze()
                # value_preds.append(value)
                # value_preds = torch.stack(value_preds)
            # Calculate loss using PPO objective with the clipped advantage
                value_loss = F.mse_loss(value,returns[i])
                loss = ppo_loss(old_log_probs[i], logprob_new, advantages[i]) + 0.5 * value_loss

                # Backpropagate and update policy model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        ref_model.load_state_dict(policy_model.state_dict())

# 6. Initialize models and optimizer
policy_model = PolicyModel()
reward_model = RewardModel()
ref_model = PolicyModel()
ref_model.eval()
reward_model.eval()
optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)

# Train the model
ppo_train(policy_model, reward_model, ref_model, optimizer)

# evaluation
X = ["human do I like coffee . system"]
for x in X:
    for _ in range(10):
        generated_text = policy_model.generate(x)
        print(generated_text)


torch.save(policy_model.model.state_dict(), "model_rlhf.pt")
