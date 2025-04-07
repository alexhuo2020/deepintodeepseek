import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleLLM, SimpleLLMConfig
import numpy as np
from post_train import resize_token_embeddings, Tokenizer
from generate import generate
sft = False


tokenizer = Tokenizer()

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


class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.model = model
        self.tokenizer = Tokenizer()


    def forward(self, input_ids):
        logits = self.model(input_ids)
        return logits

    def generate(self, prompt, max_length=2):
        input_ids = self.tokenizer.encode(prompt)
        output_ids = generate(model, input_ids, max_length=max_length)
        generated_text = self.tokenizer.decode(output_ids[0])
        return generated_text

# 2. Define the Reward Model (simulating human feedback)
base_model = SimpleLLM(SimpleLLMConfig)
base_model.embedding = resize_token_embeddings(base_model.embedding, 2)
base_model.output_proj = nn.Linear(SimpleLLMConfig.embed_dim, SimpleLLMConfig.vocab_size + 2, bias=False)
base_model.output_proj.weight = model.embedding.weight  # re-tie after replacing the embedding

class SimpleLLMForClassification(nn.Module):
    def __init__(self):
        super(SimpleLLMForClassification, self).__init__()
        self.model = base_model
        self.tokenizer = Tokenizer()
        self.head = nn.Linear(SimpleLLMConfig.embed_dim, 1)


    def forward(self, input_ids):
        outputs = self.model(input_ids, last_hidden_state=True)
        pooled = outputs[:, -1]  # last token or use mean pooling
        reward = self.head(pooled)  # (batch_size, 1)
        return reward

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.model = SimpleLLMForClassification()
    
    def forward(self, input_ids_chosen, input_ids_rejected):
        r_chosen = self.model(input_ids_chosen)
        r_rejected = self.model(input_ids_rejected)
        return r_chosen, r_rejected
    
    def score(self, input_ids):
        with torch.no_grad():
            return self.base(input_ids).logits.squeeze()

def generate_sample(prompt, good_resp, bad_resp):
    return {
        "prompt": prompt,
        "chosen": good_resp,
        "rejected": bad_resp
    }


if __name__ == '__main__':
    import pandas as pd 
    df = pd.read_csv("human_preference.csv")

    df['chosen'] = df['prompt'] + ' ' + df['chosen']
    df['rejected'] = df['prompt'] + ' ' + df['rejected']

    df['chosen'] = df['chosen'].apply(tokenizer.encode)
    df['rejected'] = df['rejected'].apply(tokenizer.encode)
    df = df.drop(columns=['prompt'])  
    data = df.to_numpy()

    reward_model = RewardModel()
    # define the optimizer
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)

    # training
    losses = []
    for epoch in range(1000):
        for ds in data:
            x, y = ds
            x = torch.tensor(x)
            y = torch.tensor(y)
            r_chosen, r_rejected= reward_model(x.unsqueeze(0), y.unsqueeze(0))
            loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
    import matplotlib.pyplot as plt 
    plt.plot(losses)
    plt.show()

    for ds in data:
        x, y = ds
        x = torch.tensor(x)
        y = torch.tensor(y)
        print(reward_model(x.unsqueeze(0), y.unsqueeze(0)))

    torch.save(reward_model.state_dict(), "reward_model.pt")
