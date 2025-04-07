import torch
import torch.nn.functional as F
from model import SimpleLLM, SimpleLLMConfig
from dataclasses import dataclass
from transformer import Tokenizer
from generate import generate
@dataclass
class TrainConfig:
    lr: float = 1e-3
    max_epochs: int = 3000


# preparing data
data = ["I like coffee .", "I like tea .", "You like tea .", "You do not like coffee ."]
tokenizer = Tokenizer()

train_ids = []
for s in data:
    train_ids.append(tokenizer.encode(s))

x = [torch.tensor(d[:-1]) for d in train_ids]
y = [torch.tensor(d[1:]) for d in train_ids]
train_ids = list(zip(x,y))

# define the model
model = SimpleLLM(SimpleLLMConfig)

# define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.lr)

# training
losses = []
for epoch in range(TrainConfig.max_epochs):
    for x, y in train_ids:
        _, loss = model(x.unsqueeze(0), y.unsqueeze(0))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

# loss curve
import matplotlib.pyplot as plt 
plt.plot(losses)
plt.show()
plt.savefig("loss.png")


# save model
torch.save(model.state_dict(), "model.pt")

# evaluation
X = ["do I like coffee .", "do You like coffee ."]
for x in X:
    x = torch.tensor([tokenizer.encode(x)])

    logits = model(x)
    print(F.softmax(logits, -1))
    for _ in range(10):
        y = generate(model, x, 2)
        for yy in y:
            print(tokenizer.decode(yy))



