import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib
import matplotlib.pyplot as plt


torch.manual_seed(1337)


#hyperparameters
batch_size = 32 #Number of training sequences to run in parallel
block_size = 8 #Max length of the sequence
learning_rate = 1e-3
max_iter = 10000
eval_interval = 300 # How often to evaluate the model
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200 #For evaluation, take average over multiple iterations
n_embed = 32

#____________


#Open the file
# Open the file to inspect the data 
with open("input.txt", "r", encoding = "utf-8") as f:
    text = f.read()

#Create a sorted list of all the unique characters in the input "text"
chars = sorted(list(set(text)))

#Vocab size is the total length of the unique tokens or characters (in this case)
vocab_size = len(chars)

#Convert char to int dict (encode)
stoi = {ch:i for i, ch in enumerate(chars)}

#Convert int to char dict (encode)
itos = {i:ch for i, ch in enumerate(chars)}


encode = lambda s : [stoi[i] for i in s]
decode = lambda l : "".join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype = torch.long) #saves the data in the form of a 1D tensor
#Get train and test splits
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


#Get batch
def get_batch(split):

    data = train_data if split=="train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) #Stack multiple examples in the form of a tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x,y

#torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X , Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        model.train()
        
    return out
            


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #Each token reads off the logits for the next token from a look up table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 
        self.pos_embedding_table = nn.Embedding(block_size, n_embed) #Each position from 0 to block-1 will also get a position embedding
        self.lm_head = nn.Linear(n_embed, vocab_size) #(B, T, vocab_size)

    def forward(self, idx, targets = None):

        B, T = idx.shape
        #idx and targets, both are (B,C) tensors. 
        #For each of these tokens, we will get a vocab_size "C" logit, and the output shape would be (B, T, C) 
        tok_emb = self.token_embedding_table(idx) #output shape: #(B, T, n_embed)
        pos_emb = self.pos_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb #(B, T, C) The batch dimenison gets added
        logits = self.lm_head(x) #output shape: #(B, T, vocab_size)

        if targets==None:
            loss = None

        else:
        #Pytorch expects the input for this loss function to be of shape (B, C, T), so we need to reshape our logit matrix
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)    # -log likelihood loss
        
        return logits, loss


    #Generate new text
    #This function takes in the entire context, even though the bigram model only needs the last index for th enext prediction. 
    #We do this so that this function can be kept constant for more complicated models.
    def generate(self, idx, max_new_tokens):

        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):

            #crop the idx to the last block_size tokens
            idx_cond = idx[ :, -block_size: ]

            #Get the predictions
            logits, loss = self(idx_cond)

            # Outputs (B, T, C) Pluck the last entry in the T dimension for the next prediction
            logits = logits[:,-1, :] #(B, C)

            #Apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1) #(B, C)
            
            #sample from distribution
            idx_next = torch.multinomial(probs, num_samples = 1) #(B, 1)
            
            #Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim = 1) #(B, T+1)
            
        return idx



model = BigramLanguageModel().to(device)

#Create a pytorch optimizer to train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) #Good learning rate for larger models can be 3e-4 etc. 



train_loss_curve = []
val_loss_curve = []
iteration = []


for iter in range(max_iter):

    #Every once in a while, calculate the train and validation loss 
    if iter % eval_interval == 0:
        loss = estimate_loss()
        print(f"Iteration number: {iter}, train_loss: {loss['train']:.4f}, val_loss: {loss['val']:.4f}")
        train_loss_curve.append(loss['train'])
        val_loss_curve.append(loss['val'])
        iteration.append(iter)

    #Sample a batch of data
    xb, yb = get_batch("train")
    
    #Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    
#Plot the curves
plt.figure()
plt.plot(iteration, train_loss_curve)
plt.plot(iteration, val_loss_curve)
plt.xlabel("steps")
plt.ylabel("cross entropy loss")
plt.title("Training and Validation loss for the Bigram Model")
plt.savefig("bigram_loss")
#Generate from the trained model
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(model.generate(context, max_new_tokens = 500)[0] .tolist()))

