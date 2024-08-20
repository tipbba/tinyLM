import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparam
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters =200
n_embd = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open("gpt-2/input.txt", 'r', encoding="utf-8") as f:
    text = f.read()


chars = sorted(set(text))

vocab_size = len(chars)

# tokenizer
# stoi 生成了一个字典，key 为字符，而 value 为对应的索引值
# encoder 中，s为输入的字符串，c是字符串中的字符，通过在stoi中查找字符对应的数值生成列表
stoi = { ch:i for i, ch in enumerate(chars) } # key = ch, value = i
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s : [stoi[c] for c in s] # encoder take a string, output a list integers
decode = lambda l : "".join([itos[i] for i in l]) # decoder take a list of integers, output a string

# e = encoder("hi there")
# print(e)
# print(decoder(e))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:100])

n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]


torch.manual_seed(1337)
batch_size = 4 # how many independent sequence will we process in parallel
block_size = 8 # what is the maximum context length for prediction

# torch.stack用于将多个相同形状的 tensor 合并为一个新的 tensor

# 原视频40分钟，有详细解释
@torch.no_grad()
def estimate_loss():
    out ={}
    model.eval()
    for spilt in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(spilt)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[spilt] = losses.mean()
    model.train()
    return out


def get_batch(split):
    """generate a small batch of data of inputs x and target y"""
    data = train_data if split == "train" else val_data
    ix   = torch.randint(len(data) - block_size, (batch_size,)) 
    # 我们的数据长度为 len(data)，现在需要从中随机选择长度不超过block_size的子序列，
    # 为了保证序列完全包含于data中，序列的起点位置不能大于 len(data) - block_size
    # ix为长度为batch_size的一维张量，元素为随机选中的序列起点。
    x    = torch.stack([data[i:i + block_size] for i in ix])
    y    = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    return x, y


class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(SelfAttention(head_size) for i in range(num_heads))
        self.proj = nn.Linear(num_heads * head_size, n_embd) # 回到残差连接路径的投影

    def forward(self, x):
        ret = torch.cat([h(x) for h in self.heads], dim=-1)
        ret = self.proj(ret)
        return ret

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        # 4这个数字来源自 attention is all you need
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), 
                                 nn.ReLU(),
                                 nn.Linear(4*n_embd, n_embd)) # 残差连接变换
        
    def forward(self,x):
        return self.net(x)
    
class block(nn.Module):
    def __init__(self,n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x)) # 残差连接，使模型对残差进行训练，而不是原始输入
        x = x + self.ffwd(self.ln2(x))
        return x        

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            block(n_embd, n_head=4),
            block(n_embd, n_head=4),
            block(n_embd, n_head=4),
            block(n_embd, n_head=4)
        )

        self.lm_head = nn.Linear(n_embd, vocab_size) # 提供从嵌入还原到vocab_size的映射

    # forward函数，首先利用token_embedding_table函数将输入数据进行词嵌入，以当前代码为例：
    # 假设 nn.Embedding 的第二个参数为65，即一个token会被映射到一个65维的向量空间上，
    # token_embedding_table(idx), idx 为 4*8 矩阵，即一个batch中有 4 个句子，每个句子8个token，
    # 从而 token_embedding_table(idx) 的结果为 4*8*65的张量。
    # idx 是其实是xb，xb是由原始数据经过token化后的序列，组成的一个batch。
    # 事实上，第一个参数是词汇表的大小，整个函数的作用是建立一个查找表，即给出一个token，就可以返回对应的词嵌入向量
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx) 
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd # (B, T, C)
        x = self.blocks(x)
        logits = self.lm_head(x)
        # print(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape 
            logits  = logits.view(B*T, C) # 转换维度是为例输入cross_entropy函数
            targets = targets.view(B*T)
            loss    = F.cross_entropy(logits, targets)
        return logits, loss
    # generate是forward函数的调用者，具体为 self(idx) 调用了forward函数
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus on the last time step
            logits = logits[:, -1, :] # become B,C
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



model = LanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses["val"]:.4f}")

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

