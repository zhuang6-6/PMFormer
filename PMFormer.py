
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Mamba import Mamba, MambaConfig

#  MoH Attention Definition
class MoHAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0.1, proj_drop=0.1,
                 shared_head=2, routed_head=2, head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * self.num_heads * self.head_dim, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.num_heads * self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.shared_head = shared_head
        self.routed_head = routed_head
        if self.routed_head > 0:
            self.wg = torch.nn.Linear(dim, num_heads - shared_head, bias=False)
            if self.shared_head > 0:
                self.wg_0 = torch.nn.Linear(dim, 2, bias=False)
        if self.shared_head > 1:
            self.wg_1 = torch.nn.Linear(dim, shared_head, bias=False)
    def forward(self, x):
        B, N, C = x.shape
        _x = x.reshape(B * N, C)
        if self.routed_head > 0:
            logits = self.wg(_x)
            gates = F.softmax(logits, dim=1)
            num_tokens, num_experts = gates.shape
            _, indices = torch.topk(gates, k=self.routed_head, dim=1)
            mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)
            if self.training:
                me = gates.mean(dim=0)
                ce = mask.float().mean(dim=0)
                l_aux = torch.mean(me * ce) * num_experts * num_experts
                MoHAttention.LOAD_BALANCING_LOSSES.append(l_aux)
            routed_head_gates = gates * mask
            denom_s = torch.sum(routed_head_gates, dim=1, keepdim=True)
            denom_s = torch.clamp(denom_s, min=torch.finfo(denom_s.dtype).eps)
            routed_head_gates /= denom_s
            routed_head_gates = routed_head_gates.reshape(B, N, -1) * self.routed_head
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        if self.routed_head > 0:
            x = x.transpose(1, 2)
            if self.shared_head > 0:
                shared_head_weight = self.wg_1(_x)
                shared_head_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
                weight_0 = self.wg_0(_x)
                weight_0 = F.softmax(weight_0, dim=1).reshape(B, N, 2) * 2
                shared_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 0], shared_head_gates)
                routed_head_gates = torch.einsum("bn,bne->bne", weight_0[:, :, 1], routed_head_gates)
                masked_gates = torch.cat([shared_head_gates, routed_head_gates], dim=2)
            else:
                masked_gates = routed_head_gates
            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)
        else:
            shared_head_weight = self.wg_1(_x)
            masked_gates = F.softmax(shared_head_weight, dim=1).reshape(B, N, -1) * self.shared_head
            x = x.transpose(1, 2)
            x = torch.einsum("bne,bned->bned", masked_gates, x)
            x = x.reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
#  Encoder Definition
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MoHAttention(d_model, num_heads=nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src):
        x = self.norm1(src + self.dropout(self.self_attn(src)))
        x = self.norm2(x + self.dropout(self.linear2(F.relu(self.linear1(x)))))
        return x

class Encoder(nn.Module):
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class ForwardEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, nhead, hidden_dim, dropout=0.1):
        super().__init__()
        self.node_embedding = nn.Linear(input_dim, d_model)
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer = Encoder(encoder_layer, num_layers)
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=1, d_state=hidden_dim))
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        h = self.node_embedding(x)
        h = self.transformer(h)
        h = self.mamba(h)
        out = self.edge_mlp(h)
        return 0.5 * (out[:, 0::2] + out[:, 1::2])

# DC power flow calculation
def DCpowerflow(Y_real: np.ndarray,
                      Y_imag: np.ndarray,
                      P: np.ndarray,
                      slack_idx: int = None):
    N = Y_real.shape[0]
    slack_idx = slack_idx if slack_idx is not None else N-1
    B_branch = -Y_imag
    B_bus = np.diag(B_branch.sum(axis=1)) - B_branch
    mask = np.arange(N) != slack_idx
    theta = np.zeros(N, float)
    theta[mask] = np.linalg.solve(B_bus[np.ix_(mask,mask)], P[mask])
    F = B_branch * (theta[:,None] - theta[None,:])
    return theta, F

#  Model Train
class Train:
    def __init__(self, model, optimizer, criterion, slack_idx=None, N=5):
        self.model = model
        self.opt = optimizer
        self.crit = criterion
        self.slack = slack_idx
        self.N = N
        self.M = int(self.N*(self.N-1)/2)
        self.seq_len = self.M * 2

    def train_loader(self, batches=3):
        for _ in range(batches):
            feats = torch.randn(1, self.seq_len, 3)
            P = np.random.randn(self.N)
            ui, uj = np.triu_indices(self.N, k=1)
            Yr = np.zeros((self.N, self.N)); Yi = np.zeros((self.N, self.N))
            vals_r = np.random.randn(self.M)
            vals_i = np.random.randn(self.M)
            for k,(i,j) in enumerate(zip(ui,uj)):
                Yr[i,j]=Yr[j,i]=vals_r[k]
                Yi[i,j]=Yi[j,i]=vals_i[k]
            Yr_t = torch.tensor(Yr, dtype=torch.float).unsqueeze(0)
            Yi_t = torch.tensor(Yi, dtype=torch.float).unsqueeze(0)
            yield feats, P, Yr_t, Yi_t

    def train(self, epochs=1):
        for epoch in range(epochs):
            for edge_feats, P, Y_real_t, Y_imag_t in self.train_loader():
                pred = self.model(edge_feats)
                arr = pred[0].detach().cpu().numpy()
                # rebuild network
                ui, uj = np.triu_indices(self.N, k=1)
                Yr_p = np.zeros((self.N, self.N)); Yi_p = np.zeros((self.N, self.N))
                for k,(i,j) in enumerate(zip(ui,uj)):
                    Yr_p[i,j]=Yr_p[j,i]=arr[k,0]
                    Yi_p[i,j]=Yi_p[j,i]=arr[k,1]
                theta_p, F_p = DCpowerflow(Yr_p, Yi_p, P, self.slack)
                theta_t, F_t = DCpowerflow(Y_real_t[0].numpy(), Y_imag_t[0].numpy(), P, self.slack)
                loss = self.crit(torch.tensor(F_p), torch.tensor(F_t))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.4f}")
        print(" prediction :", F_p)
        print(" true :", F_t)


if __name__ == '__main__':
    N = 5
    model = ForwardEncoder(3,128,2,4,64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    trainer = Train(model, opt, crit, slack_idx=N-1, N=N)
    trainer.train(epochs=100)
