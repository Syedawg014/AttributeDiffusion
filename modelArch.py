import torch
import torch.nn as nn


def get_time_embedding(time_steps, temb_dim):
    factor = 10000 ** (
        (torch.arange(start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    return torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)


class MLPBlock(nn.Module):
    def __init__(self, hidden_dim, t_emb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.t_emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, hidden_dim)
        )

    def forward(self, x, t_emb):
        out = self.fc(self.norm(x))
        out = out + self.t_emb_layer(t_emb)
        return out + x


class MLP(nn.Module):
    def __init__(self, in_dim=18 * 512, hidden_dim=2048, t_emb_dim=128, num_blocks=8):
        super().__init__()
        self.t_emb_dim = t_emb_dim

        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )

        self.multiplier = nn.Linear(t_emb_dim, 1)
        self.in_layer = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([MLPBlock(hidden_dim, t_emb_dim) for i in range(num_blocks)])
        self.out_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_dim)
        )

    def forward(self, x, t):
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)

        h = self.in_layer(x)
        for block in self.blocks:
            h = block(h, t_emb)
        noise_pred = self.out_layer(h) + self.multiplier(t_emb) * x 
        return noise_pred
