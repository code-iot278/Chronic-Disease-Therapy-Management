# =========================================================
# FULL ONE-CELL IMPLEMENTATION
# Federated Res-HyperTransformerNet + PMARL
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================================================
# 1. Residual Depthwise Block
# =========================================================
class ResDWBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.expand = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=out_channels, bias=False
        )
        self.project = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        self.skip = None
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        out = self.expand(x)
        out = self.dwconv(out)
        out = self.project(out)
        out = self.bn(out)

        if self.skip is not None:
            identity = self.skip(identity)

        return F.relu(out + identity)

# =========================================================
# 2. Hyper-Transformer Block
# =========================================================
class HyperTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads,
                                         dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x

# =========================================================
# 3. Res-HyperTransformerNet
# =========================================================
class ResHyperTransformerNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            ResDWBlock(in_channels, 32),
            ResDWBlock(32, 64),
            ResDWBlock(64, 128)
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(128, 128)

        self.transformer = HyperTransformerBlock(
            embed_dim=128,
            num_heads=4,
            ffn_dim=256
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.embedding(x)
        x = x.unsqueeze(1)             # sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)

# =========================================================
# 4. Federated Averaging (FedAvg)
# =========================================================
def fed_avg(global_model, local_models):
    global_dict = global_model.state_dict()

    for key in global_dict.keys():
        global_dict[key] = torch.mean(
            torch.stack([model.state_dict()[key].float()
                         for model in local_models]), dim=0
        )

    global_model.load_state_dict(global_dict)
    return global_model

# =========================================================
# 5. Therapy Agent (DQN-style)
# =========================================================
class TherapyAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.net(state)

# =========================================================
# 6. Personalized Multi-Agent RL System
# =========================================================
class PMARLSystem:
    def __init__(self, state_dim):
        self.agents = {
            "medication": TherapyAgent(state_dim, 3),
            "nutrition": TherapyAgent(state_dim, 4),
            "exercise": TherapyAgent(state_dim, 3),
            "mental_health": TherapyAgent(state_dim, 2)
        }

    def act(self, state):
        actions = {}
        for name, agent in self.agents.items():
            q_vals = agent(state)
            actions[name] = torch.argmax(q_vals, dim=-1)
        return actions

# =========================================================
# 7. End-to-End Pipeline
# =========================================================
def full_pipeline(iomt_data, global_model, pmarl_system):
    with torch.no_grad():
        embedding = global_model(iomt_data)
    therapy_plan = pmarl_system.act(embedding)
    return therapy_plan

# =========================================================
# 8. Example Execution (Simulation)
# =========================================================
if __name__ == "__main__":

    # Simulated IoMT image (X-ray / sensor image)
    x = torch.randn(1, 1, 224, 224)

    # Initialize global & local models
    global_model = ResHyperTransformerNet()
    local_models = [ResHyperTransformerNet() for _ in range(3)]

    # Federated aggregation
    global_model = fed_avg(global_model, local_models)

    # Initialize PMARL
    pmarl = PMARLSystem(state_dim=2)

    # Run pipeline
    actions = full_pipeline(x, global_model, pmarl)

    print("Personalized Therapy Actions:")
    for k, v in actions.items():
        print(f"{k}: {v.item()}")
