import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

def Conv1d_with_init(in_channels, out_channels, kernel_size, device):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size).to(device)
    nn.init.kaiming_normal_(layer.weight)
    
    return layer

class TimeEmbedding(nn.Module):
    def __init__(self, tp, d_model, device):
        super().__init__()
        self.device = device
        self.register_buffer('time_embedding', self._build_embedding(tp + 1, d_model), persistent=False)
    
    def forward(self, m):
        return self.time_embedding[m]
    
    def _build_embedding(self, t, d_model):
        pe = torch.zeros(t, d_model).to(self.device)
        position = torch.arange(t).unsqueeze(1).to(self.device)
        div_term = (1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)).to(self.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim, device):
        super().__init__()
        self.device = device
        self.register_buffer('diffusion_embedding', self._build_embedding(num_steps, embedding_dim / 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim).to(device)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim).to(device)
        
    def forward(self, diffusion_step):
        x = self.diffusion_embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        
        return x
    
    def _build_embedding(self, num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1).to(self.device)
        frequencies = (10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)).to(self.device)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        
        return table

class ResNet(nn.Module):
    def __init__(self, config, device):
            super().__init__()
            var, target_var = pickle.load(open('preprocess/data/var.pkl', 'rb'))
            lv = len(var)
            self.size_x = config['size']
            self.size_y = 10 * len(target_var)
            self.channels = config['channels']
            self.emb_f = nn.Embedding(lv + 1, self.channels).to(device)
            self.emb_t = TimeEmbedding(config['time_points'], config['time_embedding_dim'], device)
            self.emb_v = nn.Linear(1, self.channels).to(device)
            self.dec1 = Conv1d_with_init(self.channels, self.channels, 1, device)
            self.dec2 = Conv1d_with_init(self.channels, 1, 1, device)
            self.diffusion_embedding = DiffusionEmbedding(config['num_steps'], config['diffusion_embedding_dim'], device)
            self.diffusion_projection = nn.Linear(config['diffusion_embedding_dim'], self.channels).to(device)
            self.residual_layers = nn.ModuleList([
                Triplet_cor(config, lv, device)
                for _ in range(config['layers'])])
            
    def forward(self, samples_x, samples_y, info, diffusion_step):
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        diffusion_emb = self.diffusion_projection(diffusion_emb)
        diffusion_emb = diffusion_emb.unsqueeze(1).expand(diffusion_emb.shape[0], self.size_x, diffusion_emb.shape[1])
        triplets_x = (self.emb_f(samples_x[:, 0].to(torch.int64))
                    + self.emb_t(samples_x[:, 1].to(torch.int64))
                    + self.emb_v(samples_x[:, 2].unsqueeze(-1))
                    + diffusion_emb) * samples_x[:, 3].unsqueeze(-1)
        triplets_y = (self.emb_f(samples_y[:, 0].to(torch.int64))
                    + self.emb_t(samples_y[:, 1].to(torch.int64))
                    + self.emb_v(samples_y[:, 2].unsqueeze(-1))
                    ) * samples_y[:, 3].unsqueeze(-1)
        diffussion_emb_y = diffusion_emb[:, : self.size_y] * samples_y[:, 3].unsqueeze(-1)
        skip = []
        for layer in self.residual_layers:
            triplets_y = triplets_y + diffussion_emb_y
            triplets_y, skip_connection = layer(triplets_x, triplets_y)
            skip.append(skip_connection)
            
        output = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        output = self.dec1(output.permute(0, 2, 1))
        output = F.relu(output)
        output = self.dec2(output)
        
        return output.squeeze()

class Triplet_cor(nn.Module):
    def __init__(self, config, lv, device):
        super().__init__()
        self.channels = config['channels']
        self.attn = torch.nn.Transformer(d_model=self.channels, nhead=8, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1, activation='gelu', batch_first=True, device=device)
        self.expand = Conv1d_with_init(self.channels, 2 * self.channels, 1, device)
    
    def forward(self, triplets_x, triplets_y):
        output = self.attn(triplets_x, triplets_y)
        output = self.expand(output.transpose(1, 2)).transpose(1, 2)
        residual, skip = torch.chunk(output, 2, dim=-1)
        
        return residual, skip
