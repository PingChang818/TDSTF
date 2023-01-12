import pickle
import torch
import torch.nn as nn
import numpy as np
from attn import ResNet

class TDSTF(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config_diff = config['diffusion']
        var, _ = pickle.load(open('preprocess/data/var.pkl', 'rb'))
        self.lv = len(var)
        self.res_model = ResNet(self.config_diff, self.device)
        # parameters for diffusion model
        self.num_steps = self.config_diff['num_steps']
        self.beta = np.linspace(self.config_diff['beta_start'] ** 0.5, self.config_diff['beta_end'] ** 0.5, self.num_steps) ** 2
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1)

    def process(self, batch):
        samples_x = batch['samples_x'].to(self.device).float()
        samples_y = batch['samples_y'].to(self.device).float()
        info = batch['info'].to(self.device)
        
        return samples_x, samples_y, info

    def forward(self, batch, size_x, size_y):
        samples_x, samples_y, info = self.process(batch)
        t = torch.randint(0, self.num_steps, [len(samples_x)]).to(self.device)
        current_alpha = self.alpha_torch[t]
        noise = torch.randn((len(samples_x), size_y)).to(samples_y.device)
        mask_x = samples_x[:, 3]
        mask_y = samples_y[:, 3]
        samples_x[:, 0] = torch.where(mask_x == 1, samples_x[:, 0], self.lv)
        samples_x[:, 1] = torch.where(mask_x == 1, samples_x[:, 1], -1)
        samples_y[:, 0] = torch.where(mask_y == 1, samples_y[:, 0], self.lv)
        samples_y[:, 1] = torch.where(mask_y == 1, samples_y[:, 1], -1)
        samples_y[:, 2] = ((current_alpha ** 0.5) * samples_y[:, 2] + ((1.0 - current_alpha) ** 0.5) * noise) * mask_y
        predicted = self.res_model(samples_x, samples_y, info, t)
        residual = torch.where(mask_y == 1, noise - predicted, 0)
        loss = (residual ** 2).sum() / info[:, 2].sum()

        return loss

    def forecast(self, samples_x, samples_y, info, n_samples):
        generation = torch.zeros(n_samples, samples_y.shape[0], samples_y.shape[-1]).to(self.device)
        for i in range(n_samples):
            samples_y[:, 2] = torch.randn_like(samples_y[:, 2]) * samples_y[:, 3]
            for t in range(self.num_steps - 1, -1, -1):
                mask_x = samples_x[:, 3]
                mask_y = samples_y[:, 3]
                samples_x[:, 0] = torch.where(mask_x == 1, samples_x[:, 0], self.lv)
                samples_x[:, 1] = torch.where(mask_x == 1, samples_x[:, 1], -1)
                samples_y[:, 0] = torch.where(mask_y == 1, samples_y[:, 0], self.lv)
                samples_y[:, 1] = torch.where(mask_y == 1, samples_y[:, 1], -1)
                predicted = self.res_model(samples_x, samples_y, info, torch.tensor([t]).to(self.device))
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                samples_y[:, 2] = coeff1 * (samples_y[:, 2] - coeff2 * predicted) * samples_y[:, 3]
                if t > 0:
                    noise = torch.randn_like(samples_y[:, 2]) * samples_y[:, 3]
                    sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                    samples_y[:, 2] += sigma * noise

            generation[i] = samples_y[:, 2].detach()
            
        return generation.permute(1, 2, 0)

    def evaluate(self, batch, n_samples):
        samples_x, samples_y, info = self.process(batch)
        with torch.no_grad():
            generation = self.forecast(samples_x, samples_y, info, n_samples)
            
        return generation, samples_y, samples_x
