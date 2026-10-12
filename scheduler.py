import torch


class LinearNoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t):
        device = t.device
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(device)[t].reshape(-1, 1)
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(device)[t].reshape(-1, 1)
        return sqrt_alpha_cum_prod * original + sqrt_one_minus_alpha_cum_prod * noise

    def sample_prev_timestep(self, xt, noise_pred, t):
        device = xt.device
        sqrt_one_minus_alpha = self.sqrt_one_minus_alpha_cum_prod.to(device)[t].reshape(-1, 1)
        sqrt_alpha = self.sqrt_alpha_cum_prod.to(device)[t].reshape(-1, 1)
        beta = self.betas.to(device)[t].reshape(-1, 1)
        alpha = self.alphas.to(device)[t].reshape(-1, 1)

        x0 = (xt - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

        mutheta = xt - (beta * noise_pred) / sqrt_one_minus_alpha
        mutheta = mutheta / torch.sqrt(alpha)

        if (t == 0).all():
            return mutheta, x0
        else:
            alpha_cum_prod = self.alpha_cum_prod.to(device)[t].reshape(-1, 1)
            alpha_cum_prod_prev = self.alpha_cum_prod.to(device)[t - 1].reshape(-1, 1)
            variance = ((1 - alpha_cum_prod_prev) * beta) / (1 - alpha_cum_prod)
            sigma = variance**0.5
            z = torch.randn_like(xt)
            return mutheta + sigma * z, x0
