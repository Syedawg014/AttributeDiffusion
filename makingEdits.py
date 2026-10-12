import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from scheduler import LinearNoiseScheduler
from modelArch import MLP
from dataloader import DirectionDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lam = 0.7
gamma = 1.8
num_timesteps = 1000
save_step = 500  
num_variations = 3

with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device).eval()

with torch.no_grad():
    model = MLP(in_dim=18 * 512, hidden_dim=1024, t_emb_dim=128).to(device)
    model.load_state_dict(torch.load('hairstyles.pth', map_location=device))
    model.eval()

    scheduler = LinearNoiseScheduler(num_timesteps=num_timesteps)

    dataset = DirectionDataset()
    mean, std = dataset.mean.to(device), dataset.std.to(device)

    latents = torch.from_numpy(np.load('dataset_j/final/test_latents.npy')).float().to(device)

    idx = np.random.randint(200,len(latents))
    print("Using test latent", idx)
    w = latents[idx:idx + 1]

    x = torch.randn(1, 18 * 512, device=device)
    for step in reversed(range(num_timesteps)):
        if step == save_step:
            x_saved = x.clone()
        t = torch.full((1,), step, device=device, dtype=torch.long)
        noise_pred = model(x, t)
        x, i = scheduler.sample_prev_timestep(x, noise_pred, t)

    directions = [x]

    for _ in range(num_variations):
        v = x_saved.clone()
        for step in reversed(range(save_step + 1)):
            t = torch.full((1,), step, device=device, dtype=torch.long)
            noise_pred = model(v, t)
            v, i = scheduler.sample_prev_timestep(v, noise_pred, t)
        directions.append(v)


    def to_image(w):
        img = G.synthesis(w, noise_mode='const')
        return ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()


    fig, axes = plt.subplots(1, 2 + num_variations, figsize=(8, 8))

    axes[0].imshow(to_image(w))
    axes[0].set_title('Original')

    for i, d in enumerate(directions):
        direction = (d * std * gamma + mean).reshape(1, 18, 512)

        w_edit = w + lam * direction

        axes[i + 1].imshow(to_image(w_edit))
        name =  f'img {i}'
        axes[i + 1].set_title(f'{name}')

    for ax in axes:
        ax.axis('off')
    plt.savefig('edits.png', dpi=150)
    plt.show()