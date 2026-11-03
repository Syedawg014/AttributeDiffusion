import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import ops_native

sys.modules['op'] = ops_native
sys.modules['models.stylegan2.op'] = ops_native
sys.path.insert(0, 'stylegan2-pytorch')
sys.path.insert(0, 'encoder4editing')

from model import Generator  
from scheduler import LinearNoiseScheduler
from modelArch import MLP
from dataloader import DirectionDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False) 
print(device)

MODELS = {
    'eyeglasses': dict(name='eyeglasses', weights='eyeglasses.pth',
              dataPath='dataset_j/raw/eyeglasses_d.npy', lam=1, gamma=0.65),
    'hair': dict(name='hair', weights='modalHair.pth',
              dataPath='dataset_j/raw/hairstyle_d.npy', lam=0.65, gamma=0.85),
    'smile': dict(name='smile', weights='smile.pth',
              dataPath='dataset_j/raw/smile_d.npy', lam=1.25, gamma=0.65),
    'age': dict(name='age', weights='age.pth',
              dataPath='dataset_j/raw/age_d.npy', lam=1.75, gamma=0.65),
}


def choose_model():
    print('eyeglasses | hair | smile | age')
    choice = input('Which model: ')
    while choice not in MODELS:
        choice = input('Enter: ')
    return MODELS[choice]


def load_generator(path='stylegan2-ffhq-config-f.pt'):
    G = Generator(1024, 512, 8, channel_multiplier=2)
    G.load_state_dict(torch.load(path, map_location=device)['g_ema'])
    return G.to(device).eval()


def to_image(G, w):
    img,i = G([w], input_is_latent=True, randomize_noise=False)
    return ((img.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()


def sample_directions(cfg, num_variations=3, num_timesteps=1000, save_step=500):
    model = MLP(in_dim=18 * 512, hidden_dim=2048, t_emb_dim=128).to(device)
    model.load_state_dict(torch.load(cfg['weights'], map_location=device))
    model.eval()

    scheduler = LinearNoiseScheduler(num_timesteps=num_timesteps)

    dataset = DirectionDataset(path=cfg['dataPath'])
    mean, std = dataset.mean.to(device), dataset.std.to(device)

    x = torch.randn(1, 18 * 512, device=device)
    for step in reversed(range(num_timesteps)):
        if step == save_step:
            x_saved = x.clone()
        t = torch.full((1,), step, device=device, dtype=torch.long)
        noise_pred = model(x, t)
        x,i = scheduler.sample_prev_timestep(x, noise_pred, t)

    directions = [x]

    for _ in range(num_variations):
        v = x_saved.clone()
        for step in reversed(range(save_step + 1)):
            t = torch.full((1,), step, device=device, dtype=torch.long)
            noise_pred = model(v, t)
            v,i = scheduler.sample_prev_timestep(v, noise_pred, t)
        directions.append(v)

    return [(d * std * cfg['gamma'] + mean).reshape(1, 18, 512) for d in directions]


if __name__ == '__main__':
    cfg = choose_model()
    lam = cfg['lam']
    G = load_generator()

    latents = torch.from_numpy(np.load('dataset_j/final/test_latents.npy')).float().to(device)
    idx = np.random.randint(len(latents)-1)
    print("Using test latent", idx)
    w = latents[idx:idx + 1]

    directions = sample_directions(cfg)

    fig, axes = plt.subplots(1,5, figsize=(8, 8))
    axes[0].imshow(to_image(G, w))
    axes[0].set_title('Original')

    for i, direction in enumerate(directions):
        axes[i + 1].imshow(to_image(G, w + lam * direction))
        axes[i + 1].set_title(f'{cfg["name"]} {i}')

    for ax in axes:
        ax.axis('off')
    plt.savefig(f'outputs/edits_{cfg["name"]}.png', dpi=150)
    plt.show()