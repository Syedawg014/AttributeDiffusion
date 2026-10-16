import torch
import torch.nn as nn
from torch.optim import AdamW

from scheduler import LinearNoiseScheduler
from modelArch import MLP
from dataloader import get_dataloader


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_timesteps = 1000
    scheduler = LinearNoiseScheduler(num_timesteps=num_timesteps, beta_start=0.0001, beta_end=0.02)
    train_loader, train_dataset = get_dataloader(batch_size=256)
    model = MLP(in_dim=18 * 512, hidden_dim=2048, t_emb_dim=128).to(device)
    num_epochs = 100
    lr = 2e-4

    optimizer = AdamW(model.parameters(), weight_decay=0.01, lr=lr)
    criterion = nn.MSELoss()

    model.train()
    print("Starting Training Loop")

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch_idx, directions in enumerate(train_loader):
            directions = directions.float().to(device)

            noise = torch.randn_like(directions).to(device)

            t = torch.randint(0, num_timesteps, (directions.shape[0],), device=device)

            noisy_directions = scheduler.add_noise(directions, noise, t)

            optimizer.zero_grad()


            noise_pred = model(noisy_directions, t)

            loss = criterion(noise_pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Average Loss: {avg_loss}")

        torch.save(model.state_dict(), 'hairstyles.pth')

    print("Training Complete")


if __name__ == '__main__':
    train()
