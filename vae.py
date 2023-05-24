from dataset import fetch_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from train import train_model


# Load Dataset
train_ds, train_loader, val_ds, val_loader, attributes = fetch_dataset()

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        
        self.linear1 = nn.Linear(64 * 8 * 8, self.latent_dim)
        self.linear2 = nn.Linear(64 * 8 * 8, self.latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.linear1(h)
        log_var = self.linear2(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

# Instantiate the VAE model
model = VAE(INPUT_CHANNEL, LATENT_DIM).to(DEVICE)

# Define the optimizertrain_dataset
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train the VAE model
if TRAINING == True:
    train_model(model, train_loader, val_loader, optimizer)

