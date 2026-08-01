import torch
from torch import nn
from torch.nn import functional as F


class SequenceAutoencoder(nn.Module):
    def __init__(self, seq_len, feature_dim, latent_dim, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.hidden_dim = 96
        self.encoder = nn.LSTM(
            feature_dim,
            self.hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=True,
        )
        encoder_dim = self.hidden_dim * 2
        self.to_latent = nn.Linear(encoder_dim, latent_dim)
        self.decoder = nn.LSTM(latent_dim + 1, self.hidden_dim, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.output = nn.Sequential(nn.Linear(self.hidden_dim, feature_dim), nn.Sigmoid())

    def encode(self, x):
        y, _ = self.encoder(x)
        return self.to_latent(y[:, -1])

    def decode(self, z):
        progress = torch.linspace(0.0, 1.0, self.seq_len, device=z.device, dtype=z.dtype).view(1, -1, 1)
        progress = progress.expand(z.shape[0], -1, -1)
        repeated = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        y, _ = self.decoder(torch.cat([repeated, progress], dim=2))
        return self.output(y)

    def forward(self, x):
        return self.decode(self.encode(x))


class MLPVAE(nn.Module):
    def __init__(self, seq_len, feature_dim, latent_dim, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        flat_dim = seq_len * feature_dim
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(128, latent_dim)
        self.z_log_var = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, flat_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        y = self.encoder(x.reshape(x.shape[0], -1))
        return self.z_mean(y), self.z_log_var(y)

    def reparameterize(self, z_mean, z_log_var, epsilon=None):
        if epsilon is None:
            epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        return self.decoder(z).reshape(z.shape[0], self.seq_len, self.feature_dim)

    def forward(self, x, epsilon=None):
        z_mean, z_log_var = self.encode(x)
        return self.decode(self.reparameterize(z_mean, z_log_var, epsilon=epsilon)), z_mean, z_log_var


class SocialVAE(nn.Module):
    def __init__(self, seq_len, feature_dim, latent_dim, dropout_rate=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_dim = 96
        self.encoder = nn.LSTM(feature_dim, self.hidden_dim, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.pool = nn.Sequential(nn.Linear(self.hidden_dim * 3, 128), nn.ReLU(), nn.Dropout(dropout_rate))
        self.z_mean = nn.Linear(128, latent_dim)
        self.z_log_var = nn.Linear(128, latent_dim)
        self.decoder = nn.LSTM(latent_dim + 1, self.hidden_dim, num_layers=2, batch_first=True, dropout=dropout_rate)
        self.output = nn.Sequential(nn.Linear(self.hidden_dim, feature_dim), nn.Sigmoid())

    def encode(self, x):
        y, _ = self.encoder(x)
        h = y[:, -1]
        pooled_mean = h.mean(dim=0, keepdim=True).expand_as(h)
        pooled_std = h.std(dim=0, unbiased=False, keepdim=True).expand_as(h)
        y = self.pool(torch.cat([h, pooled_mean, pooled_std], dim=1))
        return self.z_mean(y), self.z_log_var(y)

    def reparameterize(self, z_mean, z_log_var, epsilon=None):
        if epsilon is None:
            epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def decode(self, z):
        progress = torch.linspace(0.0, 1.0, self.seq_len, device=z.device, dtype=z.dtype).view(1, -1, 1)
        progress = progress.expand(z.shape[0], -1, -1)
        repeated = z.unsqueeze(1).expand(-1, self.seq_len, -1)
        y, _ = self.decoder(torch.cat([repeated, progress], dim=2))
        return self.output(y)

    def forward(self, x, epsilon=None):
        z_mean, z_log_var = self.encode(x)
        return self.decode(self.reparameterize(z_mean, z_log_var, epsilon=epsilon)), z_mean, z_log_var


def build_model(model_name, seq_len, feature_dim, latent_dim, dropout_rate=0.1, device=None):
    if model_name == "bilstm":
        model = SequenceAutoencoder(seq_len, feature_dim, latent_dim, dropout_rate=dropout_rate)
    elif model_name == "vae":
        model = MLPVAE(seq_len, feature_dim, latent_dim, dropout_rate=dropout_rate)
    elif model_name == "socialvae":
        model = SocialVAE(seq_len, feature_dim, latent_dim, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    if device is not None:
        model = model.to(device)
    return model


def reconstruction_loss(x, reconstruction):
    return F.mse_loss(reconstruction.float(), x.float())


def vae_loss(x, reconstruction, z_mean, z_log_var, kl_weight=0.001):
    recon = reconstruction_loss(x, reconstruction)
    kl = -0.5 * (1 + z_log_var.float() - z_mean.float().square() - torch.exp(z_log_var.float()))
    kl = torch.mean(torch.sum(kl, dim=1))
    return recon + kl_weight * kl, recon, kl
