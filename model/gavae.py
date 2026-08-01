import torch
from torch import nn
from torch.nn import functional as F


GEO_AIS_FEATURE_DIM = 15
MOTION_LOSS_WEIGHT = 0.001
KL_WEIGHT = 1.0
LOWER_RECON_WEIGHT = 0.75
EDGE_RECON_WEIGHT = 0.35
COVERAGE_LOSS_WEIGHT = 1.0
RANGE_LOSS_WEIGHT = 0.0
LOWER_RANGE_LOSS_WEIGHT = 0.0


def _same_padding(kernel_size):
    total = kernel_size - 1
    left = total // 2
    return left, total - left


class SameConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        left, right = _same_padding(self.kernel_size)
        return self.conv(F.pad(x, (left, right)))


class SameConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        x = self.conv(x)
        left, right = _same_padding(self.kernel_size)
        end = x.shape[-1] - right if right else x.shape[-1]
        return x[..., left:end]


def geo_ais_features(x):
    x = x.float()
    delta = torch.diff(x, dim=1, prepend=x[:, :1])
    speed = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    direction = delta / speed.clamp_min(1e-6)
    acceleration = torch.diff(delta, dim=1, prepend=delta[:, :1])
    acceleration_norm = torch.linalg.vector_norm(acceleration, dim=-1, keepdim=True)
    previous_direction = torch.cat([direction[:, :1], direction[:, :-1]], dim=1)
    turn_sin = direction[..., 0:1] * previous_direction[..., 1:2] - direction[..., 1:2] * previous_direction[..., 0:1]
    turn_cos = (direction * previous_direction).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    relative = x - x[:, :1]
    progress = torch.linspace(0.0, 1.0, x.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1).expand(x.shape[0], -1, -1)
    return torch.cat(
        [
            x,
            relative,
            delta,
            speed,
            direction,
            acceleration,
            acceleration_norm,
            turn_sin,
            turn_cos,
            progress,
        ],
        dim=-1,
    )


class Encoder(nn.Module):
    def __init__(self, seq_len, feature_dim, latent_dim, dropout_rate=0.2, use_geo_features=True, use_social_context=True):
        super().__init__()
        self.use_geo_features = use_geo_features
        self.use_social_context = use_social_context
        self.convs = nn.ModuleList(
            [
                SameConv1d(feature_dim, 64, 10),
                SameConv1d(64, 64, 2),
                SameConv1d(64, 64, 2),
                SameConv1d(64, 64, 2),
                SameConv1d(64, 64, 4),
            ]
        )
        self.geo_projection = nn.Sequential(
            nn.Linear(GEO_AIS_FEATURE_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.geo_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.geo_projection[-1].weight)
        nn.init.zeros_(self.geo_projection[-1].bias)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])
        self.dense = nn.Linear(seq_len * 64, 512)
        self.dense_1 = nn.Linear(512, 64)
        self.social_pool = nn.Sequential(
            nn.Linear(64 * 3, 64),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64),
        )
        self.social_scale = nn.Parameter(torch.tensor(0.1))
        self.z_mean = nn.Linear(64, latent_dim)
        self.z_log_var = nn.Linear(64, latent_dim)

    def forward(self, x):
        if self.use_geo_features:
            geo = self.geo_projection(geo_ais_features(x)).transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.convs[0](x)
        if self.use_geo_features:
            x = x + self.geo_scale * geo
        x = self.dropouts[0](F.relu(x))
        for conv, dropout in zip(self.convs[1:], self.dropouts[1:]):
            x = dropout(F.relu(conv(x)))
        x = x.transpose(1, 2).reshape(x.shape[0], -1)
        x = F.relu(self.dense(x))
        x = F.relu(self.dense_1(x))
        if self.use_social_context:
            pooled_mean = x.mean(dim=0, keepdim=True).expand_as(x)
            pooled_std = x.std(dim=0, unbiased=False, keepdim=True).expand_as(x)
            x = x + self.social_scale * self.social_pool(torch.cat([x, pooled_mean, pooled_std], dim=1))
        return self.z_mean(x), self.z_log_var(x)


class Decoder(nn.Module):
    def __init__(self, seq_len, feature_dim, latent_dim, dropout_rate=0.2, use_motion_path=True):
        super().__init__()
        self.use_motion_path = use_motion_path
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.dense_2 = nn.Linear(latent_dim, 64)
        self.dense_3 = nn.Linear(64, 512)
        self.dense_4 = nn.Linear(512, 64)
        self.dense_channels = 64 * seq_len
        self.dense_5 = nn.Linear(64, seq_len * self.dense_channels)
        self.convs = nn.ModuleList(
            [
                SameConvTranspose1d(self.dense_channels, 64, 2),
                SameConvTranspose1d(64, 64, 2),
                SameConvTranspose1d(64, 64, 2),
                SameConvTranspose1d(64, 64, 2),
                SameConvTranspose1d(64, 64, 10),
                SameConvTranspose1d(64, feature_dim, 3),
            ]
        )
        self.anchor = nn.Linear(latent_dim, feature_dim)
        self.delta_head = nn.Conv1d(64, feature_dim, 1)
        self.blend_head = nn.Conv1d(64, feature_dim, 1)
        self.motion_fusion = nn.Sequential(
            nn.Conv1d(64 + feature_dim * 2, 64, 1),
            nn.SiLU(),
            nn.Conv1d(64, 64, 1),
        )
        self.motion_residual_head = nn.Conv1d(64, feature_dim, 1)
        self.motion_residual_gate = nn.Conv1d(64, feature_dim, 1)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.blend_head.weight)
        nn.init.constant_(self.blend_head.bias, 8.0)
        nn.init.zeros_(self.motion_residual_head.weight)
        nn.init.zeros_(self.motion_residual_head.bias)
        nn.init.zeros_(self.motion_residual_gate.weight)
        nn.init.constant_(self.motion_residual_gate.bias, -2.0)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])

    def forward(self, z):
        x = F.relu(self.dense_2(z))
        x = F.relu(self.dense_3(x))
        x = F.relu(self.dense_4(x))
        x = F.relu(self.dense_5(x))
        x = x.reshape(x.shape[0], self.seq_len, self.dense_channels).transpose(1, 2)
        for conv, dropout in zip(self.convs[:-1], self.dropouts):
            x = dropout(F.relu(conv(x)))
        direct = torch.sigmoid(self.convs[-1](x)).transpose(1, 2)
        if not self.use_motion_path:
            return direct
        anchor = torch.sigmoid(self.anchor(z)).unsqueeze(1)
        delta = 0.05 * torch.tanh(self.delta_head(x).transpose(1, 2))
        motion = torch.clamp(anchor + torch.cumsum(delta, dim=1), 0.0, 1.0)
        direct_weight = torch.sigmoid(self.blend_head(x).transpose(1, 2))
        fused = direct_weight * direct + (1.0 - direct_weight) * motion
        motion_context = self.motion_fusion(torch.cat([x, direct.transpose(1, 2), motion.transpose(1, 2)], dim=1))
        residual = 0.05 * torch.tanh(self.motion_residual_head(motion_context).transpose(1, 2))
        residual_gate = torch.sigmoid(self.motion_residual_gate(motion_context).transpose(1, 2))
        return torch.clamp(fused + residual_gate * residual, 0.0, 1.0)


class GAVAE(nn.Module):
    def __init__(
        self,
        seq_len,
        feature_dim,
        latent_dim,
        dropout_rate=0.2,
        use_geo_features=True,
        use_motion_path=True,
        use_social_context=True,
    ):
        super().__init__()
        self.encoder = Encoder(
            seq_len,
            feature_dim,
            latent_dim,
            dropout_rate=dropout_rate,
            use_geo_features=use_geo_features,
            use_social_context=use_social_context,
        )
        self.decoder = Decoder(seq_len, feature_dim, latent_dim, dropout_rate=dropout_rate, use_motion_path=use_motion_path)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, z_mean, z_log_var, epsilon=None):
        if epsilon is None:
            epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

    def forward(self, x, epsilon=None):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var, epsilon=epsilon)
        return self.decoder(z), z_mean, z_log_var

    def decode(self, z):
        return self.decoder(z)


def reconstruction_loss(x, reconstruction, lower_recon_weight=LOWER_RECON_WEIGHT, edge_recon_weight=EDGE_RECON_WEIGHT):
    x = x.float()
    reconstruction = reconstruction.float()
    if lower_recon_weight or edge_recon_weight:
        lower_weight = (x[..., 1:2] <= 0.5).to(x.dtype)
        edge_weight = (2.0 * (x - 0.5).abs()).mean(dim=-1, keepdim=True)
        weight = 1.0 + lower_recon_weight * lower_weight + edge_recon_weight * edge_weight
        axis0 = torch.sum(weight * (x - reconstruction) ** 2)
    else:
        axis0 = torch.sum((x - reconstruction) ** 2)
    axis1 = F.mse_loss(torch.mean(x, dim=1), torch.mean(reconstruction, dim=1), reduction="none").mean(dim=1)
    axis2 = F.mse_loss(torch.mean(x, dim=2), torch.mean(reconstruction, dim=2), reduction="none").mean(dim=1)
    return axis0 + axis1 + axis2


def _weighted_mean_std(values, weights):
    weights = weights.expand_as(values)
    denom = weights.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (values * weights).sum(dim=(0, 1)) / denom
    var = ((values - mean.view(1, 1, -1)).square() * weights).sum(dim=(0, 1)) / denom
    return mean, torch.sqrt(var.clamp_min(1e-12))


def _soft_extrema(values, temperature=0.02):
    flat = values.reshape(-1, values.shape[-1])
    upper = temperature * torch.logsumexp(flat / temperature, dim=0)
    lower = -temperature * torch.logsumexp(-flat / temperature, dim=0)
    return lower, upper


def coverage_loss(
    x,
    reconstruction,
    coverage_loss_weight=COVERAGE_LOSS_WEIGHT,
    range_loss_weight=RANGE_LOSS_WEIGHT,
    lower_range_loss_weight=LOWER_RANGE_LOSS_WEIGHT,
):
    x = x.float()
    reconstruction = reconstruction.float()
    if coverage_loss_weight == 0.0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    time_std = x.std(dim=0, unbiased=False)
    reconstruction_time_std = reconstruction.std(dim=0, unbiased=False)
    time_std_loss = torch.sum((time_std - reconstruction_time_std).square()) * x.shape[0]
    lower = (x[..., 1:2] <= 0.5).to(x.dtype)
    lower_mean, lower_std = _weighted_mean_std(x, lower)
    reconstruction_lower_mean, reconstruction_lower_std = _weighted_mean_std(reconstruction, lower)
    lower_mean_loss = torch.sum((lower_mean - reconstruction_lower_mean).square()) * x.numel() * 0.1
    lower_std_loss = torch.sum((lower_std - reconstruction_lower_std).square()) * x.numel()
    target_min = x.reshape(-1, x.shape[-1]).min(dim=0).values
    target_max = x.reshape(-1, x.shape[-1]).max(dim=0).values
    reconstruction_min, reconstruction_max = _soft_extrema(reconstruction)
    range_loss = torch.sum((target_min - reconstruction_min).square() + (target_max - reconstruction_max).square()) * x.numel() * range_loss_weight
    lower_target = x[lower.squeeze(-1) > 0.0]
    lower_reconstruction = reconstruction[lower.squeeze(-1) > 0.0]
    if lower_target.numel() > 0:
        lower_target_min = lower_target.min(dim=0).values
        lower_target_max = lower_target.max(dim=0).values
        lower_reconstruction_min, lower_reconstruction_max = _soft_extrema(lower_reconstruction.view(1, -1, x.shape[-1]))
        lower_range_loss = torch.sum(
            (lower_target_min - lower_reconstruction_min).square()
            + (lower_target_max - lower_reconstruction_max).square()
        ) * x.numel() * lower_range_loss_weight
    else:
        lower_range_loss = torch.zeros((), device=x.device, dtype=x.dtype)
    return (time_std_loss + lower_mean_loss + lower_std_loss + range_loss + lower_range_loss) * coverage_loss_weight


def motion_consistency_loss(x, reconstruction):
    x = x.float()
    reconstruction = reconstruction.float()
    velocity = torch.diff(x, dim=1)
    reconstruction_velocity = torch.diff(reconstruction, dim=1)
    acceleration = torch.diff(velocity, dim=1)
    reconstruction_acceleration = torch.diff(reconstruction_velocity, dim=1)
    velocity_loss = torch.sum((velocity - reconstruction_velocity) ** 2)
    acceleration_loss = torch.sum((acceleration - reconstruction_acceleration) ** 2)
    speed_loss = torch.sum(
        (
            torch.linalg.vector_norm(velocity, dim=-1)
            - torch.linalg.vector_norm(reconstruction_velocity, dim=-1)
        )
        ** 2
    )
    return velocity_loss + 0.5 * acceleration_loss + speed_loss


def vae_losses(
    x,
    reconstruction,
    z_mean,
    z_log_var,
    motion_loss_weight=MOTION_LOSS_WEIGHT,
    kl_weight=KL_WEIGHT,
    lower_recon_weight=LOWER_RECON_WEIGHT,
    edge_recon_weight=EDGE_RECON_WEIGHT,
    coverage_loss_weight=COVERAGE_LOSS_WEIGHT,
    range_loss_weight=RANGE_LOSS_WEIGHT,
    lower_range_loss_weight=LOWER_RANGE_LOSS_WEIGHT,
):
    recon = reconstruction_loss(
        x,
        reconstruction,
        lower_recon_weight=lower_recon_weight,
        edge_recon_weight=edge_recon_weight,
    )
    motion = motion_consistency_loss(x, reconstruction) * motion_loss_weight
    coverage = coverage_loss(
        x,
        reconstruction,
        coverage_loss_weight=coverage_loss_weight,
        range_loss_weight=range_loss_weight,
        lower_range_loss_weight=lower_range_loss_weight,
    )
    kl = -0.5 * (1 + z_log_var.float() - z_mean.float().square() - torch.exp(z_log_var.float()))
    kl = torch.mean(torch.sum(kl, dim=1))
    total = recon + motion + coverage + kl_weight * kl
    return total, recon, kl, motion, coverage


def build_vae(
    seq_len,
    feature_dim,
    latent_dim,
    dropout_rate=0.2,
    device=None,
    use_geo_features=True,
    use_motion_path=True,
    use_social_context=True,
):
    model = GAVAE(
        seq_len,
        feature_dim,
        latent_dim,
        dropout_rate=dropout_rate,
        use_geo_features=use_geo_features,
        use_motion_path=use_motion_path,
        use_social_context=use_social_context,
    )
    if device is not None:
        model = model.to(device)
    return model
