from model.base import build_model as _build_model


def build_model(seq_len, feature_dim, latent_dim, dropout_rate=0.1, device=None):
    return _build_model("socialvae", seq_len, feature_dim, latent_dim, dropout_rate=dropout_rate, device=device)
