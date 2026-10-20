import torch

from chanjing_avatar.models.dit import WanModelAudioProject


def test_tiny_model_forward_backward_with_checkpointing():
    model = WanModelAudioProject(
        dim=24,
        in_dim=4,
        ffn_dim=48,
        out_dim=2,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
        vae_stride=(4, 16, 16),
        patch_size=(1, 1, 1),
        num_heads=3,
        num_layers=1,
        has_image_input=False,
    )
    x = torch.randn(1, 2, 2, 2, 2, requires_grad=True)
    y = torch.randn_like(x)
    audio = torch.randn(1, 5, 5, 12, 768)
    output = model(
        x=x,
        timestep=torch.tensor([500.0]),
        context=audio,
        y=y,
        use_gradient_checkpointing=True,
    )
    assert output.shape == x.shape
    output.square().mean().backward()
    assert x.grad is not None
