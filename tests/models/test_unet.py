"""Unit tests for the plain 2D U-Net baseline.

All tests run on CPU with small synthetic tensors so that the suite is fast
and requires no GPU or real data.

Tests are organised into groups:

* Instantiation — the model can be constructed with various ``num_classes``.
* Shape contract — input ``[B, 1, H, W]`` produces output ``[B, C, H, W]``.
* Spatial preservation — output H and W match input H and W.
* Forward pass — no exceptions are raised; output is a float tensor.
* Batch size — different batch sizes all work correctly.
* Configurable num_classes — the final channel count matches ``num_classes``.
* Logits — output is raw logits (no softmax / sigmoid applied inside).
* Channel progression — expected encoder / decoder channels are present.
* Skip connections — four concatenation-based skip connections are present.
* Upsampling — ConvTranspose2d is used; bilinear / nearest interpolation is not.
* No normalisation — no normalisation layers exist in the model.
"""

import torch
import torch.nn as nn
import pytest

from src.models.unet import UNet
from src.models.unet.blocks import DecoderBlock, DoubleConvBlock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(batch: int = 1, height: int = 64, width: int = 64) -> torch.Tensor:
    """Return a synthetic input tensor of shape ``[B, 1, H, W]``."""
    return torch.zeros(batch, 1, height, width)


def _get_all_modules(model: nn.Module) -> list[nn.Module]:
    """Recursively collect every ``nn.Module`` inside *model*."""
    return list(model.modules())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_2cls() -> UNet:
    """Return a U-Net with two output classes."""
    return UNet(in_channels=1, num_classes=2)


@pytest.fixture
def model_4cls() -> UNet:
    """Return a U-Net with four output classes."""
    return UNet(in_channels=1, num_classes=4)


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------


def test_instantiation_default_args() -> None:
    """U-Net can be instantiated with default arguments."""
    model = UNet()
    assert isinstance(model, UNet)


def test_instantiation_explicit_args() -> None:
    """U-Net can be instantiated with explicit in_channels and num_classes."""
    model = UNet(in_channels=1, num_classes=3)
    assert isinstance(model, UNet)


def test_instantiation_single_class() -> None:
    """U-Net can be instantiated for binary segmentation (num_classes=1)."""
    model = UNet(in_channels=1, num_classes=1)
    assert isinstance(model, UNet)


def test_instantiation_many_classes() -> None:
    """U-Net can be instantiated with a large num_classes value."""
    model = UNet(in_channels=1, num_classes=10)
    assert isinstance(model, UNet)


# ---------------------------------------------------------------------------
# 2. Input / Output shape contract
# ---------------------------------------------------------------------------


def test_output_shape_single_batch(model_2cls: UNet) -> None:
    """Output shape is [1, num_classes, H, W] for a single-sample batch."""
    x = _make_input(batch=1, height=64, width=64)
    out = model_2cls(x)
    assert out.shape == (1, 2, 64, 64)


def test_output_shape_standard_input(model_2cls: UNet) -> None:
    """Output shape is correct for a 256x256 input."""
    x = _make_input(batch=1, height=256, width=256)
    out = model_2cls(x)
    assert out.shape == (1, 2, 256, 256)


def test_output_shape_matches_input_spatial_dims(model_2cls: UNet) -> None:
    """Output spatial dimensions H and W equal the input spatial dimensions."""
    h, w = 128, 128
    x = _make_input(batch=1, height=h, width=w)
    out = model_2cls(x)
    assert out.shape[2] == h
    assert out.shape[3] == w


# ---------------------------------------------------------------------------
# 3. Spatial dimension preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hw", [16, 32, 64, 128, 256])
def test_spatial_dimensions_preserved(hw: int) -> None:
    """Output H and W match input H and W for various square sizes divisible by 16."""
    model = UNet(in_channels=1, num_classes=2)
    x = torch.zeros(1, 1, hw, hw)
    out = model(x)
    assert out.shape[2] == hw, f"Height mismatch for hw={hw}"
    assert out.shape[3] == hw, f"Width mismatch for hw={hw}"


# ---------------------------------------------------------------------------
# 4. Forward pass
# ---------------------------------------------------------------------------


def test_forward_pass_completes(model_2cls: UNet) -> None:
    """Forward pass on a synthetic input raises no exceptions."""
    x = _make_input(batch=2, height=64, width=64)
    out = model_2cls(x)
    assert out is not None


def test_forward_pass_returns_float_tensor(model_2cls: UNet) -> None:
    """Forward pass returns a floating-point tensor."""
    x = _make_input(batch=1, height=64, width=64)
    out = model_2cls(x)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# 5. Different batch sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_various_batch_sizes(batch: int) -> None:
    """Forward pass works for different batch sizes."""
    model = UNet(in_channels=1, num_classes=2)
    x = torch.zeros(batch, 1, 64, 64)
    out = model(x)
    assert out.shape[0] == batch


# ---------------------------------------------------------------------------
# 6. Configurable num_classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_classes", [1, 2, 3, 5, 10])
def test_configurable_num_classes(num_classes: int) -> None:
    """The output channel count equals num_classes for any value."""
    model = UNet(in_channels=1, num_classes=num_classes)
    x = _make_input(batch=1)
    out = model(x)
    assert out.shape[1] == num_classes


# ---------------------------------------------------------------------------
# 7. Output is raw logits
# ---------------------------------------------------------------------------


def test_output_is_not_probability_distribution(model_2cls: UNet) -> None:
    """No Sigmoid or Softmax module exists in the model graph.

    This is a structural check: if neither ``nn.Sigmoid`` nor ``nn.Softmax``
    (nor ``nn.Softmax2d``) is present as a submodule, the model cannot be
    squashing its outputs into probabilities internally.  The complementary
    ``test_output_channel_sum_is_not_one`` test provides the behavioural
    verification that channel-wise sums are not 1.0.
    """
    activation_modules = [
        m
        for m in model_2cls.modules()
        if isinstance(m, (nn.Sigmoid, nn.Softmax, nn.Softmax2d))
    ]
    assert len(activation_modules) == 0, (
        f"Found unexpected output-squashing activation(s): {activation_modules}"
    )


def test_output_channel_sum_is_not_one(model_2cls: UNet) -> None:
    """Channel-wise sum is not 1.0 everywhere; softmax is not applied."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 64)
    out = model_2cls(x)
    channel_sum = out.sum(dim=1)  # [B, H, W]
    # If softmax were applied the sum would be exactly 1.0 everywhere
    not_all_one = not torch.allclose(
        channel_sum, torch.ones_like(channel_sum), atol=1e-4
    )
    assert not_all_one, "Channel sum is 1.0; the model appears to apply softmax."


def test_no_activation_after_final_conv(model_2cls: UNet) -> None:
    """The ``final_conv`` attribute is a plain Conv2d with no following activation."""
    assert isinstance(model_2cls.final_conv, nn.Conv2d)
    assert model_2cls.final_conv.kernel_size == (1, 1)


# ---------------------------------------------------------------------------
# 8. Architecture — channel progression
# ---------------------------------------------------------------------------


def test_encoder1_output_channels() -> None:
    """Encoder stage 1 produces 64-channel feature maps."""
    model = UNet(in_channels=1, num_classes=2)
    conv = list(model.encoder1.block.children())[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.out_channels == 64


def test_encoder2_output_channels() -> None:
    """Encoder stage 2 produces 128-channel feature maps."""
    model = UNet(in_channels=1, num_classes=2)
    conv = list(model.encoder2.block.children())[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.out_channels == 128


def test_encoder3_output_channels() -> None:
    """Encoder stage 3 produces 256-channel feature maps."""
    model = UNet(in_channels=1, num_classes=2)
    conv = list(model.encoder3.block.children())[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.out_channels == 256


def test_encoder4_output_channels() -> None:
    """Encoder stage 4 produces 512-channel feature maps."""
    model = UNet(in_channels=1, num_classes=2)
    conv = list(model.encoder4.block.children())[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.out_channels == 512


def test_bottleneck_output_channels() -> None:
    """Bottleneck produces 1024-channel feature maps."""
    model = UNet(in_channels=1, num_classes=2)
    conv = list(model.bottleneck.block.children())[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.out_channels == 1024


def test_bottleneck_input_channels() -> None:
    """Bottleneck receives 512-channel input from pool4."""
    model = UNet(in_channels=1, num_classes=2)
    conv = list(model.bottleneck.block.children())[0]
    assert isinstance(conv, nn.Conv2d)
    assert conv.in_channels == 512


def test_final_conv_input_channels() -> None:
    """Final 1x1 conv receives 64-channel input from decoder stage 1."""
    model = UNet(in_channels=1, num_classes=2)
    assert model.final_conv.in_channels == 64


def test_final_conv_output_channels_matches_num_classes() -> None:
    """Final 1x1 conv produces exactly num_classes output channels."""
    for nc in [1, 2, 5]:
        model = UNet(in_channels=1, num_classes=nc)
        assert model.final_conv.out_channels == nc


# ---------------------------------------------------------------------------
# 9. Skip connections — four DecoderBlocks are present
# ---------------------------------------------------------------------------


def test_four_decoder_blocks_exist() -> None:
    """The model contains exactly four DecoderBlock instances."""
    model = UNet(in_channels=1, num_classes=2)
    decoder_blocks = [m for m in model.modules() if isinstance(m, DecoderBlock)]
    assert len(decoder_blocks) == 4


def test_four_encoder_stages_exist() -> None:
    """The model contains exactly four encoder DoubleConvBlocks (excluding bottleneck)."""
    model = UNet(in_channels=1, num_classes=2)
    assert isinstance(model.encoder1, DoubleConvBlock)
    assert isinstance(model.encoder2, DoubleConvBlock)
    assert isinstance(model.encoder3, DoubleConvBlock)
    assert isinstance(model.encoder4, DoubleConvBlock)


def test_skip_connection_channels_decoder4() -> None:
    """DoubleConv in decoder4 receives out_channels + skip_channels = 512 + 512 = 1024.

    Channel flow: ConvTranspose2d(1024->512) -> cat(512 skip) -> DoubleConv(1024->512).
    """
    model = UNet(in_channels=1, num_classes=2)
    first_conv = list(model.decoder4.conv.block.children())[0]
    assert isinstance(first_conv, nn.Conv2d)
    assert first_conv.in_channels == 1024


def test_skip_connection_channels_decoder3() -> None:
    """DoubleConv in decoder3 receives out_channels + skip_channels = 256 + 256 = 512.

    Channel flow: ConvTranspose2d(512->256) -> cat(256 skip) -> DoubleConv(512->256).
    """
    model = UNet(in_channels=1, num_classes=2)
    first_conv = list(model.decoder3.conv.block.children())[0]
    assert isinstance(first_conv, nn.Conv2d)
    assert first_conv.in_channels == 512


def test_skip_connection_channels_decoder2() -> None:
    """DoubleConv in decoder2 receives out_channels + skip_channels = 128 + 128 = 256.

    Channel flow: ConvTranspose2d(256->128) -> cat(128 skip) -> DoubleConv(256->128).
    """
    model = UNet(in_channels=1, num_classes=2)
    first_conv = list(model.decoder2.conv.block.children())[0]
    assert isinstance(first_conv, nn.Conv2d)
    assert first_conv.in_channels == 256


def test_skip_connection_channels_decoder1() -> None:
    """DoubleConv in decoder1 receives out_channels + skip_channels = 64 + 64 = 128.

    Channel flow: ConvTranspose2d(128->64) -> cat(64 skip) -> DoubleConv(128->64).
    """
    model = UNet(in_channels=1, num_classes=2)
    first_conv = list(model.decoder1.conv.block.children())[0]
    assert isinstance(first_conv, nn.Conv2d)
    assert first_conv.in_channels == 128


# ---------------------------------------------------------------------------
# 10. Upsampling uses ConvTranspose2d
# ---------------------------------------------------------------------------


def test_upsamplers_are_conv_transpose2d() -> None:
    """Every DecoderBlock uses ConvTranspose2d (not bilinear / nearest)."""
    model = UNet(in_channels=1, num_classes=2)
    decoder_blocks = [m for m in model.modules() if isinstance(m, DecoderBlock)]
    for block in decoder_blocks:
        assert isinstance(block.upsample, nn.ConvTranspose2d), (
            f"Expected ConvTranspose2d but got {type(block.upsample)}"
        )


def test_upsamplers_have_kernel2_stride2() -> None:
    """Every ConvTranspose2d upsampler uses kernel_size=2 and stride=2."""
    model = UNet(in_channels=1, num_classes=2)
    decoder_blocks = [m for m in model.modules() if isinstance(m, DecoderBlock)]
    for block in decoder_blocks:
        up = block.upsample
        assert up.kernel_size == (2, 2), f"Expected (2,2) but got {up.kernel_size}"
        assert up.stride == (2, 2), f"Expected (2,2) but got {up.stride}"


@pytest.mark.parametrize(
    ("decoder_attr", "expected_in", "expected_out"),
    [
        ("decoder4", 1024, 512),
        ("decoder3", 512, 256),
        ("decoder2", 256, 128),
        ("decoder1", 128, 64),
    ],
)
def test_conv_transpose_channel_mapping(
    decoder_attr: str, expected_in: int, expected_out: int
) -> None:
    """ConvTranspose2d in each decoder stage maps the expected in/out channels.

    The channel reduction from ``in_channels`` to ``out_channels`` happens
    inside the transposed convolution, before the encoder skip-connection is
    concatenated.  The frozen progression is:

    * decoder4: 1024 → 512
    * decoder3:  512 → 256
    * decoder2:  256 → 128
    * decoder1:  128 →  64
    """
    model = UNet(in_channels=1, num_classes=2)
    up = getattr(model, decoder_attr).upsample
    assert isinstance(up, nn.ConvTranspose2d)
    assert up.in_channels == expected_in, (
        f"{decoder_attr}.upsample.in_channels: "
        f"expected {expected_in}, got {up.in_channels}"
    )
    assert up.out_channels == expected_out, (
        f"{decoder_attr}.upsample.out_channels: "
        f"expected {expected_out}, got {up.out_channels}"
    )


def test_no_upsample_module_in_model() -> None:
    """No nn.Upsample (bilinear/nearest) module exists anywhere in the model."""
    model = UNet(in_channels=1, num_classes=2)
    upsample_modules = [m for m in model.modules() if isinstance(m, nn.Upsample)]
    assert len(upsample_modules) == 0


# ---------------------------------------------------------------------------
# 11. No normalisation layers
# ---------------------------------------------------------------------------


_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)


def test_no_normalization_layers() -> None:
    """The model contains no normalisation layers of any kind."""
    model = UNet(in_channels=1, num_classes=2)
    norm_modules = [m for m in model.modules() if isinstance(m, _NORM_TYPES)]
    assert len(norm_modules) == 0, (
        f"Found unexpected normalisation layer(s): {norm_modules}"
    )


# ---------------------------------------------------------------------------
# 12. Activation — only ReLU is present in blocks
# ---------------------------------------------------------------------------


def test_only_relu_activations_in_model() -> None:
    """Every activation module present in the model is ``nn.ReLU``.

    This is a positive assertion: we collect all submodules that are
    instances of any ``nn.Module`` subclass known to act as an activation
    function and assert that each one is ``nn.ReLU``.  The set of checked
    activation types covers all common alternatives.
    """
    _ACTIVATION_TYPES = (
        nn.ReLU,
        nn.GELU,
        nn.SiLU,
        nn.LeakyReLU,
        nn.PReLU,
        nn.ELU,
        nn.Tanh,
        nn.Sigmoid,
        nn.Softmax,
        nn.Softmax2d,
        nn.Hardswish,
        nn.Mish,
    )
    model = UNet(in_channels=1, num_classes=2)
    activation_modules = [
        m for m in model.modules() if isinstance(m, _ACTIVATION_TYPES)
    ]
    non_relu = [m for m in activation_modules if not isinstance(m, nn.ReLU)]
    assert len(non_relu) == 0, (
        f"Expected only nn.ReLU activations, but found: {non_relu}"
    )
    # Also confirm ReLU is actually present (the DoubleConvBlocks use it)
    assert len(activation_modules) > 0, (
        "No activation modules found in model; expected nn.ReLU instances."
    )


# ---------------------------------------------------------------------------
# 13. Pooling layers
# ---------------------------------------------------------------------------


def test_four_maxpool_layers_exist() -> None:
    """The model contains exactly four MaxPool2d pooling layers."""
    model = UNet(in_channels=1, num_classes=2)
    pools = [m for m in model.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(pools) == 4


def test_maxpool_uses_kernel2_stride2() -> None:
    """All MaxPool2d layers use kernel_size=2 and stride=2."""
    model = UNet(in_channels=1, num_classes=2)
    pools = [m for m in model.modules() if isinstance(m, nn.MaxPool2d)]
    for pool in pools:
        assert pool.kernel_size == 2
        assert pool.stride == 2


# ---------------------------------------------------------------------------
# 14. Final conv is 1x1
# ---------------------------------------------------------------------------


def test_final_conv_is_1x1() -> None:
    """The final projection convolution uses kernel_size=1."""
    model = UNet(in_channels=1, num_classes=2)
    assert model.final_conv.kernel_size == (1, 1)


# ---------------------------------------------------------------------------
# 15. 3x3 conv padding
# ---------------------------------------------------------------------------


def test_double_conv_uses_padding_1() -> None:
    """Every 3x3 convolution inside DoubleConvBlock uses padding=1."""
    model = UNet(in_channels=1, num_classes=2)
    double_conv_blocks = [m for m in model.modules() if isinstance(m, DoubleConvBlock)]
    for block in double_conv_blocks:
        for m in block.block:
            if isinstance(m, nn.Conv2d):
                assert m.padding == (1, 1), (
                    f"Conv2d in DoubleConvBlock has padding {m.padding}, expected (1,1)"
                )
