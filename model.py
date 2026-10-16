import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

RGB_IDX = [3, 2, 1]
BAND_NAMES = [
    "Coastal", "Blue", "Green", "Red", "NIR", "SWIR1",
    "SWIR2", "QA", "MeritDEM", "CopDEM", "ESA", "WaterOcc",
]


def find_first_conv(encoder, in_channels=3):
    for name, m in encoder.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == in_channels:
            return name, m
    raise RuntimeError("no first conv found in the encoder")


def set_module(root, dotted, new):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def adapt_first_conv_rgb_zero(model, in_ch=12):
    name, conv = find_first_conv(model.encoder)
    w = conv.weight.data.clone()
    new_w = torch.zeros(w.shape[0], in_ch, *w.shape[2:], dtype=w.dtype)
    for src, dst in enumerate(RGB_IDX):
        new_w[:, dst] = w[:, src]
    new_conv = nn.Conv2d(
        in_ch,
        w.shape[0],
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    new_conv.weight = nn.Parameter(new_w)
    set_module(model.encoder, name, new_conv)
    return model


def build_model(encoder="resnet34", in_ch=12):
    base = smp.Unet(encoder, encoder_weights=None, in_channels=3, classes=1)
    return adapt_first_conv_rgb_zero(base, in_ch)


def load_model(weights_path, device="cpu", encoder="resnet34", in_ch=12):
    model = build_model(encoder, in_ch)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()