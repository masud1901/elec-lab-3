"""The networks.

- alexnet:   the network used in the AffectNet paper
- resnet18:  a newer, smaller and usually more accurate network (default)
- small_cnn: a compact CNN trained from scratch, to show how much ImageNet pretraining helps
- vit_face:  a Vision Transformer already pretrained on facial expressions (exercise 4d)

In each case the classifier is replaced with a new layer that outputs one score per expression.

Two options exist because FER2013 images are only 48x48:

- ``small_stem=True`` (resnet18) replaces the 7x7/stride-2 convolution with a 3x3/stride-1 convolution.
  The max-pool then takes 48x48 to 24x24 (instead of 6x6 with the standard stem), so the small image keeps
  useful detail without the cost of running the whole first block at 48x48. Use it with ``--img-size 48``.
- ``vit_face`` is a ViT, so it wants 224x224 (or 112 with interpolated position embeddings); it cannot be fed 48x48.
"""
import torch.nn as nn
import torchvision

FACE_VIT_CHECKPOINT = "trpakov/vit-face-expression"   # ViT-base fine-tuned on FER2013 expressions


def build_model(name, num_classes, pretrained=True, small_stem=False, freeze_encoder=False):
    tvm = torchvision.models
    if name == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        if small_stem:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # the max-pool stays: 48x48 -> 24x24, which is what a 96px ImageNet image would look like
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    elif name == "alexnet":
        model = tvm.alexnet(weights=tvm.AlexNet_Weights.DEFAULT if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "small_cnn":
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        model = nn.Sequential(block(3, 64), block(64, 128), block(128, 256), block(256, 512),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, num_classes))
    elif name == "efficientnet_b0":
        model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif name == "vit_face":
        model = ViTExpression(num_classes, pretrained=pretrained, freeze_encoder=freeze_encoder)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


class ViTExpression(nn.Module):
    """ViT-base whose encoder was pretrained on facial expressions; a fresh head predicts our classes.

    The Hugging Face checkpoint (trpakov/vit-face-expression) is a 7-class FER2013 classifier. We keep its
    encoder and replace its 7-way head, so the features already know what a face and an expression look like.
    Position embeddings are interpolated, so the encoder also accepts 112x112 (much faster on a small GPU).
    """

    def __init__(self, num_classes, pretrained=True, freeze_encoder=False, checkpoint=FACE_VIT_CHECKPOINT):
        super().__init__()
        from transformers import ViTConfig, ViTModel
        if pretrained:
            self.encoder = ViTModel.from_pretrained(checkpoint, add_pooling_layer=False)
        else:
            self.encoder = ViTModel(ViTConfig(add_pooling_layer=False))
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.2), nn.LayerNorm(hidden), nn.Linear(hidden, num_classes))

    def forward(self, x):
        # interpolate_pos_encoding lets the encoder accept any square size, not just the 224 it was trained on
        out = self.encoder(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        return self.head(out)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()      # keep the frozen encoder's batch-norm/layernorm statistics fixed
        return self
