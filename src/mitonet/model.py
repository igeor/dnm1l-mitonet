"""EfficientNetB0 classifier used for both the FT and CFT strategies.

The same model class is reused across curriculum stages: images of any
input resolution are adaptively pooled to `image_size` before the backbone,
so a checkpoint trained at one resolution can be loaded and continued at
the next.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import models
from torchvision.models import get_model_weights


class EfficientNetClassifier(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 3,
        image_size: int = 1024,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dense_layers=None,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.model_name = model_name
        self.image_size = image_size
        dense_layers = dense_layers or []

        model_fn = getattr(models, model_name)
        weights = get_model_weights(model_fn).DEFAULT if pretrained else None
        self.model = model_fn(weights=weights)

        if freeze_backbone and pretrained:
            for param in self.model.features.parameters():
                param.requires_grad = False

        # Rebuild the classifier head for `num_classes`, with optional
        # extra dense layers.
        in_features = self.model.classifier[-1].in_features
        layers = [torch.nn.Dropout(p=dropout)]
        for hidden_units in dense_layers:
            layers += [
                torch.nn.Linear(in_features, hidden_units),
                torch.nn.ReLU(inplace=False),
                torch.nn.Dropout(p=dropout),
            ]
            in_features = hidden_units
        layers.append(torch.nn.Linear(in_features, num_classes))
        self.model.classifier = torch.nn.Sequential(*layers)

        if not pretrained:
            for m in self.model.classifier.modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (self.image_size, self.image_size))
        return self.model(x)

    def fit(self, train_loader, optimizer, criterion, device="cuda"):
        self.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = self(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader, device="cuda"):
        self.eval()
        real_labels, pred_probs = [], []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_pred = self(x)
            pred_probs.append(F.softmax(y_pred, dim=1))
            real_labels.append(F.one_hot(y, num_classes=y_pred.shape[1]).float())
        return torch.cat(real_labels, dim=0), torch.cat(pred_probs, dim=0)
