"""
Гибридная SNN-CNN архитектура для M1 (sanity-check) и последующих этапов.

Дизайн границы SNN/ANN: спайковая часть (несколько свёрточных блоков с
LIF-нейронами, multi-step режим SpikingJelly) выдаёт тензор спайков
формы [T, B, C, H, W]. На границе берётся усреднение по времени T
(rate-coded передача) -> статический тензор [B, C, H, W], который далее
проходит через обычные (не спайковые) свёрточные/полносвязные слои.

Это даёт два практических преимущества для дальнейших этапов:
  - boundary_depth становится явным, управляемым параметром архитектуры
    (нужно для M3 — развёртка по позиции границы);
  - активация на границе — это один статический тензор на образец, а не
    последовательность, что сильно упрощает экстракцию признаков для
    RRR-анализа на M2 (не нужно отдельно решать, что делать с временным
    измерением на стороне анализа).
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, surrogate, functional

# Канальная "лестница" блоков. boundary_depth определяет, сколько первых
# блоков — спайковые (SNN), остальные до total_depth — обычные ANN-блоки.
CHANNELS = [2, 16, 32, 64, 64]  # [in_channels, c1, c2, c3, c4]


class HybridSNNCNN(nn.Module):
    def __init__(self, num_classes=10, boundary_depth=2, total_depth=3,
                 input_hw=34):
        """
        boundary_depth: число спайковых свёрточных блоков (SNN-часть).
        total_depth: общее число свёрточных блоков (SNN + ANN) до классификатора.
        input_hw: размер входного кадра (N-MNIST даёт 34x34).
        """
        super().__init__()
        assert 1 <= boundary_depth <= total_depth <= len(CHANNELS) - 1
        self.boundary_depth = boundary_depth
        self.total_depth = total_depth

        # --- SNN-часть (multi-step, step_mode='m') ---
        snn_blocks = []
        for i in range(boundary_depth):
            snn_blocks += [
                layer.Conv2d(CHANNELS[i], CHANNELS[i + 1], kernel_size=3, padding=1),
                layer.BatchNorm2d(CHANNELS[i + 1]),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode='m'),
                layer.MaxPool2d(2),
            ]
        self.snn = nn.Sequential(*snn_blocks)
        functional.set_step_mode(self.snn, step_mode='m')

        # --- ANN-часть (обычные torch-слои, без временного измерения) ---
        ann_blocks = []
        for i in range(boundary_depth, total_depth):
            ann_blocks += [
                nn.Conv2d(CHANNELS[i], CHANNELS[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(CHANNELS[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.ann = nn.Sequential(*ann_blocks) if ann_blocks else nn.Identity()

        # размер пространственной карты после всех пулингов
        hw_after = input_hw // (2 ** total_depth)
        feat_dim = CHANNELS[total_depth] * hw_after * hw_after
        self.classifier = nn.Linear(feat_dim, num_classes)

        self.last_boundary_features = None  # для будущей экстракции в M2

    def forward(self, x_seq):
        """x_seq: [T, B, 2, H, W] — последовательность кадров событий."""
        functional.reset_net(self.snn)

        spk_seq = self.snn(x_seq)          # [T, B, C, H', W']
        boundary = spk_seq.mean(dim=0)      # усреднение по времени -> [B, C, H', W']
        self.last_boundary_features = boundary.detach()

        feat = self.ann(boundary)           # [B, C2, H'', W'']
        feat = torch.flatten(feat, start_dim=1)
        logits = self.classifier(feat)
        return logits


def build_model(num_classes=10, boundary_depth=2, total_depth=3, input_hw=34):
    return HybridSNNCNN(num_classes=num_classes, boundary_depth=boundary_depth,
                         total_depth=total_depth, input_hw=input_hw)
