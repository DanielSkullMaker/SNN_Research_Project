"""
ЭТАП M1 — минимальный гибрид SNN-CNN, sanity-check.

Критерий готовности (из plan.txt): модель обучается стабильно, точность
в разумном диапазоне для простого бенчмарка, нет NaN/расходящихся
градиентов.

Фиксированная конфигурация для M1: boundary_depth=2, total_depth=3,
T=8 timesteps, batch_size подбирается по доступной VRAM.
Developments по режиму памяти (T, batch, precision, checkpointing) —
предмет M4, здесь не варьируются.

Запуск:  python src/train_m1.py
"""

import os
import sys
import time
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model               # noqa: E402
from dataset import get_mnist_datasets      # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

T = 8
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
BOUNDARY_DEPTH = 2
TOTAL_DEPTH = 3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    print("Загрузка rate-coded MNIST (M1: быстрый sanity-check без ручной "
          "загрузки N-MNIST; переход на настоящий N-MNIST — в M3+)...")
    train_set, test_set = get_mnist_datasets(T_steps=T)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=0, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, drop_last=False)
    print(f"  train: {len(train_set)}, test: {len(test_set)}")

    model = build_model(num_classes=10, boundary_depth=BOUNDARY_DEPTH,
                         total_depth=TOTAL_DEPTH, input_hw=28).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров в модели: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = []
    nan_detected = False

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for x, y in train_loader:
            # x приходит как [B, T, C, H, W] из SpikingJelly -> нужно [T, B, C, H, W]
            x = x.permute(1, 0, 2, 3, 4).float().to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            if torch.isnan(loss):
                print(f"  !! NaN loss на эпохе {epoch} — обучение остановлено")
                nan_detected = True
                break

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += y.size(0)

        if nan_detected:
            break

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # --- валидация ---
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.permute(1, 0, 2, 3, 4).float().to(device)
                y = y.to(device)
                logits = model(x)
                val_correct += (logits.argmax(dim=1) == y).sum().item()
                val_total += y.size(0)
        val_acc = val_correct / val_total

        peak_mem_mb = (torch.cuda.max_memory_allocated() / 1024**2
                       if device.type == "cuda" else 0.0)
        elapsed = time.time() - t0

        print(f"Эпоха {epoch}/{EPOCHS}: train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} "
              f"peak_VRAM={peak_mem_mb:.1f}MB time={elapsed:.1f}s")

        history.append(dict(epoch=epoch, train_loss=train_loss,
                             train_acc=train_acc, val_acc=val_acc,
                             peak_vram_mb=peak_mem_mb, time_s=elapsed))

    result = dict(
        config=dict(T=T, batch_size=BATCH_SIZE, epochs=EPOCHS, lr=LR,
                    boundary_depth=BOUNDARY_DEPTH, total_depth=TOTAL_DEPTH),
        n_params=n_params,
        nan_detected=nan_detected,
        history=history,
    )
    out_path = os.path.join(RESULTS_DIR, "m1_sanity_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nРезультат сохранён: {out_path}")
    if nan_detected:
        print("ИТОГ M1: ПРОВАЛ — обнаружен NaN в процессе обучения.")
    elif history and history[-1]["val_acc"] > 0.5:
        print(f"ИТОГ M1: УСПЕХ — модель сходится, "
              f"итоговая val_acc={history[-1]['val_acc']:.4f}")
    else:
        print("ИТОГ M1: ТРЕБУЕТ ВНИМАНИЯ — обучение прошло без NaN, "
              "но точность ниже ожидаемого порога (0.5) для N-MNIST.")


if __name__ == "__main__":
    main()
