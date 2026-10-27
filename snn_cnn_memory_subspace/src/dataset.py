"""
Загрузка N-MNIST через SpikingJelly.

N-MNIST (Orchard et al. 2015) — событийная версия MNIST, записанная
саккадирующей ATIS-камерой перед экраном. SpikingJelly хранит данные в
виде .npz (события) и умеет на лету нарезать их на T кадров (integrate
events into frames) — именно это нужно для подачи в multi-step SNN.

Автоматическая загрузка исходного датасета у SpikingJelly не всегда
работает (оригинальный хостинг иногда недоступен для прямой ссылки) —
при ошибке скрипт печатает инструкцию для ручного скачивания.
"""

import os
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.activation_based import encoding

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "NMNIST")
MNIST_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MNIST_raw")


def get_datasets(T=8, split_by="number"):
    """
    T: число кадров, на которые нарезается событийный поток каждого образца
       (это и есть параметр "timesteps" из плана — ключевая ось памяти для M4).
    split_by: 'number' — фиксированное число событий на кадр,
              'time' — фиксированный временной интервал на кадр.
    """
    os.makedirs(DATA_ROOT, exist_ok=True)
    try:
        train_set = NMNIST(root=DATA_ROOT, train=True, data_type="frame",
                           frames_number=T, split_by=split_by)
        test_set = NMNIST(root=DATA_ROOT, train=False, data_type="frame",
                          frames_number=T, split_by=split_by)
    except Exception as e:
        raise RuntimeError(
            "Не удалось автоматически подготовить N-MNIST.\n"
            f"Исходная ошибка: {e}\n\n"
            "Ручной способ:\n"
            "1. Скачать Train.zip и Test.zip с https://www.garrickorchard.com/datasets/n-mnist\n"
            f"2. Распаковать в {os.path.join(DATA_ROOT, 'download')}\n"
            "   (создать подпапки 'Train' и 'Test' с .bin файлами событий)\n"
            "3. Повторно запустить этот скрипт — SpikingJelly сам сконвертирует\n"
            "   события в формат .npz и затем нарежет на кадры."
        ) from e
    return train_set, test_set


class RateCodedMNIST(Dataset):
    """
    Обёртка над обычным MNIST: каждое изображение кодируется в T кадров
    спайков через Poisson rate-кодирование (интенсивность пуассоновского
    процесса = яркость пикселя). Используется ТОЛЬКО для M1 sanity-check,
    чтобы не блокироваться на ручном скачивании N-MNIST (хостится на
    Google Drive/Dropbox без прямой ссылки на файл, авто-загрузка
    SpikingJelly в принципе не поддерживается для этого датасета).

    Возвращает тензор формы [T, 1, 28, 28] — совместимо по интерфейсу
    с тем, что отдаёт NMNIST (там [T, 2, 34, 34]), с поправкой на то, что
    модель (model.py) ожидает in_channels=2: для rate-coded MNIST нужно
    либо передавать in_channels=1 в build_model, либо дублировать канал.
    Здесь дублируем канал до 2, чтобы модель/код M1 не менять.
    """

    def __init__(self, train, T=8):
        self.T = T
        self.base = torchvision.datasets.MNIST(
            root=MNIST_ROOT, train=train, download=True,
            transform=T_compose(),
        )
        self.encoder = encoding.PoissonEncoder()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]          # img: [1, 28, 28], значения в [0,1]
        frames = torch.stack([self.encoder(img) for _ in range(self.T)], dim=0)
        frames = frames.repeat(1, 2, 1, 1)    # [T, 1, 28, 28] -> [T, 2, 28, 28]
        return frames, label


def T_compose():
    return T.Compose([T.ToTensor()])


def get_mnist_datasets(T_steps=8):
    """Датасет для M1: rate-coded MNIST, авто-скачивается без ручных шагов."""
    train_set = RateCodedMNIST(train=True, T=T_steps)
    test_set = RateCodedMNIST(train=False, T=T_steps)
    return train_set, test_set


if __name__ == "__main__":
    print("Проверка rate-coded MNIST (используется в M1)...")
    train_set, test_set = get_mnist_datasets(T_steps=8)
    print(f"Train: {len(train_set)} образцов, Test: {len(test_set)} образцов")
    x, y = train_set[0]
    print(f"Форма одного образца (кадры спайков): {x.shape}, метка: {y}")
