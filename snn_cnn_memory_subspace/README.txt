Проект: "Анализ коммуникационного подпространства на границе спайковой
и свёрточной частей сети в условиях ограниченной памяти при обучении"

Полный план исследования и статьи: plan.txt (этапы M1-M7)

Структура:
  src/            - весь код
    model.py      - гибридная SNN-CNN архитектура (управляемая граница)
    dataset.py    - загрузка N-MNIST через SpikingJelly
    train_m1.py   - ЭТАП M1: минимальный гибрид, sanity-check
  data/           - датасеты (N-MNIST и др.), не коммитить в git
  results/        - результаты прогонов (json/csv/графики)
  requirements.txt - зависимости (torch, torchvision, spikingjelly, sklearn)

Запуск M1:
  .venv/Scripts/python.exe src/train_m1.py
