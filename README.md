# SNN_Research_Project

Три слабо связанных трека в одном репозитории:

- `theory/` — литературная база (аннотированные источники, PDF, два research-плана).
- `snn_cnn_memory_subspace/` — гибридная SNN-CNN модель (SpikingJelly + PyTorch),
  инженерный план про communication subspace на границе SNN/ANN
  при ограниченной памяти GPU (см. `plan.txt`).
- `prosthetic_hand/` — CAD-модель протеза кисти (CadQuery) + кинематическая
  3D-визуализация (matplotlib).

## Перенос на новый ПК

Репозиторий НЕ содержит: виртуальное окружение, датасеты, скачанные PDF
статей, результаты обучения, сгенерированные STL/GIF/rig-файлы — все они
в `.gitignore`, потому что либо слишком большие (PDF ~380МБ, venv ~5.8ГБ),
либо тривиально регенерируются скриптами. После `git clone` на новой машине:

### 1. Python-окружение

```
python -m venv .venv
.venv\Scripts\activate          # Windows
```

### 2. Зависимости по трекам

**SNN-CNN (`snn_cnn_memory_subspace/`)** — сначала CUDA-сборка PyTorch
(обычный `pip install torch` даёт CPU-сборку без ускорения на GPU):

```
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r snn_cnn_memory_subspace/requirements.txt
```

Без NVIDIA GPU/CUDA — пропустить `--index-url` и поставить обычные
`torch==2.6.0 torchvision==0.21.0` (скрипты сами переключаются на CPU через
`torch.cuda.is_available()`, но тренировка будет намного медленнее).

**Protez кисти (`prosthetic_hand/`)**:

```
pip install -r prosthetic_hand/requirements.txt
```

**Будущий биологический трек** (`theory/research_plan_communication_subspace.md`,
код пока не написан) — когда понадобится:

```
pip install brian2==2.10.1 nengo==4.1.0 pyNN==0.13.0 scikit-learn pingouin umap-learn statsmodels
```

### 3. Регенерация данных, исключённых из git

| Что | Как восстановить |
|---|---|
| MNIST (rate-coded, для M1) | автоматически при первом запуске `snn_cnn_memory_subspace/src/train_m1.py` |
| N-MNIST (для M3+) | см. инструкцию внутри `snn_cnn_memory_subspace/src/dataset.py::get_datasets` (ручная загрузка Train.zip/Test.zip) |
| PDF статей (`theory/papers/`) | список со ссылками/DOI — `theory/bibliography_links.csv`; скрипт автозагрузки не входит в репозиторий, качать вручную |
| STL протеза (`prosthetic_hand/cad/output/`) | `python prosthetic_hand/cad/generate_hand.py` |
| GIF-анимация (кубики) | `python prosthetic_hand/visualization/animate_hand.py` |
| rig_data.npz + GIF (реалистичный меш) | `python prosthetic_hand/visualization/mesh_model/build_rig.py`, затем `animate_hand_mesh.py` — **см. известный баг ниже, скрипт сейчас крашится** |
| Результаты прогонов (`snn_cnn_memory_subspace/results/`) | `python snn_cnn_memory_subspace/src/train_m1.py` и далее по `plan.txt` |

### Известные незакрытые проблемы на момент переноса

- `snn_cnn_memory_subspace/src/train_m1.py` ещё не был запущен ни разу —
  `results/` пуст, сходимость модели не подтверждена.
- `prosthetic_hand/visualization/mesh_model/animate_hand_mesh.py` сломан:
  `main()` неверно распаковывает `load_rig()` (ожидает 2 значения вместо 6)
  и вызывает `pose_vertices()` с 4 аргументами вместо 8.

Детали по каждому треку — в `snn_cnn_memory_subspace/plan.txt` и
`theory/research_plan_communication_subspace.md`.
