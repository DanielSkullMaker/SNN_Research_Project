"""
3D-анимация протеза кисти: поочерёдное сгибание пальцев.

Это НЕ рендер точной CAD-геометрии из cad/generate_hand.py (там пальцы —
цельные детали с гибкими перемычками на 2 шарнирах; физическая деталь
пока не имеет шарнира у основания). Здесь — отдельная кинематическая
модель с ТРЕМЯ суставами на палец (MCP у основания + PIP + DIP), как у
живой руки, углы которых считаются от одного параметра "сжатие" s∈[0,1]
по биомеханическим соотношениям:

    MCP ≈ 85°·s      (основной вклад в сжатие кулака)
    PIP ≈ 100°·s      (самый подвижный сустав)
    DIP ≈ (2/3)·PIP   (механически связан с PIP у сухожильных приводов)

Большой палец - 2 сустава: CMC (у основания, наклонная ось - совмещает
оппозицию и сгибание) + IP (сгибание кончика). См. README/чат-лог этой
сессии за источники по диапазонам движения и обоснование формул.

ВАЖНО: эта кинематика (в первую очередь MCP) сейчас ШИРЕ, чем способна
дать физическая CAD-деталь (там гибкий шарнир есть только между
фалангами, не у основания пальца) - т.е. это целевая/демонстрационная
кинематика, а не точный повтор текущего CAD. Чтобы протез физически
мог их воспроизвести, в generate_hand.py нужно добавить третий
flex-joint у основания каждого пальца (отдельная задача).

Логика анимации: пальцы сгибаются по очереди (один "импульс" сгиб-разгиб
на каждый палец), что имитирует независимое управление пальцами от
EMG-классификатора жестов через Arduino.

Запуск:  python animate_hand.py
Результат: output/hand_animation.gif
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cad"))
import generate_hand as gh  # noqa: E402  (повторно используем параметры CAD-модели)

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Параметры анимации
# ---------------------------------------------------------------------------
FPS = 25
TOTAL_DURATION = 7.5          # секунд
# Порядок и временные окна "импульса" сгиб-разгиб для каждого пальца (сек)
FINGER_ORDER = ["index", "middle", "ring", "pinky", "thumb"]
WINDOW = 1.6                   # длительность одного импульса сгибания, сек
OVERLAP = 0.35                 # перекрытие между соседними импульсами, сек

SPLAY_DEG = {
    # Разведение пальцев в горизонтальной плоскости (вокруг Z), веером
    # НАРУЖУ от средней линии ладони. Знак проверен численно (не на глаз):
    # rot_z(splay) даёт направление пальца local_y_x = -sin(splay); чтобы
    # мизинец (база x=-27, левый край) расходился ДАЛЬШЕ в -X, нужен
    # ПОЛОЖИТЕЛЬНЫЙ splay, а указательному (база x=+27, правый край) -
    # ОТРИЦАТЕЛЬНЫЙ, чтобы уйти в +X. Раньше знаки были перепутаны (минус
    # вместо плюса и наоборот) - пальцы сходились к центру и перекрещивались
    # вместо того чтобы расходиться, что и было видно на виде сверху.
    "pinky": 46.0,
    "ring": 18.0,
    "middle": -5.0,
    "index": -30.0,
}

# Максимальные углы суставов (градусы) при полном сжатии (s=1.0).
#
# Сверено с тремя независимыми источниками (см. research в этой сессии):
#   - чистая анатомия человека: MCP≈85°, PIP≈105°(max), DIP≈69°
#   - SCHUNK SVH (github.com/dexsuite/dex-urdf, реальная tendon-driven
#     протезная/робо-кисть - механически ближе всего к нашей конструкции):
#     MCP≈46-56°, "Distal"(≈PIP)≈76°, последний сустав(≈DIP)≈80°
#   - Shadow Hand (github.com/dexsuite/dex-urdf, эталонная исследовательская
#     кисть): MCP=PIP=DIP=90° (инженерное решение - одинаковый привод и
#     диапазон на каждый сустав, а не повтор человеческих пропорций)
# Вывод: реальные построенные кисти НЕ повторяют анатомические пропорции
# один в один - они выбирают диапазоны под свой механизм привода. Так как
# наша конструкция - одно сухожилие на палец (как у SCHUNK, а не полностью
# актуируемая Shadow), MCP снижен относительно идеальной анатомии (железу
# с одним тросом труднее дать большой ход именно у основания), а PIP/DIP
# держим ближе к идеализированному Shadow-диапазону с сохранением
# анатомического порядка PIP > DIP.
PIP_MAX = 90.0
JOINT_MAX_DEG = {
    "regular": (65.0, PIP_MAX, PIP_MAX * 2.0 / 3.0),  # MCP, PIP, DIP
    "thumb": (35.0, 80.0),                             # CMC, IP
}
# Небольшой "базовый" изгиб в покое (расслабленная кисть не держит пальцы
# идеально прямыми и параллельными - они слегка согнуты и сходятся)
REST_CURL = 0.04  # доля от JOINT_MAX_DEG, присутствующая уже при s=0


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_axis(axis, theta):
    """Поворот на угол theta вокруг произвольной оси (формула Родрига).
    axis - 3-вектор в ЛОКАЛЬНЫХ координатах сустава, не обязательно
    совпадающий с осями X/Y/Z (нужно для наклонной оси CMC-сустава
    большого пальца: см. research — реальные протезы с 1 DOF у большого
    пальца наклоняют ось вращения вперёд, а не держат её строго
    вертикальной/горизонтальной)."""
    axis = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def smooth_pulse(t, t_start, duration, max_val):
    """Гладкий импульс 0 -> max_val -> 0 на интервале [t_start, t_start+duration]."""
    if t < t_start or t > t_start + duration:
        return 0.0
    phase = (t - t_start) / duration
    return max_val * 0.5 * (1.0 - np.cos(2 * np.pi * phase))


class Finger:
    def __init__(self, name, seg_lengths, width, height, base_pos, splay_deg=0.0,
                 base_pitch_deg=0.0, flex_axis="x", flex_sign=-1.0):
        self.name = name
        self.seg_lengths = seg_lengths
        self.width = width
        self.height = height
        self.base_pos = np.array(base_pos, dtype=float)
        # базовая (неподвижная) ориентация пальца относительно ладони:
        # local_y смотрит "вперёд" (вдоль ладони), повёрнута на splay вокруг Z
        # и на base_pitch вокруг X (для большого пальца - наклон）
        self.R_base = rot_z(np.radians(splay_deg)) @ rot_x(np.radians(base_pitch_deg))
        # ОДИН сустав ПЕРЕД каждым сегментом (сустав[0]=MCP перед проксимальной
        # фалангой, сустав[1]=PIP, сустав[2]=DIP для длинных пальцев; для
        # большого пальца сустав[0]=CMC, сустав[1]=IP). Раньше основание
        # пальца было жёстко приклеено к ладони без сустава - это и была
        # главная претензия к анимации (палец не дотягивался до ладони).
        self.n_joints = len(seg_lengths)
        # ось сгибания у КАЖДОГО сустава - свой вектор в ЛОКАЛЬНОЙ системе
        # координат, накопленной к этому суставу (формула Родрига,
        # rot_axis()). 'x'=(1,0,0): сустав гнётся к ладонной стороне
        # (обычные MCP/PIP/DIP). Большой палец передаёт СПИСОК из двух
        # разных осей: наклонную для CMC (оппозиция+сгибание) и 'x' для IP.
        if isinstance(flex_axis, (str, tuple)) and not isinstance(flex_axis, list):
            flex_axis = [flex_axis] * self.n_joints
        self.flex_axis = [self._to_unit_vec(a) for a in flex_axis]
        if not isinstance(flex_sign, (list, tuple)):
            flex_sign = [flex_sign] * self.n_joints
        self.flex_sign = list(flex_sign)

    @staticmethod
    def _to_unit_vec(a):
        if isinstance(a, str):
            a = (1.0, 0.0, 0.0) if a == "x" else (0.0, 0.0, 1.0)
        v = np.array(a, dtype=float)
        return v / np.linalg.norm(v)

    def joint_angles(self, t, schedule):
        """Углы в суставах на момент времени t.

        Берём параметр сжатия s=schedule(t) ∈ [0,1] и переводим его в
        абсолютные углы суставов по анатомическим максимумам из
        JOINT_MAX_DEG (MCP≈85°, PIP≈100°, DIP=2/3·PIP для длинных пальцев;
        CMC/IP для большого пальца), плюс небольшой базовый изгиб в покое
        (REST_CURL), чтобы кисть не выглядела как полностью разогнутая
        "доска" даже когда ни один палец не активен.
        """
        s = schedule.get(self.name, lambda _t: 0.0)(t)
        max_degs = JOINT_MAX_DEG["thumb" if self.name == "thumb" else "regular"]
        max_degs = max_degs[: self.n_joints]
        return [d * (REST_CURL + (1.0 - REST_CURL) * s) for d in max_degs]

    def forward_kinematics(self, angles_deg):
        """Возвращает список (corners_global) для каждого сегмента-бокса."""
        boxes = []
        R = self.R_base.copy()
        pos = self.base_pos.copy()
        for i, length in enumerate(self.seg_lengths):
            theta = np.radians(self.flex_sign[i] * angles_deg[i])
            R = R @ rot_axis(self.flex_axis[i], theta)
            boxes.append(self._box_corners(pos, R, length))
            pos = pos + R[:, 1] * length
        return boxes

    def _box_corners(self, pos, R, length):
        w, h = self.width, self.height
        local = np.array([
            [-w / 2, 0, -h / 2], [w / 2, 0, -h / 2], [w / 2, length, -h / 2], [-w / 2, length, -h / 2],
            [-w / 2, 0, h / 2], [w / 2, 0, h / 2], [w / 2, length, h / 2], [-w / 2, length, h / 2],
        ])
        return pos + local @ R.T


def box_faces(corners):
    idx = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4],
    ]
    return [[corners[i] for i in face] for face in idx]


def build_fingers():
    fingers = []
    for name, lengths, width, height, x in gh.FINGERS:
        # z поднят на верхнюю поверхность ладони (а не в её толщу), чтобы
        # шарнир MCP вращался от края ладони, а не из середины её объёма
        base_pos = (x, gh.PALM_LENGTH, gh.PALM_THICKNESS / 2.0)
        fingers.append(Finger(name, lengths, width, height, base_pos,
                               splay_deg=SPLAY_DEG.get(name, 0.0)))

    # Большой палец: ИСПРАВЛЕНО положение стороны. Раньше база была на
    # x=-PALM_WIDTH/2 (та же, локтевая/ulnar сторона, что и мизинец x=-27) -
    # анатомически неверно: большой палец должен быть на ЛУЧЕВОЙ (radial)
    # стороне, рядом с указательным (x=+27 после фикса splay выше). Это и
    # давало вид "растёт из середины ладони, смотрит к локтевому краю".
    # z по-прежнему на ладонной (-Z) стороне, к камере.
    thumb_base = (
        gh.PALM_WIDTH / 2.0 + 2.0,
        gh.PALM_LENGTH * 0.2,
        -(gh.PALM_THICKNESS / 2.0 + gh.THUMB["height"] / 2.0),
    )
    # Направление в покое: ~45° между "вперёд" (как у пальцев) и "наружу"
    # (вбок от ладони) - анатомический ориентир "40-50° к остальным пальцам".
    # Знак splay пересчитан под НОВУЮ (+X) сторону базы: см. комментарий к
    # SPLAY_DEG выше про конвенцию знака (positive splay -> -X, значит для
    # фанирования в +X здесь нужен отрицательный угол).
    # CMC (сустав[0]): вращение вокруг вертикальной оси (Z) у основания -
    # знак подобран и проверен численно под НОВОЕ положение базы (см. сессию)
    # так, чтобы кончик при сгибании шёл ВПЕРЁД к пальцам (Y растёт) и НЕ
    # уходил дальше от центра по X. IP (сустав[1]): сгибание кончика к
    # ладонной стороне, как PIP/DIP остальных пальцев.
    fingers.append(Finger("thumb", gh.THUMB["segments"], gh.THUMB["width"], gh.THUMB["height"],
                           thumb_base, splay_deg=-45.0, base_pitch_deg=0.0,
                           flex_axis=["z", "x"], flex_sign=[1.0, -1.0]))
    return fingers


def build_schedule():
    """Временные окна импульса сгиб-разгиб для каждого пальца (с перекрытием)."""
    schedule = {}
    t0 = 0.3
    step = WINDOW - OVERLAP
    for i, name in enumerate(FINGER_ORDER):
        start = t0 + i * step
        schedule[name] = (lambda t, s=start: smooth_pulse(t, s, WINDOW, 1.0))
    return schedule


def palm_faces():
    w, l, th = gh.PALM_WIDTH, gh.PALM_LENGTH, gh.PALM_THICKNESS
    corners = np.array([
        [-w / 2, 0, -th / 2], [w / 2, 0, -th / 2], [w / 2, l, -th / 2], [-w / 2, l, -th / 2],
        [-w / 2, 0, th / 2], [w / 2, 0, th / 2], [w / 2, l, th / 2], [-w / 2, l, th / 2],
    ])
    return box_faces(corners)


def forearm_faces():
    w, th = gh.FOREARM_WIDTH, gh.FOREARM_THICK
    length = 90.0
    corners = np.array([
        [-w / 2, -length, -th / 2], [w / 2, -length, -th / 2], [w / 2, 0, -th / 2], [-w / 2, 0, -th / 2],
        [-w / 2, -length, th / 2], [w / 2, -length, th / 2], [w / 2, 0, th / 2], [-w / 2, 0, th / 2],
    ])
    return box_faces(corners)


def main():
    fingers = build_fingers()
    schedule = build_schedule()
    n_frames = int(TOTAL_DURATION * FPS)

    fig = plt.figure(figsize=(6, 9))  # портретный формат под вертикальную кисть
    ax = fig.add_subplot(111, projection="3d")

    x_range = (-55, 55)
    y_range = (-95, 195)
    z_range = (-75, 55)

    def set_equal_axes():
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_zlim(*z_range)
        try:
            ax.set_box_aspect((
                x_range[1] - x_range[0],
                y_range[1] - y_range[0],
                z_range[1] - z_range[0],
            ))
        except AttributeError:
            pass

    static_faces = palm_faces() + forearm_faces()

    def render(t):
        ax.cla()
        set_equal_axes()
        ax.set_axis_off()
        # Кисть стоит вертикально (запястье снизу, пальцы сверху), ладонь
        # обращена к камере: elev≈-90 смотрит почти строго вдоль оси Z снизу
        # вверх (палмарная/грузовая сторона - это -Z), azim=90 разворачивает
        # длинную ось руки (Y) в вертикаль на экране.
        ax.view_init(elev=-88, azim=90)
        ax.set_title(f"EMG-протез: поочерёдное сгибание пальцев  (t = {t:4.2f} c)")

        # shade=True: простое затенение граней по нормали. Без него плоская
        # заливка заставляет грани разной ориентации выглядеть "по-разному
        # высокими" - два идеально прямых, соосных сегмента визуально
        # читаются как изломанные, хотя геометрически они строго прямые
        # (см. численную/wireframe-проверку стыка сегментов).
        collections = []
        static = Poly3DCollection(static_faces, facecolors="#b0b0b8", edgecolors="k",
                                   linewidths=0.3, alpha=0.95, shade=True)
        ax.add_collection3d(static)
        collections.append(static)

        for finger in fingers:
            angles = finger.joint_angles(t, schedule)
            boxes = finger.forward_kinematics(angles)
            active = schedule.get(finger.name, lambda _t: 0.0)(t) > 0.02
            color = "#e07a3f" if active else "#d8c4a0"
            for corners in boxes:
                pc = Poly3DCollection(box_faces(corners), facecolors=color,
                                      edgecolors="k", linewidths=0.3, alpha=0.98,
                                      shade=True)
                ax.add_collection3d(pc)
                collections.append(pc)
        return collections

    def update(frame_idx):
        t = frame_idx / FPS
        return render(t)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False)

    out_path = os.path.join(OUT_DIR, "hand_animation.gif")
    print(f"Рендер {n_frames} кадров -> {out_path} ...")
    anim.save(out_path, writer=animation.PillowWriter(fps=FPS))
    print("Готово:", out_path)


if __name__ == "__main__":
    main()
