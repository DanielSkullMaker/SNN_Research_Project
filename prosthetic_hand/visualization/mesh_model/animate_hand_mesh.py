"""
Анимация РЕАЛЬНОЙ 3D-модели кисти (найдена в интернете, см. build_rig.py),
привязанной (rigid skinning) к тому же скелету MCP/PIP/DIP + CMC/IP, что и
в ../animate_hand.py. Это ОТДЕЛЬНЫЙ, более сложный путь визуализации -
исходный animate_hand.py (кубики) не изменён и продолжает работать как
раньше.

Кинематика (joint_angles, forward_kinematics, rot_axis, формулы
MCP/PIP/DIP, расписание сгибания пальцев) ЦЕЛИКОМ переиспользуется из
animate_hand.py через наследование (MeshFinger(ah.Finger)) - то есть все
фиксы из предыдущих итераций (знак splay, 3 сустава, сторона большого
пальца и т.д.) автоматически действуют и здесь.

Запуск:
  1) python build_rig.py          (один раз - строит assets/rig_data.npz)
  2) python animate_hand_mesh.py  (рендерит output/hand_mesh_animation.gif)
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import animate_hand as ah  # noqa: E402  (переиспользуем кинематику, НЕ меняя файл)

HERE = os.path.dirname(__file__)
RIG_PATH = os.path.join(HERE, "assets", "rig_data.npz")
OUT_DIR = os.path.join(HERE, "..", "output")
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 25
SKIN_COLOR = "#d9b99b"
ACTIVE_COLOR = "#e07a3f"
LIGHT_DIR = np.array([0.3, -0.5, 0.8])
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


def shaded_face_colors(triangles, base_rgba):
    """Простое плоское затенение по нормали грани (Lambert, без спекуляра).
    matplotlib's Poly3DCollection(..., shade=True) падает с ValueError при
    массиве РАЗНЫХ цветов на грань (баг в этой версии matplotlib, см.
    сессию) - поэтому считаем яркость вручную и просто не передаём
    shade=True."""
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    n = n / norm
    brightness = np.clip(np.abs(n @ LIGHT_DIR), 0.35, 1.0)
    out = base_rgba.copy()
    out[:, :3] *= brightness[:, None]
    return out


class MeshFinger(ah.Finger):
    """Как ah.Finger, но R_base задаётся готовой матрицей (из скан-меша),
    а не через splay/pitch углы. joint_angles/forward_kinematics - те же
    (унаследованы без изменений)."""

    def __init__(self, name, seg_lengths, base_pos, R_base, flex_axis="x", flex_sign=-1.0):
        self.name = name
        self.seg_lengths = list(seg_lengths)
        self.base_pos = np.array(base_pos, dtype=float)
        self.R_base = np.array(R_base, dtype=float)
        self.n_joints = len(self.seg_lengths)
        if isinstance(flex_axis, (str, tuple)) and not isinstance(flex_axis, list):
            flex_axis = [flex_axis] * self.n_joints
        self.flex_axis = [ah.Finger._to_unit_vec(a) for a in flex_axis]
        if not isinstance(flex_sign, (list, tuple)):
            flex_sign = [flex_sign] * self.n_joints
        self.flex_sign = list(flex_sign)

    def segment_transforms(self, angles_deg):
        """(pos, R) для каждого сегмента - то же накопление, что в
        forward_kinematics, но возвращает трансформы, а не углы box'а
        (нужно для позиционирования вершин реального меша)."""
        transforms = []
        R = self.R_base.copy()
        pos = self.base_pos.copy()
        for i, length in enumerate(self.seg_lengths):
            theta = np.radians(self.flex_sign[i] * angles_deg[i])
            R = R @ ah.rot_axis(self.flex_axis[i], theta)
            transforms.append((pos.copy(), R.copy()))
            pos = pos + R[:, 1] * length
        return transforms


# Та же ось/знак CMC и IP для большого пальца, что в animate_hand.py (см.
# комментарии там про геометрически проверенный выбор знака).
THUMB_FLEX_AXIS = ["z", "x"]
THUMB_FLEX_SIGN = [1.0, -1.0]


def load_rig():
    d = np.load(RIG_PATH)
    finger_names = [str(n) for n in d["finger_names"]]
    fingers = {}
    for name in finger_names:
        base_pos = d[f"base_pos__{name}"]
        R_base = d[f"R_base__{name}"]
        seg_lengths = d[f"seg_lengths__{name}"]
        if name == "thumb":
            fingers[name] = MeshFinger(name, seg_lengths, base_pos, R_base,
                                        flex_axis=THUMB_FLEX_AXIS, flex_sign=THUMB_FLEX_SIGN)
        else:
            fingers[name] = MeshFinger(name, seg_lengths, base_pos, R_base,
                                        flex_axis="x", flex_sign=-1.0)

    bone_names = [str(n) for n in d["bone_names"]]
    bone_bind_pos = d["bone_bind_pos"]
    bone_bind_R = d["bone_bind_R"]
    # предвычисленный локальный offset КАЖДОЙ вершины относительно КАЖДОЙ
    # кости (не только "своей") - нужно для LBS: вершина у стыка получает
    # вклад от двух костей одновременно, значит нужны оба offset'а.
    verts = d["verts"]
    local_offsets_per_bone = np.einsum(
        "vj,bjk->bvk", verts, bone_bind_R
    ) - np.einsum("bj,bjk->bk", bone_bind_pos, bone_bind_R)[:, None, :]
    # (эквивалент (verts - bind_pos[b]) @ bind_R[b] для каждой кости b)

    return d, fingers, bone_names, bone_bind_pos, bone_bind_R, local_offsets_per_bone


def pose_vertices(d, fingers, bone_names, bone_bind_pos, bone_bind_R,
                   local_offsets_per_bone, schedule, t):
    verts = d["verts"]
    weights = d["weights"]  # (n_verts, n_bones)
    owner = d["owner"]
    finger_names = list(fingers.keys())

    posed = np.zeros_like(verts)
    bone_pos = {"palm": (np.zeros(3), np.eye(3))}
    bone_index = {name: i for i, name in enumerate(bone_names)}

    # ладонь (кость 0) неподвижна
    w_palm = weights[:, 0]
    posed += w_palm[:, None] * local_offsets_per_bone[0]

    active_finger_idx = -2
    for fi, name in enumerate(finger_names):
        finger = fingers[name]
        angles = finger.joint_angles(t, schedule)
        transforms = finger.segment_transforms(angles)
        s = schedule.get(name, lambda _t: 0.0)(t)
        if s > 0.02:
            active_finger_idx = fi
        for seg, (pos, R) in enumerate(transforms):
            bone_i = bone_index[f"{name}:{seg}"]
            w = weights[:, bone_i]
            nz = w > 1e-4
            if not np.any(nz):
                continue
            contrib = local_offsets_per_bone[bone_i][nz] @ R.T + pos
            posed[nz] += w[nz, None] * contrib
    return posed, active_finger_idx, owner


def main():
    d, fingers = load_rig()
    faces = d["faces"]
    vf = d["vertex_finger"]
    finger_names = list(fingers.keys())
    schedule = ah.build_schedule()
    n_frames = int(ah.TOTAL_DURATION * FPS)

    face_finger = vf[faces[:, 0]]  # индекс пальца для каждой грани (-1 = ладонь)

    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111, projection="3d")

    verts0 = d["verts"]
    mn, mx = verts0.min(axis=0), verts0.max(axis=0)
    pad = 0.3
    x_range = (mn[0] - pad, mx[0] + pad)
    y_range = (mn[1] - pad, mx[1] + pad)
    z_range = (mn[2] - pad, mx[2] + pad)

    def render(t):
        ax.cla()
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.set_zlim(*z_range)
        ax.set_box_aspect((x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]))
        ax.set_axis_off()
        # Длина кисти у этой модели идёт вдоль X. Запястье СНИЗУ, пальцы
        # вверх (подобрано эмпирически под новую модель-кожу).
        ax.view_init(elev=55, azim=180)
        ax.set_title(f"EMG-протез (реалистичная модель): t = {t:4.2f} c")

        posed, active_idx = pose_vertices(d, fingers, schedule, t)
        colors = np.where(face_finger == active_idx, ACTIVE_COLOR, SKIN_COLOR)
        base_rgba = np.array([mcolors.to_rgba(c) for c in colors])
        tris = posed[faces]
        fc = shaded_face_colors(tris, base_rgba)
        pc = Poly3DCollection(tris, facecolors=fc, edgecolors="none")
        ax.add_collection3d(pc)
        return [pc]

    def update(frame_idx):
        return render(frame_idx / FPS)

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / FPS, blit=False)
    out_path = os.path.join(OUT_DIR, "hand_mesh_animation.gif")
    print(f"Рендер {n_frames} кадров -> {out_path} ...")
    anim.save(out_path, writer=animation.PillowWriter(fps=FPS))
    print("Готово:", out_path)


if __name__ == "__main__":
    main()
