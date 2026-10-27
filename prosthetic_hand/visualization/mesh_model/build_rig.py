"""
Построение рига для РЕАЛИСТИЧНОЙ (кожа, не скелет) 3D-модели кисти,
найденной в интернете: github.com/sfschouten/3d_mesh_descriptor, hand.obj
(экспорт из Blender). Мэш не был манифолдным (OBJ хранил отдельные
vt/vn на вершину) - склеен через trimesh.merge_vertices(), после чего
децимирован (6000 граней) без артефактов.

Суставы определяются по РЕАЛЬНОЙ геометрии меша: вдоль каждого пальца
считается радиус поперечного сечения, и места локальных МАКСИМУМОВ
радиуса (костяшки) используются как границы сегментов (PIP/DIP, IP у
большого пальца).

ПРИВЯЗКА ВЕРШИН (skinning) - через ГЕОДЕЗИЧЕСКОЕ расстояние по графу
меша от кончика каждого пальца, а не через евклидово расстояние до
трека/bounding-box. Это принципиально: у большого пальца (и у основания
длинных пальцев) поверхность ладони геометрически (по прямой) находится
очень БЛИЗКО к поверхности пальца, хотя топологически (вдоль поверхности)
до неё далеко - нужно обойти всю длину пальца. Евклидово расстояние или
bounding-box это не различают и "воровали" вершины ладони, которые потом
рвались при повороте кости (отчёт пользователя после первой версии рига).
Геодезическое расстояние по рёбрам меша не может "срезать по воздуху" и
поэтому корректно останавливается на границе пальца.

Запуск:  python build_rig.py
"""

import os
import sys

import numpy as np
import trimesh
import networkx as nx
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import animate_hand as ah  # noqa: E402,F401 (оставлено для совместимости/возможного переиспользования)

HERE = os.path.dirname(__file__)
MESH_PATH = os.path.join(HERE, "assets", "hand_skin.obj")
OUT_PATH = os.path.join(HERE, "assets", "rig_data.npz")

# Где у ЭТОЙ модели пальцы уже разошлись на 4 отдельных контура, но ещё не
# слиты с ладонью (см. анализ в сессии: 4 loops уже при x=14, 1 loop при x=13)
PALM_X_CUTOFF = 14.0
TIP_X_MAX = 29.0

# Грубая область для поиска САМОЙ ВЕРШИНЫ-КОНЧИКА большого пальца (НЕ для
# финальной привязки - та идёт через геодезическое расстояние от этой
# вершины, так что неточность границ здесь не критична)
THUMB_SEED_BOUNDS = dict(x_min=-2.0, x_max=12.0, y_min=7.0)

# Множитель запаса к длине пальца при геодезическом отсечении (чуть больше
# 1.0, чтобы захватить костяшку у основания, но не уйти в ладонь)
GEODESIC_MARGIN = 1.12


def mesh_graph(mesh):
    g = mesh.vertex_adjacency_graph
    v = mesh.vertices
    for u, w in g.edges():
        g[u][w]["weight"] = float(np.linalg.norm(v[u] - v[w]))
    return g


def geodesic_distances(g, seed_vertex, n_verts):
    dist = nx.single_source_dijkstra_path_length(g, int(seed_vertex), weight="weight")
    arr = np.full(n_verts, np.inf)
    for k, val in dist.items():
        arr[k] = val
    return arr


def extract_finger_tracks(mesh):
    """Трекинг 4 длинных пальцев срезами (x=const), с радиусом сечения -
    даёт чистый профиль для поиска суставов (см. find_joint_positions)."""
    xs = np.arange(PALM_X_CUTOFF, TIP_X_MAX, 0.25)
    history = {}
    prev_centroids = None
    prev_ids = None
    next_id = 0

    for x_cut in xs:
        sec = mesh.section(plane_origin=[x_cut, 0, 0], plane_normal=[1, 0, 0])
        if sec is None:
            prev_centroids = None
            continue
        loops = sec.discrete
        cents = np.array([loop.mean(axis=0) for loop in loops])
        radii = np.array([np.mean(np.linalg.norm(loop - loop.mean(axis=0), axis=1)) for loop in loops])
        if prev_centroids is None or not prev_ids:
            ids = list(range(next_id, next_id + len(cents)))
            next_id += len(cents)
        else:
            ids = [None] * len(cents)
            used_prev = set()
            for i, c in enumerate(cents):
                d = np.linalg.norm(prev_centroids[:, 1:3] - c[1:3], axis=1)
                for u in used_prev:
                    d[u] = 1e9
                j = int(np.argmin(d))
                if d[j] < 0.8:
                    ids[i] = prev_ids[j]
                    used_prev.add(j)
                else:
                    ids[i] = next_id
                    next_id += 1
        for i, (c, r) in enumerate(zip(cents, radii)):
            history.setdefault(ids[i], []).append((x_cut, c[1], c[2], r))
        prev_centroids = cents
        prev_ids = ids

    tracks = {fid: np.array(pts) for fid, pts in history.items() if len(pts) >= 10}
    return tracks


def identify_fingers(tracks):
    info = [(fid, pts[0, :3], pts[-1, :3], float(np.linalg.norm(pts[-1, :3] - pts[0, :3])))
            for fid, pts in tracks.items()]
    info.sort(key=lambda x: x[1][1])  # по Y основания
    names = ["pinky", "ring", "middle", "index"]
    mapping = {name: fid for name, (fid, *_rest) in zip(names, info)}
    return mapping


def find_joint_positions(radii, min_idx_frac=0.12, max_idx_frac=0.92):
    """2 выраженных локальных максимума радиуса (костяшки) в профиле
    поперечного сечения. Возвращает их индексы в массиве radii."""
    n = len(radii)
    k = max(1, n // 20)
    kernel = np.ones(k) / k
    smooth = np.convolve(radii, kernel, mode="same")
    lo, hi = int(n * min_idx_frac), int(n * max_idx_frac)
    peaks, props = find_peaks(smooth[lo:hi], prominence=0.02)
    peaks = peaks + lo
    if len(peaks) < 2:
        return [int(n * 0.45), int(n * 0.75)]
    prom = props["prominences"]
    order = np.argsort(prom)[::-1][:2]
    return sorted(peaks[order].tolist())


def build_segment_frame(base, tip):
    y_axis = tip - base
    length = float(np.linalg.norm(y_axis))
    y_axis = y_axis / length
    world_z = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(y_axis, world_z)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, y_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, length


def find_tip_seed(mesh, mask):
    """Вершина-кончик пальца: внутри грубой маски - та, что дальше всего
    от центра ладони (запястной области)."""
    v = mesh.vertices
    cand = np.where(mask)[0]
    palm_centroid = v[v[:, 0] < 0].mean(axis=0)
    d = np.linalg.norm(v[cand] - palm_centroid, axis=1)
    return int(cand[np.argmax(d)])


def main():
    print("Загрузка меша:", MESH_PATH)
    mesh = trimesh.load(MESH_PATH)
    verts = mesh.vertices.copy()
    faces = mesh.faces.copy()
    n_verts = len(verts)
    print("  вершин:", n_verts, "граней:", len(faces), "watertight:", mesh.is_watertight)

    print("Граф меша для геодезических расстояний...")
    g = mesh_graph(mesh)

    print("Трекинг 4 пальцев по срезам (профиль радиуса для поиска суставов)...")
    tracks = extract_finger_tracks(mesh)
    mapping = identify_fingers(tracks)

    finger_names = list(mapping.keys()) + ["thumb"]
    rig = {}
    tip_seeds = {}

    for name, fid in mapping.items():
        pts = tracks[fid]
        base, tip = pts[0, :3], pts[-1, :3]
        radii = pts[:, 3]
        joint_idx = find_joint_positions(radii)
        R, total_len = build_segment_frame(base, tip)
        n = len(pts)
        fracs = [joint_idx[0] / n, (joint_idx[1] - joint_idx[0]) / n, 1.0 - joint_idx[1] / n]
        seg_lengths = [f * total_len for f in fracs]
        rig[name] = dict(base_pos=base, R_base=R, seg_lengths=seg_lengths, total_len=total_len)
        print(f"  {name:7s} length={total_len:5.2f} суставы на долях={[round(j/n,2) for j in joint_idx]} "
              f"seg_lengths={[round(s,2) for s in seg_lengths]}")

    # Точная вершина-кончик для каждого длинного пальца - ближайшая к уже
    # найденной 3D-точке tip (а не грубым глобальным поиском, чтобы не
    # перепутать пальцы между собой)
    for name in mapping:
        tip_point = rig[name]["base_pos"] + rig[name]["R_base"][:, 1] * rig[name]["total_len"]
        tip_seeds[name] = int(np.argmin(np.linalg.norm(verts - tip_point, axis=1)))

    print("Поиск большого пальца (срезы вдоль его собственной оси)...")
    thumb_seed_mask = ((verts[:, 0] > THUMB_SEED_BOUNDS["x_min"]) & (verts[:, 0] < THUMB_SEED_BOUNDS["x_max"])
                        & (verts[:, 1] > THUMB_SEED_BOUNDS["y_min"]))
    thumb_tip_seed = find_tip_seed(mesh, thumb_seed_mask)
    tip_seeds["thumb"] = thumb_tip_seed
    tip_point_t = verts[thumb_tip_seed]
    # Грубое направление (кончик -> центр ладони) - достаточно для того,
    # чтобы секущие плоскости шли ПОПЕРЁК пальца, а не вдоль Х, как у
    # остальных 4 пальцев (большой палец стоит под сильным углом к X).
    palm_centroid = verts[verts[:, 0] < 0].mean(axis=0)
    axis_t = tip_point_t - palm_centroid
    axis_t = axis_t / np.linalg.norm(axis_t)

    # Трекинг контура (та же логика, что extract_finger_tracks, но вдоль
    # axis_t от кончика пальца, а не вдоль мировой X от условной линии
    # костяшек) + остановка там, где радиус сечения РЕЗКО (>1.7x) скачет -
    # это и есть переход "палец -> ладонь" (ладонь на порядок шире пальца).
    # Радиус рос ПЛАВНО (2.2 -> 5.3) до t~5-6, потом скачок до 6-8 - видно
    # прямо в данных (см. сессию) - такой же сигнал, какой раньше выделил
    # 4 обычных пальца через количество контуров среза.
    ts = np.arange(0.0, 25.0, 0.4)
    track_list = []
    prev_centroid = None
    for t in ts:
        origin = tip_point_t - axis_t * t
        sec = mesh.section(plane_origin=origin, plane_normal=axis_t)
        if sec is None:
            continue
        loops = sec.discrete
        cents = [loop.mean(axis=0) for loop in loops]
        radii_here = [float(np.mean(np.linalg.norm(loop - c, axis=1))) for loop, c in zip(loops, cents)]
        if prev_centroid is None:
            i_best = int(np.argmin([np.linalg.norm(c - tip_point_t) for c in cents]))
        else:
            i_best = int(np.argmin([np.linalg.norm(c - prev_centroid) for c in cents]))
        track_list.append((t, cents[i_best], radii_here[i_best]))
        prev_centroid = cents[i_best]

    # Радиус сечения растёт МОНОТОННО и ПЛАВНО от кончика к основанию
    # большого пальца (нет резкого одношагового скачка даже при мелком
    # шаге среза - проверено численно в сессии), поэтому ищем не "скачок",
    # а точку МАКСИМУМА радиуса: дальше профиль выполаживается/снижается -
    # значит, секущая плоскость уже вышла в объём ладони. Максимум радиуса
    # физически - самое широкое место у основания пальца (возвышение
    # большого пальца), что и есть граница палец/ладонь.
    radii_seq = np.array([r for _, _, r in track_list])
    k = max(1, len(radii_seq) // 15)
    smooth = np.convolve(radii_seq, np.ones(k) / k, mode="same")
    cut_i = int(np.argmax(smooth)) + 1
    cut_i = max(cut_i, 6)
    track_list = track_list[:cut_i]

    track_t = np.array([p for _, p, _ in track_list])       # 0=кончик .. -1=у основания
    radii_t = np.array([r for _, _, r in track_list])
    thumb_total_len = float(track_list[-1][0])  # параметр t = длина вдоль оси от кончика
    print(f"  thumb: {len(track_t)} точек трека, длина~{thumb_total_len:.2f}, "
          f"радиусы={np.round(radii_t,2).tolist()}")

    radii_t_rev = radii_t[::-1]  # find_joint_positions ждёт профиль от основания к кончику
    n_t = len(radii_t_rev)
    t_joint_idx_rev = find_joint_positions(radii_t_rev, min_idx_frac=0.2, max_idx_frac=0.85) if n_t >= 6 else [n_t // 2]
    j = t_joint_idx_rev[len(t_joint_idx_rev) // 2]
    frac_j = j / max(n_t - 1, 1)
    base_point_t = track_t[-1]
    R_t, _ = build_segment_frame(base_point_t, tip_point_t)
    seg_lengths_t = [frac_j * thumb_total_len, (1 - frac_j) * thumb_total_len]
    rig["thumb"] = dict(base_pos=base_point_t, R_base=R_t, seg_lengths=seg_lengths_t, total_len=thumb_total_len)
    print(f"  thumb length={thumb_total_len:5.2f} сустав на доле={round(frac_j,2)} "
          f"seg_lengths={[round(s,2) for s in seg_lengths_t]}")

    # --- Бинд-трансформы сегментов (углы=0) ---
    bind_segments = {}
    for name, data in rig.items():
        pos = data["base_pos"].copy()
        R = data["R_base"]
        for i, length in enumerate(data["seg_lengths"]):
            bind_segments[(name, i)] = (pos.copy(), R.copy())
            pos = pos + R[:, 1] * length

    # --- Skinning: LINEAR BLEND SKINNING (LBS) с плавными весами ---
    # Раньше вершина жёстко принадлежала РОВНО одной кости (палец И
    # сегмент). На стыке (особенно MCP - палец/ладонь, и у большого
    # пальца, где один сустав поворачивается почти на 90°) это давало
    # "корону" из острых треугольников: вершины по разные стороны жёсткой
    # границы либо едут за костью целиком, либо не едут вообще - разрыв.
    # Теперь вес плавно переходит между соседними костями в узкой зоне
    # вокруг каждой границы (ladonь<->MCP, MCP<->PIP, PIP<->DIP), через
    # smoothstep. Вершина у САМОЙ границы движется как среднее двух костей.
    print("Геодезические расстояния от кончиков 5 пальцев...")
    all_dist = {}
    for name in finger_names:
        all_dist[name] = geodesic_distances(g, tip_seeds[name], n_verts)

    # Большой палец у основания гораздо толще и площе врастает в ладонь,
    # чем длинные пальцы (возвышение большого пальца) - стандартный margin
    # там всё ещё давал рваные края при повороте (см. сессию), поэтому для
    # большого пальца запас резко уже: лучше чуть короче, но БЕЗ разрывов.
    margins = {name: GEODESIC_MARGIN for name in finger_names}
    margins["thumb"] = 1.05

    BLEND_PALM_FRAC = 0.22   # ширина перехода ладонь<->основание пальца (доля длины пальца)
    BLEND_JOINT_FRAC = 0.14  # ширина перехода между соседними фалангами

    def ramp_up(x, edge, width):
        t = np.clip((x - (edge - width / 2.0)) / max(width, 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    # --- кости: 0=ладонь (неподвижна), дальше по 2-3 сегмента на палец ---
    bone_names = ["palm"]
    bone_bind_pos = [np.zeros(3)]
    bone_bind_R = [np.eye(3)]
    bone_of = {}  # (finger_name, seg_idx) -> индекс кости
    for name in finger_names:
        for seg in range(len(rig[name]["seg_lengths"])):
            bone_of[(name, seg)] = len(bone_names)
            pos, R = bind_segments[(name, seg)]
            bone_names.append(f"{name}:{seg}")
            bone_bind_pos.append(pos)
            bone_bind_R.append(R)
    n_bones = len(bone_names)
    bone_bind_pos = np.array(bone_bind_pos)
    bone_bind_R = np.array(bone_bind_R)
    PALM_BONE = 0

    # --- владелец (ближайший по геодезике палец) определяет, кто вообще
    # участвует в смешивании для данной вершины; остальные пальцы веса не
    # дают (без этого при пересечении зон блендинга соседних пальцев сумма
    # весов поплыла бы) ---
    owner = np.full(n_verts, -1, dtype=int)
    owner_score = np.full(n_verts, np.inf)
    for name in finger_names:
        dist = all_dist[name]
        total_len = rig[name]["total_len"]
        threshold = total_len * margins[name]
        blend_w_abs = total_len * BLEND_PALM_FRAC
        domain = dist < threshold + blend_w_abs * 0.5
        score = dist / max(threshold, 1e-6)
        better = domain & (score < owner_score)
        fi = finger_names.index(name)
        owner[better] = fi
        owner_score[better] = score[better]

    weights = np.zeros((n_verts, n_bones), dtype=np.float32)
    weights[owner == -1, PALM_BONE] = 1.0

    for name in finger_names:
        fi = finger_names.index(name)
        idx = np.where(owner == fi)[0]
        if len(idx) == 0:
            continue
        dist = all_dist[name][idx]
        total_len = rig[name]["total_len"]
        frac_from_base = 1.0 - dist / total_len  # 0=граница с ладонью, 1=кончик (может выходить за [0,1])

        seg_lengths = rig[name]["seg_lengths"]
        fracs = np.array(seg_lengths) / sum(seg_lengths)
        bounds = np.concatenate([[0.0], np.cumsum(fracs)])
        n_seg = len(seg_lengths)

        palm_w = 1.0 - ramp_up(frac_from_base, bounds[0], BLEND_PALM_FRAC)
        seg_w = np.zeros((len(idx), n_seg))
        for s in range(n_seg):
            w_lo = BLEND_PALM_FRAC if s == 0 else BLEND_JOINT_FRAC
            lo_ramp = ramp_up(frac_from_base, bounds[s], w_lo)
            if s == n_seg - 1:
                hi_ramp = 0.0  # последний сегмент тянется до кончика, верхней границы нет
            else:
                hi_ramp = ramp_up(frac_from_base, bounds[s + 1], BLEND_JOINT_FRAC)
            seg_w[:, s] = lo_ramp * (1.0 - hi_ramp)

        total = palm_w + seg_w.sum(axis=1)
        total[total < 1e-9] = 1.0
        weights[idx, PALM_BONE] = palm_w / total
        for s in range(n_seg):
            weights[idx, bone_of[(name, s)]] = seg_w[:, s] / total

    for name in finger_names:
        fi = finger_names.index(name)
        n = int((owner == fi).sum())
        print(f"  {name:7s}: {n} вершин в зоне влияния")
    print(f"  чистая ладонь (вес ладони=1): {int((weights[:, PALM_BONE] >= 0.999).sum())} вершин")

    print("Готово. Сохраняю в", OUT_PATH)
    np.savez(
        OUT_PATH,
        verts=verts,
        faces=faces,
        weights=weights,
        bone_names=np.array(bone_names),
        bone_bind_pos=bone_bind_pos,
        bone_bind_R=bone_bind_R,
        finger_names=np.array(finger_names),
        owner=owner,
        **{f"base_pos__{n}": rig[n]["base_pos"] for n in finger_names},
        **{f"R_base__{n}": rig[n]["R_base"] for n in finger_names},
        **{f"seg_lengths__{n}": np.array(rig[n]["seg_lengths"]) for n in finger_names},
    )
    print("OK")


if __name__ == "__main__":
    main()
