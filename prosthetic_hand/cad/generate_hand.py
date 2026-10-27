"""
Параметрическая 3D-модель протеза кисти с запястьем (исследовательский прототип).

Концепция:
  - Каждый палец (кроме большого) печатается ЦЕЛЬНОЙ деталью с 2 "живыми"
    шарнирами (flexure joints) — тонкими перемычками вместо осевых
    пальцев/подшипников. Печатается эластичным филаментом (TPU) либо тонким
    PLA. Это стандартный приём в открытых протезах (Flexy-Hand, e-NABLE),
    который убирает необходимость в сборке мелких осевых шарниров пальцев.
  - Большой палец — отдельная деталь с 1 шарниром, закреплён на ладони под
    углом (упрощённая оппозиция, без полной 3D-кинематики запястно-пястного
    сустава — это сознательное упрощение для 1-й итерации).
  - На ладонной стороне каждого сегмента пальца — "ушко" (eyelet) с
    отверстием вдоль пальца: через него пропускается сухожилие (леска/шнур),
    которое тянет серводвигатель в предплечье. Возврат пальца в исходное
    положение — через упругость самого flexure-шарнира (или резинку).
  - Запястье — одноосевой шарнир типа "вилка" (clevis): два выступа на
    детали предплечья охватывают выступ ладони, через все три отверстия
    проходит печатный штифт (pin).
  - Деталь предплечья (forearm_mount) — монтажная площадка с отверстиями
    под крепление к корпусу электроники/серво-блоку (не моделируется здесь:
    это отдельная задача по чертежу электроники).

Требования: cadquery (pip install cadquery). Проверено в этом проекте на
Python 3.12 / Windows, cadquery 2.8.0.

Запуск:  python generate_hand.py
Результат: STL-файлы в ./output/ — печатать КАЖДУЮ деталь отдельно.
"""

import math
import os

import cadquery as cq

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Общие параметры
# ---------------------------------------------------------------------------
TENDON_HOLE_D = 2.2      # диаметр отверстия для лески/шнура (сухожилие)
FLEX_WEB_T = 1.8         # толщина перемычки "живого" шарнира
FLEX_GROOVE_W = 4.0      # длина зоны сгиба вдоль пальца
EYELET_OD = 5.0          # внешний диаметр "ушка" для сухожилия
PIN_D = 4.0              # диаметр штифта запястья (ось сгибания)
PIN_CLEARANCE = 0.3      # зазор на штифт (для посадки с вращением)

# Палец: (имя, [длины сегментов от основания к кончику], ширина, высота, x-позиция на ладони)
FINGERS = [
    ("pinky",  [38, 20, 18], 13.0, 12.0, -27.0),
    ("ring",   [45, 26, 20], 16.0, 14.0, -9.5),
    ("middle", [50, 28, 22], 17.0, 15.0, 9.5),
    ("index",  [45, 25, 20], 16.0, 14.0, 27.0),
]
THUMB = dict(name="thumb", segments=[32, 28], width=18.0, height=16.0)

PALM_WIDTH = 75.0
PALM_LENGTH = 85.0      # от запястья до основания пальцев
PALM_THICKNESS = 14.0

WRIST_KNUCKLE_W = 10.0   # ширина выступа ладони в шарнире запястья
WRIST_KNUCKLE_R = PALM_THICKNESS * 0.6
FOREARM_LEN = 70.0
FOREARM_WIDTH = 50.0
FOREARM_THICK = 12.0
PRONG_THICKNESS = 6.0


def export(shape, name):
    path = os.path.join(OUT_DIR, name)
    cq.exporters.export(shape, path)
    bb = shape.val().BoundingBox()
    print(f"  -> {name}: {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f} mm")
    return shape


def add_flex_joint(solid, y_center, width, web_t=FLEX_WEB_T, groove_w=FLEX_GROOVE_W):
    """Прорезает тонкую гибкую перемычку поперёк пальца в точке y_center."""
    big = 60.0
    top_cut = (
        cq.Workplane("XY")
        .center(0, y_center)
        .box(width * 1.6, groove_w, big, centered=(True, True, False))
        .translate((0, 0, web_t / 2.0))
    )
    bottom_cut = (
        cq.Workplane("XY")
        .center(0, y_center)
        .box(width * 1.6, groove_w, big, centered=(True, True, False))
        .translate((0, 0, -web_t / 2.0 - big))
    )
    return solid.cut(top_cut).cut(bottom_cut)


def add_eyelet(solid, y_center, width, height):
    """Добавляет 'ушко' на ладонной (нижней) стороне сегмента для сухожилия."""
    length = EYELET_OD
    tube_outer = (
        cq.Workplane("XZ")
        .workplane(offset=-(y_center + length / 2.0))
        .circle(EYELET_OD / 2.0)
        .extrude(length)
    )
    tube_outer = tube_outer.translate((0, 0, -height / 2.0))
    bore = (
        cq.Workplane("XZ")
        .workplane(offset=-(y_center + length / 2.0))
        .circle(TENDON_HOLE_D / 2.0)
        .extrude(length)
    )
    bore = bore.translate((0, 0, -height / 2.0))
    tube = tube_outer.cut(bore)
    return solid.union(tube)


def make_finger(name, seg_lengths, width, height):
    """Один цельный палец с flexure-шарнирами и ушками под сухожилие."""
    total_len = sum(seg_lengths)
    body = (
        cq.Workplane("XY")
        .box(width, total_len, height, centered=(True, False, True))
    )
    # fillet вертикальных рёбер для удобства захвата/печати без острых кромок
    try:
        body = body.edges("|Z").fillet(min(width, height) * 0.18)
    except Exception:
        pass  # если fillet не сошёлся на крайних деталях - печатаем без него

    joint_ys = []
    y = 0.0
    for seg in seg_lengths[:-1]:
        y += seg
        joint_ys.append(y)

    for jy in joint_ys:
        body = add_flex_joint(body, jy, width)

    # ушко под сухожилие в середине каждого сегмента (ближе к кончику сегмента)
    y = 0.0
    for seg in seg_lengths:
        eyelet_y = y + seg * 0.7
        body = add_eyelet(body, eyelet_y, width, height)
        y += seg

    return body, total_len


def make_thumb(name, segments, width, height):
    body, total_len = make_finger(name, segments, width, height)
    return body, total_len


def make_palm(finger_x_positions, finger_widths):
    palm = (
        cq.Workplane("XY")
        .box(PALM_WIDTH, PALM_LENGTH, PALM_THICKNESS, centered=(True, False, True))
    )
    palm = palm.edges("|Z").fillet(4.0)

    # выступ (knuckle) для шарнира запястья на нижнем (y=0) торце ладони:
    # диск с осью вдоль Y, вынесенный в y=[-WRIST_KNUCKLE_W, 0]
    cyl = (
        cq.Workplane("XZ")
        .circle(WRIST_KNUCKLE_R)
        .extrude(WRIST_KNUCKLE_W)
        .translate((-WRIST_KNUCKLE_W / 2.0, 0.0, 0.0))
    )
    palm = palm.union(cyl)

    # отверстие штифта в выступе запястья (ось X), центр совмещён с центром
    # выступа по Y (выступ занимает y=[-WRIST_KNUCKLE_W, 0])
    pin_hole = (
        cq.Workplane("YZ")
        .circle((PIN_D + PIN_CLEARANCE) / 2.0)
        .extrude(WRIST_KNUCKLE_W + 2.0)
        .translate((-(WRIST_KNUCKLE_W + 2.0) / 2.0, -WRIST_KNUCKLE_W / 2.0, 0.0))
    )
    palm = palm.cut(pin_hole)

    # сквозные каналы для сухожилий от каждого пальца к запястью (упрощённо:
    # вертикальные отверстия у верхнего торца ладони, выходящие на нижний торец)
    for x in finger_x_positions:
        hole = (
            cq.Workplane("XY")
            .center(x, PALM_LENGTH - 6.0)
            .circle(TENDON_HOLE_D / 2.0 + 0.3)
            .extrude(PALM_THICKNESS + 2, both=True)
        )
        palm = palm.cut(hole)

    return palm


def make_forearm_mount():
    block = cq.Workplane("XY").box(
        FOREARM_WIDTH, FOREARM_LEN, FOREARM_THICK, centered=(True, False, True)
    )
    block = block.edges("|Z").fillet(3.0)

    gap = WRIST_KNUCKLE_W + 1.0  # зазор между зубьями вилки под выступ ладони
    prong_x_offset = gap / 2.0 + PRONG_THICKNESS / 2.0

    def prong(x_center):
        p = (
            cq.Workplane("XZ")
            .workplane(offset=-(FOREARM_LEN + PRONG_THICKNESS))
            .circle(WRIST_KNUCKLE_R + 3.0)
            .extrude(PRONG_THICKNESS)
            .translate((x_center - PRONG_THICKNESS / 2.0, 0.0, 0.0))
        )
        hole = (
            cq.Workplane("YZ")
            .circle((PIN_D + PIN_CLEARANCE) / 2.0)
            .extrude(PRONG_THICKNESS + 2.0)
            .translate((
                x_center - PRONG_THICKNESS / 2.0 - 1.0,
                FOREARM_LEN + PRONG_THICKNESS / 2.0,
                0.0,
            ))
        )
        return p.cut(hole)

    block = block.union(prong(-prong_x_offset))
    block = block.union(prong(prong_x_offset))

    # 4 крепёжных отверстия M3 под корпус серво-блока/электроники
    hole_positions = [
        (-FOREARM_WIDTH / 2 + 8, 8),
        (FOREARM_WIDTH / 2 - 8, 8),
        (-FOREARM_WIDTH / 2 + 8, FOREARM_LEN - 8),
        (FOREARM_WIDTH / 2 - 8, FOREARM_LEN - 8),
    ]
    for x, y in hole_positions:
        h = (
            cq.Workplane("XY")
            .center(x, y)
            .circle(3.2 / 2.0)
            .extrude(FOREARM_THICK + 2, both=True)
        )
        block = block.cut(h)

    return block


def make_pin(length):
    pin = cq.Workplane("XY").circle(PIN_D / 2.0).extrude(length)
    # конус на одном конце для лёгкой установки
    tip = cq.Workplane("XY").workplane(offset=length).sphere(PIN_D / 2.0)
    return pin.union(tip)


def main():
    print("Генерация деталей протеза кисти...\n")

    print("Пальцы:")
    finger_x = [f[4] for f in FINGERS]
    finger_w = [f[2] for f in FINGERS]
    for name, lengths, width, height, x in FINGERS:
        finger, total_len = make_finger(name, lengths, width, height)
        finger = finger.translate((x, PALM_LENGTH, 0.0))
        export(finger, f"finger_{name}.stl")

    print("\nБольшой палец:")
    thumb, thumb_len = make_thumb(**THUMB)
    # большой палец: смещён к краю ладони, развернут на 55° для оппозиции
    thumb = thumb.rotate((0, 0, 0), (0, 0, 1), -55)
    thumb = thumb.translate((-PALM_WIDTH / 2.0 - 2.0, PALM_LENGTH * 0.25, 0.0))
    export(thumb, "finger_thumb.stl")

    print("\nЛадонь:")
    palm = make_palm(finger_x, finger_w)
    export(palm, "palm.stl")

    print("\nУзел запястья (предплечье):")
    forearm = make_forearm_mount()
    export(forearm, "forearm_mount.stl")

    print("\nШтифт запястья:")
    pin = make_pin(WRIST_KNUCKLE_W + 2 * PRONG_THICKNESS + 4.0)
    export(pin, "wrist_pin.stl")

    print("\nГотово. STL-файлы находятся в:", OUT_DIR)
    print("Печатать каждую деталь отдельно; пальцы и ладонь — гибким/полугибким")
    print("филаментом (TPU 95A либо тонкий PLA в зоне шарниров), остальное — PLA/PETG.")


if __name__ == "__main__":
    main()
