# map_builder.py
import math
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import Polygon, Point, box
from shapely.affinity import rotate, translate

def build_map_from_poly(polygon_file_path: str, cell_size: float):
    """
    读取 .poly -> 转换到米制 -> 旋转+平移到第一象限 -> 网格化
    输出：
        S0          : (n_rows, n_cols) 初始覆盖矩阵（Mu=-1, Mn=0）
        rows_list   : 每一行对应的 y 坐标（左下角）
        cols_list   : 每一列对应的 x 坐标（左下角）
        inside_cells: 所有 mission 区格子的 (row, col)
        obs_pos     : 所有障碍/非任务格子的 (row, col)
    """
    # === 1. 读 poly ===
    polygon_data = []
    with open(polygon_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#') and "QGC WPL" not in line:
                parts = line.split()
                if len(parts) >= 2:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    polygon_data.append((lat, lon))

    if not polygon_data:
        raise RuntimeError("poly 文件里没有坐标")

    # === 2. 经纬度 -> 局部米制坐标 ===
    crs_wgs = CRS('epsg:4326')
    first_point = (polygon_data[0][0], polygon_data[0][1])
    local_projection = (
        f"+proj=aeqd +lat_0={first_point[0]} +lon_0={first_point[1]} "
        "+x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
    )
    crs_local = CRS.from_proj4(local_projection)
    transformer = Transformer.from_crs(crs_wgs, crs_local, always_xy=True)

    meter_coords = []
    for lat, lon in polygon_data:
        x, y = transformer.transform(lon, lat)  # 注意顺序
        meter_coords.append((x, y))

    polygon_m = Polygon(meter_coords)

    # === 3. 旋转+平移（跟你原来的代码一样） ===
    convex = polygon_m.convex_hull
    mrr = convex.minimum_rotated_rectangle
    mrr_coords = list(mrr.exterior.coords)[:4]

    p0 = np.array(mrr_coords[0])
    p1 = np.array(mrr_coords[1])
    p2 = np.array(mrr_coords[2])

    e0 = p1 - p0
    e1 = p2 - p1
    len0 = np.linalg.norm(e0)
    len1 = np.linalg.norm(e1)
    width_vec = e0 if len0 <= len1 else e1
    theta = math.atan2(width_vec[1], width_vec[0])
    alpha = math.pi/2 - theta
    alpha_deg = math.degrees(alpha)

    centroid = polygon_m.centroid
    polygon_rot = rotate(polygon_m, alpha_deg, origin=(centroid.x, centroid.y), use_radians=False)

    minx0, miny0, maxx0, maxy0 = polygon_rot.bounds
    polygon_rot = translate(polygon_rot, xoff=-minx0, yoff=-miny0)  # 平移到第一象限

    # === 4. 网格化，计算 inside / partial / outside ===
    minx, miny, maxx, maxy = polygon_rot.bounds

    minx = cell_size * (math.floor(minx / cell_size) - 1)
    maxx = cell_size * (math.ceil(maxx / cell_size) + 1)
    miny = cell_size * (math.floor(miny / cell_size) - 1)
    maxy = cell_size * (math.ceil(maxy / cell_size) + 1)

    rows_list = np.arange(miny, maxy, cell_size)  # i -> y
    cols_list = np.arange(minx, maxx, cell_size)  # j -> x
    n_rows = len(rows_list)
    n_cols = len(cols_list)

    inside_cells = []
    out_pos = []

    for i, y in enumerate(rows_list):
        for j, x in enumerate(cols_list):
            cell = box(x, y, x + cell_size, y + cell_size)
            cx = x + cell_size / 2.0
            cy = y + cell_size / 2.0
            center_point = Point(cx, cy)

            if polygon_rot.contains(center_point):
                # mission 区
                inside_cells.append((i, j))
            elif polygon_rot.intersects(cell):
                # partial：
                inside_cells.append((i, j))
            else:
                # 完全在外面：障碍
                out_pos.append((i, j))

    # === 5. 构造初始 S0（Coverage Area Modeling） ===
    #   Mu = -1: non-mission / obstacle
    #   Mn =  0: mission & non-searched
    S0 = np.full((n_rows, n_cols), -1, dtype=np.int8)
    for (i, j) in inside_cells:
        S0[i, j] = 0

    return S0, rows_list, cols_list, inside_cells, out_pos, polygon_rot
