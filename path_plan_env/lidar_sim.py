# -*- coding: utf-8 -*-
"""
Môi trường học tăng cường cho bài toán lập kế hoạch đường đi.

Tạo: Ngày 12 tháng 12 năm 2024, 17:54:17
Cập nhật: Ngày 12 tháng 12 năm 2024

Tác giả: Đào Thành Mạnh
GitHub: https://github.com/PineappleCuteCute
"""

import numpy as np
from shapely import affinity
from shapely import Polygon, Point, LineString, LinearRing
#MakeCircle = lambda position, radius: Point(position).buffer(radius)

from typing import Union
ObstacleLike = Union[Polygon, Point, LineString, LinearRing]

__all__ = ['LidarModel', 'plot_shapely']



# Vẽ hình học shapely
def plot_shapely(geom, ax=None, color=None):
    from shapely import geometry as geo
    from shapely.plotting import plot_line, plot_points, plot_polygon
    if isinstance(geom, (geo.MultiPolygon, geo.Polygon)):
        plot_polygon(geom, ax, False, color)
    elif isinstance(geom, (geo.MultiPoint, geo.Point)):
        plot_points(geom, ax, color, marker='x') # Điểm không có hình dạng, để phân biệt với hình tròn nhỏ, dùng 'x' để thể hiện
    elif isinstance(geom, (geo.MultiLineString, geo.LineString, geo.LinearRing)):
        plot_line(geom, ax, False, color)



# Mô hình Lidar đơn giản
class LidarModel:
    def __init__(self, max_range=500.0, scan_angle=128.0, num_angle=128):
        """Mô hình Lidar (hệ tọa độ Bắc-Đông-Trên)
        Args:
            max_range (float): Khoảng cách quét tối đa (m).
            scan_angle (float): Góc quét tối đa (độ).
            num_angle (int): Số lượng góc quét.
        """
        # Tham số radar
        self.max_range = float(max_range)
        self.scan_angle = float(scan_angle) # độ
        self.num_angle = int(num_angle)
        # Thuộc tính tia sáng
        self.__angles = np.deg2rad(np.linspace(-self.scan_angle/2, self.scan_angle/2, self.num_angle)) # rad
        self.__d = self.max_range
        # Chướng ngại vật
        self.__obstacles: list[ObstacleLike] = [] # Hình dạng của chướng ngại vật
        self.__obstacle_intensities: list[int] = [] # Cường độ phản xạ của chướng ngại vật

    @property
    def obstacles(self):
        """Tất cả các chướng ngại vật"""
        return self.__obstacles

    def add_obstacles(self, obstacles: Union[ObstacleLike, list[ObstacleLike]], intensities: Union[int, list[int]]=255):
        """Thêm/khởi tạo chướng ngại vật
        Args:
            obstacles (ObstacleLike, list[ObstacleLike]): Các chướng ngại vật.
            intensities (int, list[int]): Cường độ phản xạ của chướng ngại vật, 0~255.
        """
        if isinstance(obstacles, list):
            if isinstance(intensities, list):
                assert len(obstacles) == len(intensities), "Số lượng chướng ngại vật phải bằng số lượng cường độ phản xạ"
            else:
                intensities = [intensities] * len(obstacles)
            self.__obstacles.extend(obstacles)
            self.__obstacle_intensities.extend(intensities)
        else:
            self.__obstacles.append(obstacles)
            self.__obstacle_intensities.append(intensities)
    
    def move_obstacle(self, index: int, dx: float, dy: float, drot: float = 0.0):
        """Di chuyển và xoay chướng ngại vật
        Args:
            index (int): Chỉ số của chướng ngại vật.
            dx (float): Khoảng cách di chuyển theo trục x.
            dy (float): Khoảng cách di chuyển theo trục y.\n
            drot (float): Góc xoay ngược chiều kim đồng hồ (rad).
        """
        obstacle = self.__obstacles[index]
        if drot != 0.0:
            obstacle = affinity.rotate(obstacle, drot, use_radians=True)
        obstacle = affinity.translate(obstacle, dx, dy)
        self.__obstacles[index] = obstacle
    
    def scan(self, x: float, y: float, yaw: float, *, mode=0):
        """Quét

        Args:
            x (float): Tọa độ x (m).\n
            y (float): Tọa độ y (m).\n
            yaw (float): Góc lệch (rad).\n
            mode (int): Chế độ trả về (mặc định là 0)\n

        Returns:
            scan_data (ndarray): Dữ liệu quét laser, shape = (3, num_angle), chiều 0 là góc quét, chiều 1 là khoảng cách (-1 có nghĩa là không có chướng ngại vật, 0 có nghĩa là trong Polygon chướng ngại vật), chiều 2 là cường độ điểm mây.\n
            scan_points (list[list], mode!=0): Vị trí các điểm chướng ngại vật đo được, list rỗng nếu không có chướng ngại vật, len0 = 0~num_angle, len1 = 2.\n
        """
        scan_data = np.vstack((self.__angles, -np.ones_like(self.__angles), np.zeros_like(self.__angles))) # (3, num_angle)
        scan_points = []
        # Kiểm tra va chạm
        for o in self.__obstacles:
            if o.geom_type == "Polygon" and o.contains(Point(x, y)):
                scan_data[1, :] = 0
                return scan_data if mode == 0 else (scan_data, scan_points)
        # Quét bằng radar
        for i, angle in enumerate(self.__angles):
            line = LineString([
                (x, y), 
                (x + self.__d * np.cos(yaw + angle), y + self.__d * np.sin(yaw + angle))
            ])
            P, distance, intensity = self.__compute_intersection(line)
            if P is not None:
                scan_data[1][i] = distance
                scan_data[2][i] = intensity
                if mode != 0:
                    scan_points.append(P)
        # Kết thúc một lần quét
        return scan_data if mode == 0 else (scan_data, scan_points)
    
    def __compute_intersection(self, line: LineString):
        """Lấy giao điểm của tia sáng và chướng ngại vật, đo khoảng cách và cường độ phản xạ"""
        P_nearest = None
        distance = self.__d # Khoảng cách 0-max_range
        intensity = None    # Cường độ điểm mây 0-255
        for obstacle, obstacle_intensity in zip(self.__obstacles, self.__obstacle_intensities):
            #if obstacle.intersects(line): # Kiểm tra xem có giao nhau không
            intersections = obstacle.intersection(line) # Giao điểm của đoạn thẳng và hình học: Điểm hoặc Đoạn thẳng / Multi, không thể là MultiPolygon
            if intersections.is_empty:
                continue
            if intersections.geom_type in {'MultiPoint', 'MultiLineString', 'GeometryCollection'}:
                multi_geom = list(intersections.geoms)
            else:
                multi_geom = [intersections]
            for single_geom in multi_geom:
                for P in single_geom.coords: # list(coords) trả về danh sách các điểm tạo thành đoạn thẳng [(x, y), ...]
                    d = np.linalg.norm(np.array(P) - line.coords[0])
                    if d < distance:
                        distance = d
                        P_nearest = list(P) # [x, y]
                        intensity = obstacle_intensity
                #end for
            #end for
        #end for
        return P_nearest, distance, intensity


r"""
1. Polygon
buffer cũng là Polygon
Có thuộc tính exterior (LinearRing), trả về cạnh đóng của mặt Polygon LinearRing(coords)
Có thuộc tính interiors (iter[LinearRing]), trả về các cạnh đóng của các lỗ bên trong mặt, có thể chuyển thành list(interiors) để có dạng list[LinearRing]
Lưu ý: Polygon không có thuộc tính coords và xy (Polygon là mặt, không phải điểm hoặc đoạn thẳng!!)

2. Point
Có thuộc tính coords, trả về [[x, y]]
Có thuộc tính xy, trả về (array('d', [x]), array('d', [y])), có thể chuyển sang dạng list thành ([x], [y])
Có thuộc tính x/y, trả về x và y

3. LineString
Có thuộc tính coords, trả về các điểm tạo thành đoạn thẳng [(x0, y0), ...]
Có thuộc tính xy, trả về (array('d', [x0, ...]), array('d', [y0, ...]), có thể chuyển thành list thành ([x0, ...], [y0, ...])

4. LinearRing
Có thuộc tính coords, trả về các điểm tạo thành đoạn thẳng đóng [(x0, y0), ..., (x0, y0)]
Có thuộc tính xy, trả về (array('d', [x0, ..., x0]), array('d', [y0, ..., y0]), có thể chuyển thành list thành ([x0, ..., x0], [y0, ..., y0])

5. Hình học đa dạng
MultiPoint, MultiLineString, MultiPolygon, GeometryCollection (kiểu hỗn hợp)
Có thuộc tính geoms (iter), trả về iterator chứa tất cả các hình học, Lưu ý: GeometryCollection trả về list???

6. Các thuộc tính chung
Tất cả đều có thuộc tính geom_type (str), trả về loại hình học, có thể thay thế isinstance để kiểm tra
Tất cả đều có thuộc tính centroid (Point), trả về tâm hình học, có thể dùng cho thuật toán tránh chướng ngại vật
Tất cả đều có thuộc tính area (float), trả về diện tích của hình học, Lưu ý: Diện tích của đoạn thẳng đóng là 0
Tất cả đều có thuộc tính length (float), trả về chu vi của hình học
"""
