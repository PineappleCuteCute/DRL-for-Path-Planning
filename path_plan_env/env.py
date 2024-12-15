# -*- coding: utf-8 -*-
"""
Môi trường học tăng cường cho bài toán lập kế hoạch đường đi.

Tạo: Ngày 12 tháng 12 năm 2024, 17:54:17
Cập nhật: Ngày 12 tháng 12 năm 2024

Tác giả: Đào Thành Mạnh
GitHub: https://github.com/PineappleCuteCute
"""
#
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
from gym import spaces
from copy import deepcopy
from pathlib import Path
from collections import deque
from scipy.integrate import odeint
from shapely import geometry as geo
from shapely.plotting import plot_polygon

__all__ = ["DynamicPathPlanning", "StaticPathPlanning", "NormalizedActionsWrapper"]



# ----------------------------- ↓↓↓↓↓ Cài đặt bản đồ ↓↓↓↓↓ ------------------------------#

class MAP:
    """
    Lớp này định nghĩa các thông tin về bản đồ, bao gồm kích thước bản đồ,
    vị trí điểm bắt đầu, điểm kết thúc và các vật cản (obstacles).
    
    Các thuộc tính:
        size (list): Kích thước của bản đồ dưới dạng danh sách 2 phần tử, mỗi phần tử
                     là một danh sách chứa các giá trị min, max cho trục x và z.
        start_pos (list): Tọa độ của điểm xuất phát trên bản đồ.
        end_pos (list): Tọa độ của điểm kết thúc trên bản đồ.
        obstacles (list): Danh sách các vật cản trên bản đồ, có thể là `geo.Polygon` 
                          hoặc các đối tượng `geo.Point/geo.LineString` có thể có vùng đệm (buffer).
    """
    size = [[-15.0, -15.0], [15.0, 15.0]]  # x, z giá trị tối thiểu; x, z giá trị tối đa
    start_pos = [0, -9]  # Tọa độ điểm bắt đầu
    end_pos = [2.5, 9]  # Tọa độ điểm kết thúc
    obstacles = [  # Danh sách các vật cản, yêu cầu là geo.Polygon hoặc geo.Point/geo.LineString có vùng đệm (buffer)
        geo.Point(0, 2.5).buffer(4),  # Vật cản 1
        geo.Point(-6, -5).buffer(3),  # Vật cản 2
        geo.Point(6, -5).buffer(3),  # Vật cản 3
        geo.Polygon([(-10, 0), (-10, 5), (-7.5, 5), (-7.5, 0)])  # Vật cản 4
    ]

    @classmethod
    def show(cls):
        """
        Hàm này dùng để hiển thị bản đồ với các vật cản, điểm bắt đầu và điểm kết thúc.
        """
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.close('all')
        fig, ax = plt.subplots()
        ax.clear()
        cls.plot(ax)  # Vẽ bản đồ
        # Đánh dấu điểm bắt đầu và điểm kết thúc
        ax.scatter(cls.start_pos[0], cls.start_pos[1], s=30, c='k', marker='x', label='Điểm bắt đầu')
        ax.scatter(cls.end_pos[0], cls.end_pos[1], s=30, c='k', marker='o', label='Điểm kết thúc')
        ax.legend(loc='best').set_draggable(True)
        plt.show(block=True)

    @classmethod
    def plot(cls, ax, title='Bản đồ'):
        """
        Hàm này dùng để vẽ bản đồ, bao gồm các vật cản, trục tọa độ và lưới.
        
        Args:
            ax: Đối tượng trục của matplotlib để vẽ.
            title: Tiêu đề của bản đồ.
        """
        ax.clear()
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(alpha=0.3, ls=':')
        ax.set_xlim(cls.size[0][0], cls.size[1][0])
        ax.set_ylim(cls.size[0][1], cls.size[1][1])
        ax.invert_yaxis()  # Đảo ngược trục y (do đặc thù hệ tọa độ trong bản đồ)
        # Vẽ các vật cản trên bản đồ
        for o in cls.obstacles:
            plot_polygon(o, ax=ax, facecolor='w', edgecolor='k', add_points=False)

        



# ----------------------------- ↓↓↓↓↓ Môi trường động học tránh chướng ngại vật ↓↓↓↓↓ ------------------------------#

# Import mô hình lidar từ file lidar_sim.py
if __name__ == '__main__':
    from lidar_sim import LidarModel  # Nếu là chạy trực tiếp, nhập khẩu LidarModel
else:
    from .lidar_sim import LidarModel  # Nếu là phần của gói module, nhập khẩu LidarModel

# Lớp Logger (Hiện tại chưa có mã thực thi)
class Logger:
    pass

# Cài đặt vận tốc
V_LOW = 0.05  # Vận tốc thấp nhất
V_HIGH = 0.2  # Vận tốc cao nhất
V_MIN = V_LOW + 0.03  # Giới hạn dưới cho khu vực phạt (lớn hơn V_LOW)
V_MAX = V_HIGH - 0.03  # Giới hạn trên cho khu vực phạt (nhỏ hơn V_HIGH)

# Cài đặt trạng thái động lực học của khối tâm
STATE_LOW = [MAP.size[0][0], MAP.size[0][1], V_LOW, -math.pi]  # Trạng thái tối thiểu: x, z, vận tốc (V), góc hướng (ψ)
STATE_HIGH = [MAP.size[1][0], MAP.size[1][1], V_HIGH, math.pi]  # Trạng thái tối đa: x, z, vận tốc (V), góc hướng (ψ)

# Cài đặt quan sát
OBS_STATE_LOW = [0, V_LOW, -math.pi]  # Tọa độ tương đối đến điểm kết thúc + vận tốc + góc giữa điểm kết thúc và vận tốc (đơn vị radian)
OBS_STATE_HIGH = [1.414*max(STATE_HIGH[0]-STATE_LOW[0], STATE_HIGH[1]-STATE_LOW[1]), V_HIGH, math.pi]  # Tọa độ tương đối + vận tốc + góc giữa điểm kết thúc và vận tốc (đơn vị radian)

# Cài đặt điều khiển
CTRL_LOW = [-0.02, -0.005]  # Tăng tốc (tangential acceleration) và góc quay vận tốc (velocity roll angle) tối thiểu (đơn vị rad/s)
CTRL_HIGH = [0.02, 0.005]  # Tăng tốc và góc quay vận tốc tối đa (đơn vị rad/s)

# Cài đặt radar
SCAN_RANGE = 2.5  # Khoảng cách quét tối đa của radar (đơn vị mét)
SCAN_ANGLE = 128  # Góc quét radar (đơn vị độ)
SCAN_NUM = 128  # Số điểm quét của radar
SCAN_CEN = 48  # Chỉ số bắt đầu vùng trung tâm trong mảng quét (nhỏ hơn SCAN_NUM/2)

# Cài đặt khoảng cách an toàn và vùng đệm
D_SAFE = 0.5  # Bán kính va chạm an toàn (đơn vị mét)
D_BUFF = 1.0  # Khoảng cách vùng đệm (bigger than D_SAFE)
D_ERR = 0.5  # Khoảng cách sai lệch mục tiêu (đơn vị mét)

# Cài đặt độ dài chuỗi quan sát
TIME_STEP = 4  # Số bước thời gian trong một chuỗi quan sát



class DynamicPathPlanning(gym.Env):
    """ 
    Lớp này định nghĩa một môi trường lập kế hoạch đường đi dựa trên lực học và điều khiển (trong hệ tọa độ Đông-Tây-Nam).
    Các phương trình động lực học:
    >>> dx/dt = V * cos(ψ)          # Tốc độ thay đổi theo hướng x
    >>> dz/dt = -V * sin(ψ)         # Tốc độ thay đổi theo hướng z
    >>> dV/dt = g * nx               # Tốc độ thay đổi theo vận tốc V
    >>> dψ/dt = -g / V * tan(μ)      # Thay đổi góc ψ theo vận tốc V và lực điều khiển
    >>> u = [nx, μ]                  # Lực điều khiển: nx và góc μ

    """
    
    def __init__(self, max_episode_steps=500, dt=0.5, normalize_observation=True, old_gym_style=True):
        """
        Khởi tạo môi trường DynamicPathPlanning.

        Args:
            max_episode_steps (int): Số bước tối đa trong một episode (mặc định 500).
            dt (float): Thời gian quyết định (mặc định 0.5).
            normalize_observation (bool): Quyết định có sử dụng giá trị quan sát đã chuẩn hóa hay không (mặc định True).
            old_gym_style (bool): Quyết định có sử dụng giao diện gym cũ hay không (mặc định True).
        """
        
        # Thiết lập các tham số môi trường
        self.dt = dt  # Thời gian quyết định (Time step)
        self.max_episode_steps = max_episode_steps  # Số bước tối đa trong một episode
        self.log = Logger()  # Đối tượng Logger, để theo dõi hoặc ghi lại thông tin trong môi trường
        
        # Thiết lập chướng ngại vật và radar
        self.obstacles = MAP.obstacles  # Lấy danh sách chướng ngại vật từ lớp MAP
        self.lidar = LidarModel(SCAN_RANGE, SCAN_ANGLE, SCAN_NUM)  # Khởi tạo mô hình radar lidar
        self.lidar.add_obstacles(MAP.obstacles)  # Thêm các chướng ngại vật vào mô hình lidar
        
        # Thiết lập không gian trạng thái và không gian điều khiển
        self.state_space = spaces.Box(np.array(STATE_LOW), np.array(STATE_HIGH))  # Không gian trạng thái (position, velocity, etc.)
        self.control_space = spaces.Box(np.array(CTRL_LOW), np.array(CTRL_HIGH))  # Không gian điều khiển (force, angle, etc.)
        
        # Thiết lập không gian quan sát và không gian hành động
        # Không gian quan sát gồm hai phần: các điểm lidar và các vector quan sát
        points_space = spaces.Box(-1, SCAN_RANGE, (TIME_STEP, SCAN_NUM, ))  # Không gian quan sát các điểm lidar (chuỗi dài TIME_STEP)
        vector_space = spaces.Box(np.array([OBS_STATE_LOW]*TIME_STEP), np.array([OBS_STATE_HIGH]*TIME_STEP))  # Không gian các vector quan sát (tốc độ, góc, etc.)
        
        # Không gian quan sát là một Dictionary bao gồm seq_points và seq_vector
        self.observation_space = spaces.Dict({'seq_points': points_space, 'seq_vector': vector_space})
        
        # Không gian hành động (giới hạn giữa -1 và 1 đối với các điều khiển)
        self.action_space = spaces.Box(-1, 1, (len(CTRL_LOW), )) 
        
        # Khởi tạo các bộ đệm cho chuỗi quan sát (deque dùng để lưu trữ và duy trì độ dài cố định cho chuỗi quan sát)
        self.deque_points = deque(maxlen=TIME_STEP)
        self.deque_vector = deque(maxlen=TIME_STEP)
        
        # Cài đặt điều khiển môi trường
        self.__render_not_called = True  # Biến kiểm tra xem có cần render không
        self.__need_reset = True  # Biến kiểm tra xem có cần reset môi trường không
        self.__norm_observation = normalize_observation  # Cài đặt quan sát có chuẩn hóa hay không
        self.__old_gym = old_gym_style  # Cài đặt sử dụng giao diện gym cũ hay mới
        
        # Cài đặt cấu hình cho việc vẽ biểu đồ (matplotlib)
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Cấu hình font hiển thị tiếng Trung
        plt.rcParams['axes.unicode_minus'] = False  # Cấu hình hiển thị dấu âm
        plt.close("all")  # Đóng tất cả các cửa sổ vẽ hiện tại


    def reset(self, mode=0):
        """Khôi phục lại môi trường.
        mode=0: Khởi tạo ngẫu nhiên vị trí bắt đầu và kết thúc, tốc độ và hướng ngẫu nhiên.
        mode=1: Khởi tạo vị trí bắt đầu và kết thúc theo thiết lập trong bản đồ, tốc độ và hướng ngẫu nhiên.
        """
        self.__need_reset = False  # Đánh dấu rằng môi trường không cần phải reset lại.
        self.time_step = 0  # Đặt lại số bước thời gian về 0.

        # Khởi tạo hành trình / trạng thái / điều khiển
        while 1:
            self.state = self.state_space.sample()  # Lấy mẫu ngẫu nhiên từ không gian trạng thái.
            if mode == 0:
                # Nếu mode = 0, khởi tạo vị trí bắt đầu và kết thúc ngẫu nhiên trong không gian trạng thái.
                self.start_pos = deepcopy(self.state[:2])  # Sao chép vị trí bắt đầu (lấy 2 phần tử đầu tiên).
                self.end_pos = deepcopy(self.state_space.sample()[:2])  # Sao chép vị trí kết thúc (lấy 2 phần tử đầu tiên).
            else:
                # Nếu mode = 1, khởi tạo vị trí bắt đầu và kết thúc từ MAP đã được định sẵn.
                self.start_pos = np.array(MAP.start_pos, dtype=np.float32)  # Vị trí bắt đầu từ bản đồ.
                self.end_pos = np.array(MAP.end_pos, dtype=np.float32)  # Vị trí kết thúc từ bản đồ.
                # Cập nhật trạng thái với vị trí bắt đầu và kết thúc.
                self.state = np.array([*self.start_pos[:2], *self.state[2:]], dtype=np.float32)
            
            # Kiểm tra xem vị trí bắt đầu và kết thúc có nằm trong chướng ngại vật không.
            for o in self.obstacles:
                if o.contains(geo.Point(*self.start_pos)) or o.contains(geo.Point(*self.end_pos)):
                    # Nếu vị trí nằm trong chướng ngại vật, và mode != 0, báo lỗi.
                    if mode != 0:
                        raise ValueError("Vị trí bắt đầu/kết thúc không thể nằm trong chướng ngại vật!!!")
                    break
            else:
                break  # Nếu không có lỗi, thoát khỏi vòng lặp.

        # Khởi tạo hành trình, điều khiển và các giá trị khác.
        self.L = 0.0  # Khởi tạo quãng đường đi được (L) bằng 0.
        self.ctrl = np.zeros(self.action_space.shape, dtype=np.float32)  # Khởi tạo điều khiển ban đầu là mảng không (zero vector).

        # Khởi tạo quan sát
        self.deque_points.extend([np.array([-1]*SCAN_NUM, dtype=np.float32)]*(TIME_STEP-1))  # Khởi tạo các điểm lidar với giá trị -1.
        self.deque_vector.extend([np.array(OBS_STATE_LOW, dtype=np.float32)]*(TIME_STEP-1))  # Khởi tạo vector quan sát với giá trị thấp nhất.
        obs = self._get_obs(self.state)  # Lấy quan sát từ trạng thái hiện tại.

        # Khởi tạo bộ nhớ (memory)
        self.exist_last = None  # Lưu trạng thái của vùng quét trung tâm ở thời điểm trước.
        self.D_init = deepcopy(obs['seq_vector'][-1][0])  # Lưu khoảng cách ban đầu từ điểm hiện tại đến mục tiêu.
        self.D_last = deepcopy(obs['seq_vector'][-1][0])  # Lưu khoảng cách hiện tại từ điểm hiện tại đến mục tiêu.

        # Khởi tạo log
        self.log.start_pos = self.start_pos  # Lưu vị trí bắt đầu vào log.
        self.log.end_pos = self.end_pos  # Lưu vị trí kết thúc vào log.
        self.log.path = [self.start_pos]  # Khởi tạo đường đi ban đầu chỉ có vị trí bắt đầu.
        self.log.ctrl = [self.ctrl]  # Khởi tạo điều khiển ban đầu vào log.
        self.log.speed = [self.state[2]]  # Lưu tốc độ ban đầu vào log.
        self.log.yaw = [self.state[3]]  # Lưu góc yaw (góc phương vị) ban đầu vào log.
        self.log.length = [[self.L, self.D_last]]  # Lưu quãng đường và khoảng cách đến mục tiêu vào log.
        self.log.curr_scan_pos = []  # Lưu các tọa độ chướng ngại vật đã quét trong lần quét hiện tại.

        # Trả về quan sát đã chuẩn hóa
        if self.__old_gym:
            return self._norm_obs(obs)  # Nếu sử dụng giao diện cũ của gym, trả về quan sát đã chuẩn hóa.
        return self._norm_obs(obs), {}  # Trả về quan sát đã chuẩn hóa cùng với một từ điển trống (empty dict).

    def _get_ctrl(self, act, tau=0.9):
        """Lấy tín hiệu điều khiển
        Phương thức này chuyển đổi hành động từ không gian [-1, 1] sang không gian điều khiển thực tế,
        đồng thời làm mượt tín hiệu điều khiển để tránh các thay đổi đột ngột.
        """
        lb = self.control_space.low  # Lấy giới hạn dưới của không gian điều khiển.
        ub = self.control_space.high  # Lấy giới hạn trên của không gian điều khiển.
        u = lb + (act + 1.0) * 0.5 * (ub - lb)  # Chuyển đổi hành động từ [-1, 1] sang [lb, ub].
        u = np.clip(u, lb, ub)  # Đảm bảo tín hiệu điều khiển không vượt quá giới hạn của không gian điều khiển.
        
        # Làm mượt tín hiệu điều khiển nếu cần.
        if tau is not None:
            return (1.0 - tau) * self.ctrl + tau * u  # Tín hiệu điều khiển mượt hơn.
        return u  # Trả về tín hiệu điều khiển thô nếu không cần làm mượt.


    def _get_obs(self, state):
        """Lấy quan sát gốc"""
        x, z, V, ψ = state  # Giải nén trạng thái: x (tọa độ x), z (tọa độ z), V (tốc độ), ψ (góc yaw)
        
        # Tính toán trạng thái tương đối
        V_vec = np.array([V*math.cos(ψ), -V*math.sin(ψ)], np.float32)  # Vecto vận tốc theo hướng x và z (cos(ψ), -sin(ψ))
        R_vec = self.end_pos - state[:2]  # Vecto từ vị trí hiện tại (x, z) tới mục tiêu (end_pos)
        D = np.linalg.norm(R_vec)  # Khoảng cách từ vị trí hiện tại đến mục tiêu
        
        # Tính góc phương nhìn (hướng của đối tượng so với hướng di chuyển)
        q = self._vector_angle(V_vec, R_vec)  # Tính góc giữa vecto vận tốc và vecto hướng tới mục tiêu
        
        # Tạo vector quan sát
        vector = np.array([D, V, q], np.float32)  # Quan sát bao gồm: khoảng cách đến mục tiêu, tốc độ, góc phương nhìn
        
        # Thêm vào bộ nhớ deque để giữ lại các quan sát theo thời gian
        self.deque_vector.append(vector)
        
        # Quét lidar để đo khoảng cách
        points, self.log.curr_scan_pos = self.lidar.scan(x, z, -ψ, mode=1)  # Quét lidar tại vị trí (x, z) với góc quay -ψ
        # Lưu ý: Tọa độ trong hệ tọa độ Đông-Bắc-Tây (LidarModel) và hệ tọa độ Đông-Tây-Nam (ControlModel) có sự khác biệt về hướng góc ψ
        
        # Lưu lại các điểm quét lidar (chỉ lấy dữ liệu khoảng cách)
        self.deque_points.append(points[1])  # Thêm điểm quét vào bộ nhớ deque

        # Trả về quan sát
        return {'seq_points': np.array(self.deque_points, np.float32),  # Dữ liệu quét lidar theo chuỗi
                'seq_vector': np.array(self.deque_vector, np.float32)}  # Dữ liệu quan sát trạng thái theo chuỗi

    
    def _get_rew(self):
        """Lấy phần thưởng"""
        rew = -0.01  # Khởi tạo phần thưởng (phạt nhẹ khi không có hành động cụ thể)

        # 1. Phần thưởng tránh chướng ngại vật chủ động [-2, 2]
        point0 = self.deque_points[-2]  # Lấy điểm quét lidar của lần trước (tại thời điểm t-1)
        center0 = point0[SCAN_CEN:-SCAN_CEN]  # Lấy các điểm trong vùng trung tâm của lần trước
        point1 = self.deque_points[-1]  # Lấy điểm quét lidar của lần hiện tại (tại thời điểm t)
        center1 = point1[SCAN_CEN:-SCAN_CEN]  # Lấy các điểm trong vùng trung tâm của lần hiện tại
        
        # Tính toán sự thay đổi của chướng ngại vật trong khu vực trung tâm
        if self.exist_last is None:
            self.exist_last = np.any(center0 > -0.5)  # Nếu không có dữ liệu trước đó, kiểm tra xem có chướng ngại vật trong vùng trung tâm không
        exist = np.any(center1 > -0.5)  # Kiểm tra xem có chướng ngại vật trong vùng trung tâm tại thời điểm hiện tại

        if exist:
            # Nếu luôn có chướng ngại vật: Đánh giá sự thay đổi khoảng cách
            if self.exist_last:
                effective_center0, effective_center1 = center0[center0 > -0.5], center1[center1 > -0.5]  # Lọc các điểm có khoảng cách thực tế
                d0_mean = np.mean(effective_center0) + 1e-8  # Khoảng cách trung bình của lần trước (để tránh chia cho 0)
                d1_mean = np.mean(effective_center1) + 1e-8  # Khoảng cách trung bình của lần hiện tại
                d0_min = min(effective_center0)  # Khoảng cách nhỏ nhất trong vùng trung tâm tại thời điểm t-1
                d1_min = min(effective_center1)  # Khoảng cách nhỏ nhất trong vùng trung tâm tại thời điểm t
                # Nếu khoảng cách nhỏ nhất tại thời điểm t lớn hơn t-1, thưởng cho việc tránh chướng ngại vật
                rew += np.clip(d1_mean / d0_mean, 0.2, 2) if d1_min > d0_min else -np.clip(d0_mean / d1_mean, 0.2, 2)
            else:
                # Nếu không có chướng ngại vật -> có chướng ngại vật
                rew -= 0.5  # Phạt khi có sự xuất hiện của chướng ngại vật mới
        else:
            # Nếu có chướng ngại vật -> không có chướng ngại vật
            if self.exist_last:
                rew += 1.0  # Thưởng khi chướng ngại vật biến mất
            # Nếu không có chướng ngại vật suốt quá trình, phần thưởng không thay đổi
            pass
        
        # 2. Phần thưởng tránh chướng ngại vật bị động [-1, 0]
        d_min = min([*point1[point1 > -0.5], np.inf])  # Khoảng cách tối thiểu từ lidar (không tính các điểm -0.5)
        if d_min <= D_BUFF:
            rew += d_min / D_BUFF - 1  # Phạt khi quá gần chướng ngại vật (khoảng cách càng nhỏ, phần thưởng càng giảm)
        
        # 3. Phần thưởng gần mục tiêu {-1.5, 2.0}
        D = self.deque_vector[-1][0]  # Khoảng cách đến mục tiêu tại thời điểm hiện tại
        rew += 2.0 if D < self.D_last else -1.5  # Thưởng khi gần hơn mục tiêu và phạt khi xa hơn
        
        # 4. Phần thưởng giữ tốc độ [-1, 0]
        V = self.deque_vector[-1][1]  # Tốc độ tại thời điểm hiện tại
        if V < V_MIN:
            rew += (V - V_MIN) / (V_MIN - V_LOW)  # Phạt nếu tốc độ dưới mức tối thiểu
        elif V > V_MAX:
            rew += (V - V_MAX) / (V_MAX - V_HIGH)  # Phạt nếu tốc độ trên mức tối đa
        
        # 5. Phần thưởng góc phương nhìn [-1, 1]
        q = self.deque_vector[-1][2]  # Góc phương nhìn (góc giữa hướng di chuyển và mục tiêu)
        rew += math.sin(q)  # Phần thưởng hoặc phạt tùy theo góc phương nhìn, sin(x) trong khoảng [-1, 1]
        
        # 6. Phần thưởng cho nhiệm vụ
        done = False
        info = {'state': 'none'}
        if d_min < D_SAFE:  # Kiểm tra va chạm
            rew -= 150  # Phạt nặng khi va chạm xảy ra
            done = True
            info['state'] = 'fail'
        elif D < D_ERR:  # Kiểm tra xem đã hoàn thành nhiệm vụ (đạt mục tiêu chưa)
            η = np.nanmax([3.5 - 2.5 * self.L / (self.D_init + 1e-8), 0.5])  # Tính hệ số giảm phần thưởng tùy theo quãng đường đã đi
            rew += 200 * η  # Thưởng khi đạt mục tiêu, phần thưởng có thể dao động từ 100 đến 700+
            done = True
            info['state'] = 'success'
        
        # Phạt nếu tốc độ không hợp lý hoặc quá gần chướng ngại vật
        if V < V_MIN or V > V_MAX or d_min < D_BUFF:
            rew -= 5  # Phạt nếu tốc độ quá thấp hoặc quá cao, hoặc nếu quá gần chướng ngại vật
        
        # Cập nhật trạng thái nhớ lại
        self.exist_last = deepcopy(exist)
        self.D_last = deepcopy(D)
        
        # Trả về phần thưởng, trạng thái kết thúc và thông tin bổ sung
        return rew, done, info


    def step(self, act: np.ndarray, tau: float = None):
        """Chuyển trạng thái
        Args:
            act (np.ndarray): Đầu vào hành động a (giá trị trong phạm vi -1 đến 1).
            tau (float): Hệ số làm mượt điều khiển u (giá trị trong phạm vi u_min đến u_max): u = tau * u + (1 - tau) * u_last. Mặc định là None (không làm mượt).
        """
        assert not self.__need_reset, "Phải gọi reset trước khi gọi step"

        # Cập nhật bước thời gian
        self.time_step += 1

        # Tính toán điều khiển (u) từ hành động (act)
        u = self._get_ctrl(act, tau)

        # Giải phương trình vi phân (hoặc tính toán bước chuyển tiếp trạng thái)
        new_state = self._ode45(self.state, u, self.dt)

        # Kiểm tra xem có kết thúc không (do đạt số bước tối đa hoặc va chạm)
        truncated = False
        if self.time_step >= self.max_episode_steps:
            truncated = True

        # Kiểm tra nếu khoảng cách đến mục tiêu vượt quá giới hạn
        elif self.D_last > self.observation_space["seq_vector"].high[-1][0]:
            truncated = True

        # Cập nhật hành trình và trạng thái
        self.L += np.linalg.norm(new_state[:2] - self.state[:2])  # Tính toán quãng đường di chuyển
        self.state = deepcopy(new_state)  # Cập nhật trạng thái
        self.ctrl = deepcopy(u)  # Cập nhật điều khiển

        # Lấy thông tin quan sát mới
        obs = self._get_obs(new_state)

        # Tính toán phần thưởng, kiểm tra trạng thái kết thúc và các thông tin bổ sung
        rew, done, info = self._get_rew()
        info["done"] = done
        info["truncated"] = truncated

        # Kiểm tra trạng thái kết thúc (nếu đã vượt quá bước tối đa hoặc xảy ra va chạm)
        if truncated or done:
            info["terminal"] = True
            self.__need_reset = True  # Cần reset lại môi trường
        else:
            info["terminal"] = False

        # Cập nhật thông tin phần thưởng và các thông tin khác
        info["reward"] = rew
        info["time_step"] = self.time_step
        info["voyage"] = self.L
        info["distance"] = self.D_last

        # Lưu lại lịch sử (log) của quá trình
        self.log.path.append(self.state[:2])  # Lưu tọa độ hiện tại
        self.log.ctrl.append(u)  # Lưu điều khiển
        self.log.speed.append(self.state[2])  # Lưu tốc độ
        self.log.yaw.append(self.state[3])  # Lưu góc phương vị
        self.log.length.append([self.L, self.D_last])  # Lưu quãng đường và khoảng cách còn lại

        # Trả về kết quả:
        if self.__old_gym:
            return self._norm_obs(obs), rew, done, info
        return self._norm_obs(obs), rew, done, truncated, info

    
    def _norm_obs(self, obs):
        """Chuẩn hóa quan sát"""
        if not self.__norm_observation:
            return obs  # Nếu không cần chuẩn hóa, trả về quan sát gốc

        # Chuẩn hóa seq_vector (chuỗi vector)
        obs['seq_vector'] = self._linear_mapping(
            obs['seq_vector'], 
            self.observation_space['seq_vector'].low, 
            self.observation_space['seq_vector'].high
        )

        # Chuẩn hóa các điểm quét (scan points)
        obs['seq_points'] = self._normalize_points(obs['seq_points'])

        return obs

    
    def render(self, mode="human", figsize=[8,8]):
        """Hiển thị trực quan môi trường trong quá trình thử nghiệm, gọi liên tục với step (Không nên gọi với plot cùng lúc, dễ bị treo)"""
        assert not self.__need_reset, "Phải gọi reset trước khi gọi render"

        # Tạo cửa sổ vẽ đồ thị
        if self.__render_not_called:
            self.__render_not_called = False
            with plt.ion():  # Kích hoạt chế độ interactive của matplotlib
                fig = plt.figure("render", figsize=figsize)  # Tạo cửa sổ vẽ với kích thước cho trước
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # Tạo một axes trong cửa sổ vẽ
            MAP.plot(ax, "Path Plan Environment")  # Vẽ bản đồ môi trường (MAP) lên axes
            self.__plt_car_path, = ax.plot([], [], 'k-.')  # Đường đi của xe (màu đen, kiểu nét chấm gạch)
            self.__plt_car_point = ax.scatter([], [], s=15, c='b', marker='o', label='Agent')  # Vị trí của agent (xe), màu xanh
            self.__plt_targ_range, = ax.plot([], [], 'g:', linewidth=1.0)  # Vòng tròn phạm vi mục tiêu (màu xanh lá)
            self.__plt_targ_point = ax.scatter([], [], s=15, c='g', marker='o', label='Target')  # Vị trí của mục tiêu (màu xanh lá)
            self.__plt_lidar_scan, = ax.plot([], [], 'ro', markersize=1.5, label='Points')  # Các điểm quét lidar (màu đỏ)
            self.__plt_lidar_left, = ax.plot([], [], 'c--', linewidth=0.5)  # Lằn ranh trái của lidar (màu xanh da trời, nét đứt)
            self.__plt_lidar_right, = ax.plot([], [], 'c--', linewidth=0.5)  # Lằn ranh phải của lidar (màu xanh da trời, nét đứt)
            ax.legend(loc='best').set_draggable(True)  # Hiển thị chú giải và cho phép kéo thả

        # Cập nhật các phần tử vẽ đồ thị
        self.__plt_car_path.set_data(np.array(self.log.path).T)  # Cập nhật đường đi của xe (theo trục x và y)
        self.__plt_car_point.set_offsets(self.log.path[-1])  # Cập nhật vị trí hiện tại của xe
        θ = np.linspace(0, 2 * np.pi, 18)  # Tạo một mảng góc từ 0 đến 2pi (dùng cho vòng tròn xung quanh mục tiêu)
        self.__plt_targ_range.set_data(self.log.end_pos[0] + D_ERR * np.cos(θ), self.log.end_pos[1] + D_ERR * np.sin(θ))  # Vẽ vòng tròn phạm vi mục tiêu
        self.__plt_targ_point.set_offsets(self.log.end_pos)  # Cập nhật vị trí của mục tiêu
        if self.log.curr_scan_pos:
            points = np.array(self.log.curr_scan_pos)  # Cập nhật các điểm quét lidar hiện tại
            self.__plt_lidar_scan.set_data(points[:, 0], points[:, 1])
        else:
            self.__plt_lidar_scan.set_data([], [])  # Nếu không có dữ liệu quét, vẽ trống
        x, y, yaw = *self.log.path[-1], self.log.yaw[-1]  # Lấy vị trí và góc quay của xe
        x1 = x + self.lidar.max_range * np.cos(-yaw + np.deg2rad(self.lidar.scan_angle / 2))  # Tính toán vị trí cực đại quét lidar bên trái
        x2 = x + self.lidar.max_range * np.cos(-yaw - np.deg2rad(self.lidar.scan_angle / 2))  # Tính toán vị trí cực đại quét lidar bên phải
        y1 = y + self.lidar.max_range * np.sin(-yaw + np.deg2rad(self.lidar.scan_angle / 2))  # Tính toán vị trí cực đại quét lidar bên trái
        y2 = y + self.lidar.max_range * np.sin(-yaw - np.deg2rad(self.lidar.scan_angle / 2))  # Tính toán vị trí cực đại quét lidar bên phải
        self.__plt_lidar_left.set_data([x, x1], [y, y1])  # Vẽ đường quét bên trái của lidar
        self.__plt_lidar_right.set_data([x, x2], [y, y2])  # Vẽ đường quét bên phải của lidar

        # Dừng lại một chút để cửa sổ vẽ cập nhật
        plt.pause(0.001)


    def close(self): 
        """Đóng môi trường"""
        self.__render_not_called = True  # Đánh dấu rằng render chưa được gọi
        self.__need_reset = True  # Đánh dấu rằng môi trường cần được reset khi gọi lại
        plt.close("render")  # Đóng cửa sổ vẽ đồ họa có tên "render"


    def plot(self, file, figsize=[10,10], dpi=100):
        """Quan sát trạng thái sau mỗi lần huấn luyện (tránh gọi cùng với render, dễ gây lag)"""
        file = Path(file).with_suffix(".png")  # Chuyển đổi tên tệp đầu vào thành định dạng ".png"
        file.parents[0].mkdir(parents=True, exist_ok=True)  # Tạo thư mục chứa tệp nếu chưa có
        fig = plt.figure("Output", figsize=figsize)  # Tạo cửa sổ đồ họa
        gs = fig.add_gridspec(2, 2)  # Tạo lưới con 2x2 cho các đồ thị
        ax1 = fig.add_subplot(gs[0, 0])  # Đồ thị quỹ đạo (trajectory)
        ax2 = fig.add_subplot(gs[0, 1])  # Đồ thị tín hiệu điều khiển (control signal)
        ax3 = fig.add_subplot(gs[1, 0])  # Đồ thị tín hiệu tốc độ (speed signal)
        ax4 = fig.add_subplot(gs[1, 1])  # Đồ thị tín hiệu chiều dài (length signal)

        # Vẽ đồ thị quỹ đạo
        MAP.plot(ax1, "Trajectory")
        ax1.scatter(*self.log.path[0], s=30, c='k', marker='x', label='start')  # Vẽ điểm bắt đầu
        ax1.scatter(*self.log.end_pos, s=30, c='k', marker='o', label='target')  # Vẽ điểm đích
        ax1.plot(*np.array(self.log.path).T, color='b', label='path')  # Vẽ quỹ đạo
        ax1.legend(loc="best").set_draggable(True)

        # Vẽ đồ thị tín hiệu điều khiển
        ax2.set_title("Control Signal")
        ax2.set_xlabel("time step")
        ax2.set_ylabel("control")
        ctrl = np.array(self.log.ctrl).T
        for i, u in enumerate(ctrl):
            ax2.plot(u, label=f'u{i}')  # Vẽ các tín hiệu điều khiển
        ax2.legend(loc="best").set_draggable(True)

        # Vẽ đồ thị tín hiệu tốc độ
        ax3.set_title("Speed Signal")
        ax3.set_xlabel("time step")
        ax3.set_ylabel("speed")
        ax3.plot(self.log.speed, label='V')  # Vẽ tín hiệu tốc độ
        ax3.legend(loc="best").set_draggable(True)

        # Vẽ đồ thị tín hiệu chiều dài (voyage và distance)
        ax4.set_title("Length Signal")
        ax4.set_xlabel("time step")
        ax4.set_ylabel("length")
        length = np.array(self.log.length).T
        ax4.plot(length[0], label='voyage')  # Vẽ chiều dài quãng đường đã đi
        ax4.plot(length[1], label='distance')  # Vẽ khoảng cách đến đích
        ax4.legend(loc="best").set_draggable(True)

        plt.tight_layout()  # Điều chỉnh layout cho vừa vặn
        fig.savefig(fname=file, dpi=dpi)  # Lưu hình ảnh vào tệp với chất lượng dpi
        plt.close("Output")  # Đóng cửa sổ đồ họa

    
    @staticmethod
    def _limit_angle(x, domain=1):
        """Hàm giới hạn góc x trong một phạm vi:
        domain=1 giới hạn trong phạm vi (-π, π],
        domain=2 giới hạn trong phạm vi [0, 2π)"""
        x = x - x // (2 * math.pi) * 2 * math.pi  # Bất kỳ góc nào -> [0, 2π)
        if domain == 1 and x > math.pi:
            return x - 2 * math.pi  # [0, 2π) -> (-π, π]
        return x

    @staticmethod
    def _linear_mapping(x, x_min, x_max, left=0.0, right=1.0):
        """Chuyển đổi x theo phép biến đổi tuyến tính: [x_min, x_max] -> [left, right]"""
        y = left + (right - left) / (x_max - x_min) * (x - x_min)
        return y

    @staticmethod
    def _normalize_points(points, max_range=SCAN_RANGE):
        """Chuẩn hóa các điểm đo từ radar (biến điểm không hợp lệ thành 0, điểm hợp lệ vào khoảng 0.1~1)"""
        points = np.array(points)
        points[points > -0.5] = 0.9 * points[points > -0.5] / max_range + 0.1  # Điểm hợp lệ (nhỏ hơn max_range)
        points[points < -0.5] = 0.0  # Điểm không hợp lệ (giới hạn là -0.5)
        return points

    @staticmethod
    def _vector_angle(x_vec, y_vec, EPS=1e-8):
        """Tính góc giữa hai vector x_vec và y_vec trong phạm vi [0, π]"""
        x = np.linalg.norm(x_vec) * np.linalg.norm(y_vec)
        y = np.dot(x_vec, y_vec)
        if x < EPS:  # Trường hợp với vector 0
            return 0.0
        if y < EPS:  # Trường hợp góc 90°
            return math.pi / 2
        return math.acos(np.clip(y / x, -1, 1))  # Lưu ý: khi x rất nhỏ, có thể vượt quá ±1

    @staticmethod
    def _compute_azimuth(pos1, pos2, use_3d_pos=False):
        """Tính toán phương vị và góc độ của pos2 so với pos1 trong hệ tọa độ Đông-Tây-Nam. 
        Nếu use_3d_pos=True, tính cả góc độ và góc cao 3D, nếu không thì chỉ tính phương vị"""
        if use_3d_pos:
            x, y, z = np.array(pos2) - pos1
            q = math.atan(y / (math.sqrt(x ** 2 + z ** 2) + 1e-8))  # Góc độ [-π/2, π/2]
            ε = math.atan2(-z, x)  # Phương vị [-π, π]
            return ε, q
        else:
            x, z = np.array(pos2) - pos1
            return math.atan2(-z, x)

    @staticmethod
    def _fixed_wing_2d(s, t, u):
        """Mô hình động lực học chuyển động máy bay cánh cố định trong hệ tọa độ Đông-Tây-Nam (phiên bản đơn giản của mô hình 3D).
        s = [x, z, V, ψ]
        u = [nx, μ]"""
        _, _, V, ψ = s
        nx, μ = u
        dsdt = [
            V * math.cos(ψ),
            -V * math.sin(ψ),
            9.8 * nx,
            -9.8 / V * math.tan(μ)  # μ<90, không có trường hợp vô cùng
        ]
        return dsdt

    
    # @staticmethod
    # def _fixed_wing_3d(s, t, u):
    #     """Mô hình động lực học chuyển động máy bay cánh cố định trong không gian 3D hệ tọa độ Đông-Tây-Nam.
    #     s = [x, y, z, V, θ, ψ]  # Vị trí (x, y, z), vận tốc (V), góc nghiêng (θ), và phương vị (ψ)
    #     u = [nx, ny, μ]  # Lực điều khiển (nx, ny), góc nghiêng (μ)
    #     """
    #     _, _, _, V, θ, ψ = s  # Giải nén các tham số trạng thái
    #     nx, ny, μ = u  # Giải nén các tham số điều khiển
    #     if abs(math.cos(θ)) < 0.01:
    #         dψdt = 0  # Nếu góc nghiêng θ gần 90°, không thể tính tích phân dψ/dt
    #     else:
    #         dψdt = -9.8 * ny * math.sin(μ) / (V * math.cos(θ))  # Tính dψ/dt (thay đổi phương vị)
    #     dsdt = [
    #         V * math.cos(θ) * math.cos(ψ),  # Tính tốc độ chuyển động trong phương x
    #         V * math.sin(θ),  # Tính tốc độ chuyển động trong phương y
    #         -V * math.cos(θ) * math.sin(ψ),  # Tính tốc độ chuyển động trong phương z
    #         9.8 * (nx - math.sin(θ)),  # Tính gia tốc trong phương x (bao gồm trọng lực)
    #         9.8 / V * (ny * math.cos(μ) - math.cos(θ)),  # Tính gia tốc trong phương y
    #         dψdt  # Tính thay đổi phương vị
    #     ]
    #     return dsdt  # Trả về đạo hàm của trạng thái theo thời gian

    @classmethod
    def _ode45(cls, s_old, u, dt):
        """Giải phương trình vi phân bằng phương pháp tích phân ode45"""
        # Sử dụng phương pháp odeint để tích phân phương trình động lực học của máy bay cánh cố định 2D
        s_new = odeint(cls._fixed_wing_2d, s_old, (0.0, dt), args=(u, ))  # shape=(len(t), len(s))
        
        # Lấy trạng thái tại thời điểm cuối cùng sau khi tích phân
        x, z, V, ψ = s_new[-1]
        
        # Giới hạn vận tốc trong phạm vi cho phép
        V = np.clip(V, V_LOW, V_HIGH)
        
        # Giới hạn góc phương vị ψ trong phạm vi hợp lệ
        ψ = cls._limit_angle(ψ)
        
        # Trả về trạng thái mới dưới dạng mảng numpy với kiểu dữ liệu float32
        return np.array([x, z, V, ψ], dtype=np.float32)  # deepcopy


    







'''
------------------------↑↑↑↑↑ Môi trường tránh chướng ngại động ↑↑↑↑↑---------------------------
#                                          ...
#                                       .=BBBB#-
#                                      .B%&&&&&&#
#                                .=##-.#&&&&&#%&%
#                              -B&&&&&#=&&&&&B=-.
#                             =&@&&&&&&==&&&&@B
#                           -%@@@&&&&&&&.%&&&&@%.
#                          =&@@@%%@&&&&@#=B&&@@@%
#                         =@@@$#.%@@@@@@=B@@&&@@@-
#                         .&@@@%&@@@@@@&-&@@@%&@@=
#                            #&@@@&@@@@@B=@@@@&B@@=
#                             -%@@@@@@@#B@@@@@B&@-
#                              .B%&&&&@B&@@@@@&@#
#                             #B###BBBBBBB%%&&%#
#                            .######BBBBBBBBBB.
#                            =####BBBBBBBBBBBB#-
#                          .=####BB%%B%%%%%%BB##=
#                         .=##BBB%%#-  -#%%%BBB##.
#                        .=##BBB%#.      .#%%BBBB#.
#                        =##BB%%-          =%%BBBB=
#                       =#BB%%B-            .B%%%B#-
#                      =##BBB-                -BB###.
#                     -=##BB-                  -##=#-
#                     ==##B=-                  -####=
#                     =##B#-                   -####=
#                     ###B=                     =###=
#                    =##B#-                      ###=
#                    =BB#=                       =BB=
#                   -%&%                         =&&#
#                   %&%%                         B%&&=
---------------------------↓↓↓↓↓ Môi trường tránh chướng ngại tĩnh ↓↓↓↓↓---------------------------
'''










#----------------------------- ↓↓↓↓↓ Môi trường tìm kiếm đường đi ↓↓↓↓↓ ------------------------------#
class StaticPathPlanning(gym.Env):
    """Lập kế hoạch từ góc độ tìm kiếm điểm hành trình"""

    def __init__(self, num_pos=6, max_search_steps=200, old_gym_style=True):
        """
        Args:
            num_pos (int): Số điểm hành trình giữa điểm bắt đầu và điểm kết thúc. Mặc định là 6.
            max_search_steps (int): Số bước tìm kiếm tối đa. Mặc định là 200.
            old_gym_style (bool): Sử dụng giao diện Gym cũ hay không. Mặc định là True.
        """
        self.num_pos = num_pos
        self.map = MAP  # Mô hình bản đồ (có thể là một đối tượng hoặc dữ liệu đã được định nghĩa sẵn)
        self.max_episode_steps = max_search_steps

        lb = np.array(self.map.size[0] * num_pos)  # Giới hạn dưới của không gian quan sát
        ub = np.array(self.map.size[1] * num_pos)  # Giới hạn trên của không gian quan sát
        self.observation_space = spaces.Box(lb, ub, dtype=np.float32)  # Không gian quan sát
        self.action_space = spaces.Box(lb/10, ub/10, dtype=np.float32)  # Không gian hành động

        self.__render_not_called = True
        self.__need_reset = True
        self.__old_gym = old_gym_style  # Kiểm tra xem có sử dụng giao diện Gym cũ hay không

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Thiết lập phông chữ cho matplotlib
        plt.rcParams['axes.unicode_minus'] = False  # Đảm bảo dấu trừ hiển thị đúng
        plt.close("all")  # Đóng tất cả cửa sổ vẽ matplotlib

    def reset(self):
        self.__need_reset = False
        self.time_steps = 0  # NOTE: Để tránh bị nhầm lẫn với phương thức step, sử dụng tên khác
        self.obs = self.observation_space.sample()  # Lấy một quan sát ngẫu nhiên từ không gian quan sát
        # Gym mới: trả về quan sát, thông tin
        # Gym cũ: chỉ trả về quan sát
        if self.__old_gym:
            return self.obs  # Nếu là giao diện cũ, chỉ trả về quan sát
        return self.obs, {}  # Nếu là giao diện mới, trả về cả quan sát và thông tin rỗng

    
    def step(self, act):
        """
        Mô hình chuyển trạng thái 1
        Pos_new = act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new)
        
        Mô hình chuyển trạng thái 2
        Pos_new = Pos_old + act, act = Actor(Pos_old)
        Q = Critic(Pos_old, act) = Critic(Pos_old, Pos_new-Pos_old)
        """
        assert not self.__need_reset, "Phải gọi reset trước khi gọi step"
        # Chuyển trạng thái
        obs = np.clip(self.obs + act, self.observation_space.low, self.observation_space.high)
        self.time_steps += 1
        # Tính toán phần thưởng
        rew, done, info = self._get_reward(obs)
        # Kết thúc tập
        truncated = self.time_steps >= self.max_episode_steps
        if truncated or done:
            info["terminal"] = True
            self.__need_reset = True
        else:
            info["terminal"] = False
        # Cập nhật trạng thái
        self.obs = deepcopy(obs)
        # Gym mới: obs, rew, done, truncated, info
        # Gym cũ: obs, rew, done, info
        if self.__old_gym:
            return obs, rew, done, info
        return obs, rew, done, truncated, info

    
    def _get_reward(self, obs):
        traj = np.array(self.map.start_pos + obs.tolist() + self.map.end_pos) # [x,y,x,y,x,y,...]
        traj = traj.reshape(self.num_pos+2, 2) # [[x,y],[x,y],...]

        # Phạt
        num_over = 0 # Số trạng thái vượt quá giới hạn
        for o, l, u in zip(obs, self.observation_space.low, self.observation_space.high):
            if o <= l or o >= u: # Nếu trạng thái vượt ngoài phạm vi cho phép
                num_over += 2 # Tăng mức phạt

        # Kiểm tra quỹ đạo
        d = 0.0      # Tổng chiều dài -> Đánh giá năng lượng
        dθall = 0.0  # Tổng thay đổi góc -> Đánh giá năng lượng
        θ_last = 0.0 # Thay đổi góc từ lần trước < 45 độ
        num_theta = 0 # Số lần thay đổi góc không hợp lý -> Phạt
        num_crash = 0 # Số lần va chạm -> Phạt
        for i in range(len(traj) - 1):
            vec = traj[i+1] - traj[i]   # Vecto quỹ đạo (2, )
            # Cộng dồn chiều dài
            d += np.linalg.norm(vec) 
            # Kiểm tra góc quay
            θ = math.atan2(vec[1], vec[0])
            dθ = abs(θ - θ_last) if i != 0 else 0.0 # Đo thay đổi góc, lần đầu không có thay đổi
            if dθ >= math.pi / 4: 
                num_theta += 1 # Nếu thay đổi góc lớn hơn 45 độ, coi như không hợp lý
            dθall += dθ
            θ_last = deepcopy(θ)
            # Kiểm tra va chạm
            for o in self.map.obstacles:
                line = geo.LineString(traj[i:i+2])
                if o.intersects(line): # Kiểm tra xem có va chạm với chướng ngại vật không
                    num_crash += 1
            # Kết thúc vòng kiểm tra va chạm
        # Kết thúc vòng quỹ đạo
        
        # Tổng phần thưởng
        rew = -d - dθall / self.num_pos - num_theta - num_crash - num_over
        # Kiểm tra xem có kết thúc hay không
        if num_theta == 0 and num_crash == 0:
            rew += 100  # Phần thưởng cho việc hoàn thành quỹ đạo hợp lý
            done = True  # Quỹ đạo hợp lý
        else:
            done = False  # Quỹ đạo không hợp lý
        
        info = {}
        info['Số lần va chạm'] = num_crash # Thông tin về số lần va chạm
        info['Số lần thay đổi góc không hợp lý'] = num_theta # Thông tin về số lần thay đổi góc không hợp lý
        info['done'] = done # Trạng thái kết thúc của bài toán
        
        return rew, done, info

    
    def render(self, mode="human"):
        """Hiển thị môi trường, gọi luân phiên với step"""
        assert not self.__need_reset, "Phải gọi reset trước khi gọi render"
        if self.__render_not_called:
            self.__render_not_called = False
            plt.ion() # Mở chế độ vẽ đồ thị tương tác, chỉ có thể bật một lần
        # Tạo cửa sổ
        plt.clf()
        plt.axis('equal')
        plt.xlim(self.map.size[0][0], self.map.size[1][0])
        plt.ylim(self.map.size[1][1], self.map.size[0][1]) # Lưu ý: đổi thứ tự min/max có thể đảo ngược trục tọa độ
        # Vẽ các đối tượng
        for o in self.map.obstacles:
            plot_polygon(o, facecolor='w', edgecolor='k', add_points=False)
        plt.scatter(self.map.start_pos[0], self.map.start_pos[1], s=30, c='k', marker='x', label='Điểm bắt đầu')
        plt.scatter(self.map.end_pos[0], self.map.end_pos[1], s=30, c='k', marker='o', label='Điểm kết thúc')
        traj = self.map.start_pos + self.obs.tolist() + self.map.end_pos # [x,y,x,y,x,y,...]
        plt.plot(traj[::2], traj[1::2], label='Đường đi', color='b')
        # Thiết lập thông tin
        plt.title('Môi trường Tìm đường')
        plt.legend(loc='best')
        plt.xlabel("x")
        plt.ylabel("z")
        plt.grid(alpha=0.3, ls=':')
        # Đóng cửa sổ
        plt.pause(0.001)
        plt.ioff()

    def close(self):
        """Đóng môi trường"""
        self.__render_not_called = True
        self.__need_reset = True
        plt.close()

    def plot(self, *args, **kwargs):
        """Xuất đồ thị"""
        print("Bạn đang làm gì vậy?")













#----------------------------- ↓↓↓↓↓ Môi trường - Thích nghi thuật toán ↓↓↓↓↓ ------------------------------#
class NormalizedActionsWrapper(gym.ActionWrapper):
    """Bộ trang trí môi trường với không gian hành động liên tục ngoài phạm vi [-1, 1]"""
    
    def __init__(self, env):
        super(NormalizedActionsWrapper, self).__init__(env)
        assert isinstance(env.action_space, spaces.Box), 'Chỉ áp dụng cho không gian hành động kiểu Box'
  
    # Chuyển đầu ra của mạng nơ-ron thành định dạng đầu vào của gym
    def action(self, action): 
        # Trường hợp liên tục: scale action [-1, 1] -> [lb, ub]
        lb, ub = self.action_space.low, self.action_space.high
        action = lb + (action + 1.0) * 0.5 * (ub - lb)  # Điều chỉnh hành động về khoảng [lb, ub]
        action = np.clip(action, lb, ub)  # Giới hạn giá trị trong khoảng [lb, ub]
        return action

    # Chuyển đổi đầu vào của gym thành đầu ra của mạng nơ-ron
    def reverse_action(self, action):
        # Trường hợp liên tục: hành động chuẩn hóa [lb, ub] -> [-1, 1]
        lb, ub = self.action_space.low, self.action_space.high
        action = 2 * (action - lb) / (ub - lb) - 1  # Chuẩn hóa hành động về phạm vi [-1, 1]
        return np.clip(action, -1.0, 1.0)  # Giới hạn hành động trong phạm vi [-1, 1]

       








if __name__ == '__main__':
    # MAP.show()  # Nếu cần, có thể bật hàm này để hiển thị bản đồ
    env = DynamicPathPlanning()  # Khởi tạo môi trường DynamicPathPlanning
    for ep in range(10):  # Lặp qua 10 tập (episode)
        print(f"episode{ep}: begin")  # In ra thông báo bắt đầu tập
        obs = env.reset()  # Khởi tạo lại môi trường và lấy quan sát ban đầu
        while 1:  # Tiến hành chạy môi trường cho đến khi hoàn thành
            try:
                env.render()  # Hiển thị môi trường (vẽ trạng thái hiện tại)
                obs, rew, done, info = env.step(np.array([0.5, 0.2]))  # Chọn hành động (ở đây là [0.5, 0.2])
                print(info)  # In ra thông tin từ môi trường (ví dụ: va chạm, không hợp lý...)
            except AssertionError:
                break  # Nếu gặp lỗi AssertionError, kết thúc tập này (thường là do chưa gọi reset())
        # env.plot(f"output{ep}")  # Nếu cần, có thể lưu hoặc vẽ kết quả
        print(f"episode{ep}: end")  # In ra thông báo kết thúc tập
