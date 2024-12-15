# -*- coding: utf-8 -*-
"""
Môi trường học tăng cường cho bài toán lập kế hoạch đường đi.

Tạo: Ngày 12 tháng 12 năm 2024, 17:54:17
Cập nhật: Ngày 12 tháng 12 năm 2024

Tác giả: Đào Thành Mạnh
GitHub: https://github.com/PineappleCuteCute
"""
#


'''Định nghĩa thuật toán'''
import numpy as np
import torch as th
import torch.nn as nn
from copy import deepcopy
from sac_agent import *
# 1. Định nghĩa bộ nhớ trải nghiệm (tùy thuộc vào cấu trúc dữ liệu quan sát và hành động)
class Buffer(BaseBuffer):
    def __init__(self, memory_size, obs_space, act_space):
        super(Buffer, self).__init__()
        # Biểu diễn kiểu dữ liệu
        self.device = 'cuda'
        self.obs_space = obs_space
        self.act_space = act_space
        # Thuộc tính bộ nhớ
        self._ptr = 0    # Vị trí hiện tại
        self._idxs = [0] # PER ghi nhớ vị trí mẫu lần trước, danh sách một chiều hoặc ndarray
        self._memory_size = int(memory_size) # Tổng dung lượng bộ nhớ
        self._current_size = 0               # Dung lượng hiện tại
        # Container bộ nhớ
        obs_shape = obs_space.shape or (1, )
        act_shape = act_space.shape or (1, ) # LƯU Ý: Shape của DiscreteSpace là (), cần đặt collections là (1, )
        self._data = {}
        self._data["s"] = np.empty((memory_size, *obs_shape), dtype=obs_space.dtype) # (size, *obs_shape, ) liên tục (size, 1) rời rạc
        self._data["s_"] = deepcopy(self._data["s"])                                 # (size, *obs_shape, ) liên tục (size, 1) rời rạc
        self._data["a"] = np.empty((memory_size, *act_shape), dtype=act_space.dtype) # (size, *act_shape, ) liên tục (size, 1) rời rạc
        self._data["r"] = np.empty((memory_size, 1), dtype=np.float32)               # (size, 1)
        self._data["done"] = np.empty((memory_size, 1), dtype=bool)                  # (size, 1) 

    def reset(self, *args, **kwargs):
        self._ptr = 0
        self._idxs = [0]
        self._current_size = 0

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self._data.values())

    def push(self, transition, terminal=None, **kwargs):
        self._data["s"][self._ptr] = transition[0]
        self._data["a"][self._ptr] = transition[1]
        self._data["r"][self._ptr] = transition[2]
        self._data["s_"][self._ptr] = transition[3]
        self._data["done"][self._ptr] = transition[4]
        # Cập nhật
        self._ptr = (self._ptr + 1) % self._memory_size                     # Cập nhật con trỏ
        self._current_size = min(self._current_size + 1, self._memory_size) # Cập nhật dung lượng

    def __len__(self):
        return self._current_size 
    
    def sample(self, batch_size=1, *, idxs=None, rate=None, **kwargs):
        self._idxs = idxs or np.random.choice(self._current_size, size=batch_size, replace=False)
        batch = {k: th.FloatTensor(self._data[k][self._idxs]).to(self.device) for k in self._data.keys()}
        return batch
    
    def state_to_tensor(self, state, use_rnn=False):
        return th.FloatTensor(state).unsqueeze(0).to(self.device)

    
# 2. Định nghĩa mạng thần kinh (tùy thuộc vào cấu trúc dữ liệu quan sát)
# Mạng Q
QEncoderNet = nn.Identity

class QNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(QNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + act_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 1),
        )
    def forward(self, feature_and_action):
        return self.mlp(feature_and_action)

# Mạng Pi
class PiEncoderNet(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super(PiEncoderNet, self).__init__()
        obs_dim = np.prod(obs_shape)
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, feature_dim),
            nn.ReLU(True),
        )
    def forward(self, obs):
        return self.mlp(obs)
    
class PiNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(PiNet, self).__init__()
        self.mlp = nn.Linear(feature_dim, act_dim)
    def forward(self, feature):
        return self.mlp(feature)



'''Khởi tạo môi trường'''
from path_plan_env import StaticPathPlanning, NormalizedActionsWrapper
env = NormalizedActionsWrapper(StaticPathPlanning())
obs_space = env.observation_space
act_space = env.action_space



'''Khởi tạo thuật toán'''
# 1. Cài đặt bộ nhớ
buffer = Buffer(10000, obs_space, act_space)

# 2. Cài đặt mạng thần kinh
actor = SAC_Actor(
        PiEncoderNet(obs_space.shape, 128),
        PiNet(128, act_space.shape[0]),
        PiNet(128, act_space.shape[0]),
    )
critic = SAC_Critic(
        QEncoderNet(),
        QNet(obs_space.shape[0], act_space.shape[0]),
        QNet(obs_space.shape[0], act_space.shape[0]),
    )


# 3. Cài đặt thuật toán
agent = SAC_Agent(env)
agent.set_buffer(buffer)
agent.set_nn(actor, critic)
agent.cuda()



'''Vòng lặp huấn luyện''' 
MAX_EPISODE = 2000
for episode in range(MAX_EPISODE):
    ## Đặt lại phần thưởng cho mỗi tập
    ep_reward = 0
    ## Lấy quan sát ban đầu
    obs = env.reset()
    ## Tiến hành mô phỏng một tập
    for steps in range(env.max_episode_steps):
        # Quyết định hành động
        act = agent.select_action(obs)
        # Mô phỏng
        next_obs, reward, done, info = env.step(act)
        ep_reward += reward
        # Lưu trữ vào bộ nhớ
        agent.store_memory((obs, act, reward, next_obs, done))
        # Tối ưu hóa
        agent.learn()
        # Kết thúc tập
        if info["terminal"]:
            mean_reward = ep_reward / (steps + 1)
            print('Tập: ', episode, '| Tổng phần thưởng: ', round(ep_reward, 2), '| Phần thưởng trung bình: ', round(mean_reward, 2), '| Trạng thái: ', info, '| Bước: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
    # Kết thúc vòng lặp
# Kết thúc vòng lặp các tập
agent.export("./path_plan_env/policy_static.onnx")  # Xuất mô hình chính sách
agent.save("./checkpoint")  # Lưu tiến trình huấn luyện thuật toán
agent.load("./checkpoint")  # Tải lại tiến trình huấn luyện thuật toán








r'''
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            Phật tổ bảo vệ    Không bao giờ có BUG

'''
