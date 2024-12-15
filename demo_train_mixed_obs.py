# -*- coding: utf-8 -*-
"""
Ví dụ huấn luyện chiến lược
 Tạo vào ngày Sat Nov 04 2023 15:37:28
 Sửa đổi vào 2023-11-4 15:37:28
 
 @tác giả: HJ https://github.com/zhaohaojie1998
"""
#


'''Định nghĩa thuật toán'''
import numpy as np
import torch as th
import torch.nn as nn
from copy import deepcopy
from sac_agent import *
# 1. Định nghĩa bộ nhớ hồi tưởng (Tùy thuộc vào cấu trúc dữ liệu quan sát và hành động)
class Buffer(BaseBuffer):
    def __init__(self, memory_size, obs_space, act_space):
        super(Buffer, self).__init__()
        # Kiểu dữ liệu
        self.device = 'cuda'
        self.obs_space = obs_space
        self.act_space = act_space
        # Thuộc tính bộ nhớ
        self._ptr = 0    # Vị trí hiện tại
        self._idxs = [0] # PER ghi nhớ vị trí mẫu lần trước, danh sách một chiều hoặc ndarray
        self._memory_size = int(memory_size) # Dung lượng tổng
        self._current_size = 0               # Dung lượng hiện tại
        # Bộ nhớ container
        self._data = {}
        self._data["points"] = np.empty((memory_size, *obs_space['seq_points'].shape), dtype=obs_space['seq_points'].dtype)
        self._data["points_"] = deepcopy(self._data["points"])
        self._data["vector"] = np.empty((memory_size, *obs_space['seq_vector'].shape), dtype=obs_space['seq_vector'].dtype)
        self._data["vector_"] = deepcopy(self._data["vector"])
        self._data["a"] = np.empty((memory_size, *act_space.shape), dtype=act_space.dtype)
        self._data["r"] = np.empty((memory_size, 1), dtype=np.float32)
        self._data["done"] = np.empty((memory_size, 1), dtype=bool)
    
    def reset(self, *args, **kwargs):
        self._ptr = 0
        self._idxs = [0]
        self._current_size = 0

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self._data.values())

    def push(self, transition, terminal=None, **kwargs):
        self._data["points"][self._ptr] = transition[0]['seq_points']
        self._data["vector"][self._ptr] = transition[0]['seq_vector']
        self._data["a"][self._ptr] = transition[1]
        self._data["r"][self._ptr] = transition[2]
        self._data["points_"][self._ptr] = transition[3]['seq_points']
        self._data["vector_"][self._ptr] = transition[3]['seq_vector']
        self._data["done"][self._ptr] = transition[4]
        # Cập nhật
        self._ptr = (self._ptr + 1) % self._memory_size                     # Cập nhật con trỏ
        self._current_size = min(self._current_size + 1, self._memory_size) # Cập nhật dung lượng

    def __len__(self):
        return self._current_size 
    
    def sample(self, batch_size=1, *, idxs=None, rate=None, **kwargs):
        self._idxs = idxs or np.random.choice(self._current_size, size=batch_size, replace=False)
        batch = {
            's': {
                'seq_points': th.FloatTensor(self._data['points'][self._idxs]).to(self.device),
                'seq_vector': th.FloatTensor(self._data['vector'][self._idxs]).to(self.device),
            },
            'a': th.FloatTensor(self._data['a'][self._idxs]).to(self.device),
            'r': th.FloatTensor(self._data['r'][self._idxs]).to(self.device),
            's_': {
                'seq_points': th.FloatTensor(self._data['points_'][self._idxs]).to(self.device),
                'seq_vector': th.FloatTensor(self._data['vector_'][self._idxs]).to(self.device),
            },
            'done': th.FloatTensor(self._data['done'][self._idxs]).to(self.device),
        }
        return batch
    
    def state_to_tensor(self, state, use_rnn=False):
        return {'seq_points': th.FloatTensor(state['seq_points']).unsqueeze(0).to(self.device),
                'seq_vector': th.FloatTensor(state['seq_vector']).unsqueeze(0).to(self.device)}

    

# 2. Định nghĩa mạng nơ-ron (Tùy thuộc vào cấu trúc dữ liệu quan sát)
# Bộ mã hóa quan sát hỗn hợp
class EncoderNet(nn.Module):
    def __init__(self, obs_space, feature_dim):
        super(EncoderNet, self).__init__()
        # Mã hóa điểm mây
        c, cnn_dim = obs_space['seq_points'].shape
        in_kernel_size = min(cnn_dim//2, 8)
        in_stride = min(cnn_dim - in_kernel_size, 4)
        self.cnn = nn.Sequential(
            nn.Conv1d(c, 32, kernel_size=in_kernel_size, stride=in_stride, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        cnn_out_dim = self._get_cnn_out_dim(self.cnn, (c, cnn_dim))
        self.cnn_mlp = nn.Sequential(
            nn.Linear(cnn_out_dim, feature_dim),
            nn.ReLU(True),
        )
        # Mã hóa trạng thái vector
        _, rnn_dim = obs_space['seq_vector'].shape
        rnn_hidden_dim = 256
        rnn_num_layers = 1
        self.rnn_mlp1 = nn.Sequential(
            nn.Linear(rnn_dim, rnn_hidden_dim),
            nn.ReLU(True),
        )
        self.rnn = nn.GRU(rnn_hidden_dim, rnn_hidden_dim, rnn_num_layers, batch_first=True)
        self.rnn_mlp2 = nn.Sequential(
            nn.Linear(rnn_hidden_dim, feature_dim),
            nn.ReLU(True),
        )
        # Mạng kết hợp đặc trưng
        self.fusion = nn.Sequential(
            nn.Linear(2 * feature_dim, feature_dim),
            nn.ReLU(True),
        )

    def forward(self, obs):
        f1 = self.cnn_mlp(self.cnn(obs['seq_points']))  # batch, dim
        f2_n, _ = self.rnn(self.rnn_mlp1(obs['seq_vector']), None)  # batch, seq, dim
        f2 = self.rnn_mlp2(f2_n[:, -1, :])  # batch, dim
        return self.fusion(th.cat([f1, f2], dim=-1))  # batch, dim
    
    @staticmethod
    def _get_cnn_out_dim(cnn: nn.Module, input_shape: tuple[int, ...]):
        # out_dim = (in_dim + 2*pad - dilation*(k_size-1) -1 ) / stride + 1
        cnn_copy = deepcopy(cnn).to('cpu')
        output = cnn_copy(th.zeros(1, *input_shape))
        return int(np.prod(output.size()))

# Hàm Q
class QNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(QNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + act_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 1),
        )

    def forward(self, feature_and_action):
        return self.mlp(feature_and_action)

# Hàm chính sách
class PiNet(nn.Module):
    def __init__(self, feature_dim, act_dim):
        super(PiNet, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, act_dim),
        )

    def forward(self, feature):
        return self.mlp(feature)



'''Khởi tạo môi trường'''
from path_plan_env import DynamicPathPlanning
env = DynamicPathPlanning()
obs_space = env.observation_space
act_space = env.action_space



'''Khởi tạo thuật toán'''
# 1. Cài đặt bộ nhớ
buffer = Buffer(100000, obs_space, act_space)

# 2. Cài đặt mạng nơ-ron
actor = SAC_Actor(
        EncoderNet(obs_space, 256),
        PiNet(256, act_space.shape[0]),
        PiNet(256, act_space.shape[0]),
    )
critic = SAC_Critic(
        EncoderNet(obs_space, 256),
        QNet(256, act_space.shape[0]),
        QNet(256, act_space.shape[0]),
    )


# 3. Cài đặt thuật toán
agent = SAC_Agent(env, batch_size=2048)
agent.set_buffer(buffer)
agent.set_nn(actor, critic)
agent.cuda()

'''VÒNG LẶP HUẤN LUYỆN'''
from torch.utils.tensorboard import SummaryWriter # TensorBoard, khởi động!!!
log = SummaryWriter(log_dir = "./tb_log") 

MAX_EPISODE = 100
LEARN_FREQ = 100
OUTPUT_FREQ = 50
for episode in range(MAX_EPISODE):
    ## Đặt lại phần thưởng cho mỗi vòng
    ep_reward = 0
    ## Lấy quan sát ban đầu
    obs = env.reset()
    ## Tiến hành một vòng mô phỏng
    for steps in range(env.max_episode_steps):
        # Quyết định hành động
        act = agent.select_action(obs)
        # Mô phỏng môi trường
        next_obs, reward, done, info = env.step(act)
        ep_reward += reward
        # Lưu vào bộ nhớ
        agent.store_memory((obs, act, reward, next_obs, done))
        # Kiểm tra kết thúc vòng
        if info["terminal"]:
            mean_reward = ep_reward / (steps + 1)
            print('Vòng lặp: ', episode, '| Tổng phần thưởng: ', round(ep_reward, 2), '| Phần thưởng trung bình: ', round(mean_reward, 2), '| Trạng thái: ', info, '| Số bước: ', steps)
            break
        else:
            obs = deepcopy(next_obs)
    # Kết thúc vòng lặp

    ## Ghi lại dữ liệu vào TensorBoard
    log.add_scalar('Return', ep_reward, episode)
    log.add_scalar('MeanReward', mean_reward, episode)
    
    ## Huấn luyện mô hình
    if episode % LEARN_FREQ == 0:
        train_info = agent.learn()
    if episode % OUTPUT_FREQ == 0:
        env.plot(f"./output/out{episode}.png")

# Kết thúc vòng lặp huấn luyện
agent.export("./path_plan_env/policy_dynamic.onnx") # Xuất mô hình chính sách
agent.save("./checkpoint") # Lưu tiến trình huấn luyện thuật toán
agent.load("./checkpoint") # Tải lại tiến trình huấn luyện thuật toán








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
            Phật tổ bảo vệ       Mãi không có BUG
'''
