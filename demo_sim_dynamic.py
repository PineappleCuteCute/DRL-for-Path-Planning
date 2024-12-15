# -*- coding: utf-8 -*-
"""
Ví dụ về Lập kế hoạch đường đi động học (Quan sát hỗn hợp)
 Tạo vào ngày Wed Mar 13 2024 18:18:07
 Sửa đổi vào 2024-3-13 18:18:07
 
 @tác giả: HJ https://github.com/zhaohaojie1998
"""
#

# 1. Khởi tạo môi trường
from path_plan_env import DynamicPathPlanning
env = DynamicPathPlanning(800) # Không gian hành động mặc định là -1,1


# 2. Tải chiến lược
import onnxruntime as ort
policy = ort.InferenceSession("./path_plan_env/policy_dynamic.onnx")


# 3. Vòng lặp mô phỏng
from copy import deepcopy

MAX_EPISODE = 5
for episode in range(MAX_EPISODE):
    ## Lấy quan sát ban đầu
    obs = env.reset()
    ## Thực hiện một vòng mô phỏng
    for steps in range(env.max_episode_steps):
        # Hiển thị trực quan
        env.render()
        # Quyết định
        seq_points = obs['seq_points'].reshape(1, *obs['seq_points'].shape) # (1, seq_len, *points_shape, )
        seq_vector = obs['seq_vector'].reshape(1, *obs['seq_vector'].shape) # (1, seq_len, vector_dim, )
        act = policy.run(['action'], {'seq_points': seq_points, 'seq_vector': seq_vector})[0] # Trả về [action, ...]
        act = act.flatten()                                                                   # (1, dim, ) -> (dim, )
        # Mô phỏng
        next_obs, _, _, info = env.step(act)
        # Vòng kết thúc
        if info["terminal"]:
            print('Vòng: ', episode, '| Trạng thái: ', info, '| Số bước: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
    # kết thúc vòng for
# kết thúc vòng for




#             ⠰⢷⢿⠄
#         ⠀⠀⠀⠀⠀⣼⣷⣄
#         ⠀⠀⣤⣿⣇⣿⣿⣧⣿⡄
#         ⢴⠾⠋⠀⠀⠻⣿⣷⣿⣿⡀
#         🏀   ⢀⣿⣿⡿⢿⠈⣿
#          ⠀⠀⢠⣿⡿⠁⢠⣿⡊⠀⠙
#          ⠀⠀⢿⣿⠀⠀⠹⣿
#           ⠀⠀⠹⣷⡀⠀⣿⡄
#            ⠀⣀⣼⣿⠀⢈⣧ 
#
#       Bạn đang làm gì vậy...?
#       Haha... Ủa... 
