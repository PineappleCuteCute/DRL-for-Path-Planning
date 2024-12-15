# -*- coding: utf-8 -*-
"""
Môi trường học tăng cường cho bài toán lập kế hoạch đường đi.

Tạo: Ngày 12 tháng 12 năm 2024, 17:54:17
Cập nhật: Ngày 12 tháng 12 năm 2024

Tác giả: Đào Thành Mạnh
GitHub: https://github.com/PineappleCuteCute
"""
#

# 1. Khởi tạo môi trường
from path_plan_env import StaticPathPlanning, NormalizedActionsWrapper
env = NormalizedActionsWrapper(StaticPathPlanning())


# 2. Tải chiến lược
import onnxruntime as ort
policy = ort.InferenceSession("./path_plan_env/policy_static.onnx")


# 3. Vòng lặp mô phỏng
from copy import deepcopy

MAX_EPISODE = 20
for episode in range(MAX_EPISODE):
    ## Lấy quan sát ban đầu
    obs = env.reset()
    ## Thực hiện một vòng mô phỏng
    for steps in range(env.max_episode_steps):
        # Hiển thị trực quan
        env.render()
        # Quyết định
        obs = obs.reshape(1, *obs.shape)                      # (*shape, ) -> (1, *shape, )
        act = policy.run(['action'], {'observation': obs})[0] # Trả về [action, ...]
        act = act.flatten()                                   # (1, dim, ) -> (dim, )
        # Mô phỏng
        next_obs, _, _, info = env.step(act)
        # Vòng kết thúc
        if info["terminal"]:
            print('Vòng: ', episode,'| Trạng thái: ', info,'| Số bước: ', steps) 
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
