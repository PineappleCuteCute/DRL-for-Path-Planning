# -*- coding: utf-8 -*-
"""
Thuật toán SAC-Auto
 Được tạo vào ngày Fri Mar 03 2023 19:58:10
 Được chỉnh sửa vào 2023-3-3 19:58:
     
 Tác giả: HJ https://github.com/zhaohaojie1998
"""
# Chạy trên GPU #
from rl_typing import *
from abc import abstractmethod

import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal

from pathlib import Path
from copy import deepcopy

__all__ = [
    "BaseBuffer",
    "SAC_Critic",
    "SAC_Actor",
    "SAC_Agent",
]







#----------------------------- ↓↓↓↓↓ Bộ đệm Replay Kinh nghiệm ↓↓↓↓↓ ------------------------------#
class BaseBuffer:
    """Lớp Bộ đệm Replay cơ bản, cần hoàn thiện các chức năng cụ thể theo nhiệm vụ"""

    obs_space: ObsSpace
    act_space: ActSpace
    device: DeviceLike = 'cpu'

    # 0. Đặt lại
    @abstractmethod
    def reset(self, *args, **kwargs):
        """Đặt lại bộ đệm replay"""
        raise NotImplementedError
    
    @property
    def is_rnn(self) -> bool:
        """Kiểm tra xem có phải là replay với RNN hay không"""
        return False
    
    @property
    def nbytes(self) -> int:
        """Bộ đệm chiếm bao nhiêu bộ nhớ"""
        return 0
    
    # 1. Lưu trữ
    @abstractmethod
    def push(
        self, 
        transition: tuple[Obs, Act, float, Obs, bool], 
        terminal: bool = None, 
        **kwargs
    ):
        """Lưu một mẫu\n
            transition = (trạng thái, hành động, phần thưởng, trạng thái kế tiếp, kết thúc)
            terminal được dùng để điều khiển REPLAY EPISODE của DRQN
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        """Dung lượng hiện tại của bộ đệm"""
        return 0
    
    def extend(
        self, 
        transition_list: list[tuple[Obs, Act, float, Obs, bool]], 
        terminal_list: list[bool] = None, 
        **kwargs
    ):
        """Lưu một loạt các mẫu\n
            extend(List[(trạng thái, hành động, phần thưởng, trạng thái kế tiếp, kết thúc)], List[terminal])
        """
        for transition, terminal in zip(transition_list, terminal_list):
            self.push(transition, terminal)

    # 2. Lấy mẫu
    @abstractmethod
    def sample(
        self, 
        batch_size: int = 1, 
        *,
        idxs: ListLike = None,
        rate: float = None,
        **kwargs,
    ) -> dict[str, Union[ObsBatch, ActBatch, th.FloatTensor]]:
        """Lấy mẫu ngẫu nhiên

        Tham số
        ----------
        batch_size : int, tùy chọn
            Kích thước mẫu, mặc định là 1.
        
        KwArgs
        ----------
        idxs : ListLike, tùy chọn
            Nếu chỉ định các chỉ số mẫu, thì lấy mẫu theo chỉ số (trong trường hợp này batch_size không có tác dụng), nếu không, ngẫu nhiên tạo chỉ số mẫu, mặc định là None.
        rate : float, tùy chọn
            Dùng để cập nhật tham số beta trong PER, mặc định là None.
            rate = số bước học / số bước huấn luyện tối đa
            beta = beta0 + (1-beta0) * rate

        Trả về
        -------
        Dict[str, Union[ObsBatch, ActBatch, th.FloatTensor]]
            Trả về key là "s", "a", "r", "s_", "done", "IS_weight", ... dưới dạng Tensor/MixedTensor cho GPU
        """  
        raise NotImplementedError

    def __getitem__(self, index):
        """Truy cập mẫu theo chỉ số\n
           Tức là batch = buffer[index] và batch = buffer.sample(idxs=index) có hiệu ứng tương tự
        """
        if isinstance(index, int): index = [index]
        return self.sample(idxs=index)
    
    # 3. Chức năng PER
    def update_priorities(self, td_errors: np.ndarray):
        """Cập nhật độ ưu tiên của PER sử dụng sai số TD"""
        pass
    
    @property
    def is_per(self) -> bool:
        """Kiểm tra xem có phải là bộ đệm PER hay không"""
        return False
    
    # 4. Chuyển đổi một trạng thái thành mẫu
    @abstractmethod
    def state_to_tensor(self, state: Obs, use_rnn=False) -> ObsBatch:
        """Giao diện của thuật toán select_action và export, dùng để chuyển đổi một trạng thái thành tensor với batch_size=1
        use_rnn = False : (*state_shape, ) -> (1, *state_shape)
        use_rnn = True : (*state_shape, ) -> (1, 1, *state_shape)
        """
        raise NotImplementedError
        # TODO Nếu muốn hỗ trợ không gian hành động hỗn hợp, cần định nghĩa phương thức action_to_numpy
    
    # 5. Giao diện IO
    def save(self, data_dir: PathLike, buffer_id: Union[int, str] = None):
        """Lưu bộ đệm\n
        Lưu trong data_dir / buffer_id hoặc trong data_dir
        """
        pass

    def load(self, data_dir: PathLike, buffer_id: Union[int, str] = None):
        """Đọc bộ đệm\n
        Lưu trong data_dir / buffer_id hoặc trong data_dir
        """
        pass

    # 6. Chức năng PyTorch
    def to(self, device: DeviceLike):
        """Đặt tensor mẫu lên device"""
        self.device = device
        return self

    def cuda(self, cuda_id=None):
        """Đặt tensor mẫu thành tensor CUDA"""
        device = 'cpu' if not th.cuda.is_available() else 'cuda' if cuda_id is None else 'cuda:' + str(cuda_id)
        self.to(device)
        return self

    def cpu(self):
        """Đặt tensor mẫu thành tensor CPU"""
        self.to('cpu')
        return self

    
    
    


#----------------------------- ↓↓↓↓↓ Soft Actor-Critic ↓↓↓↓↓ ------------------------------#

# Mạng Q
class SAC_Critic(nn.Module):
    def __init__(self, encoder: nn.Module, q1_layer: nn.Module, q2_layer: nn.Module):
        """Thiết lập Critic của SAC\n
        Yêu cầu encoder đầu vào là obs, đầu ra là đặc trưng (batch, dim).\n
        Yêu cầu q1_layer và q2_layer đầu vào là (batch, dim + act_dim) - vector ghép [x, a], đầu ra là (batch, 1) của Q.\n
        """
        super().__init__()
        self.encoder_layer = deepcopy(encoder)
        self.q1_layer = deepcopy(q1_layer)
        self.q2_layer = deepcopy(q2_layer)

    def forward(self, obs, act):
        feature = self.encoder_layer(obs)  # (batch, feature_dim)
        x = th.cat([feature, act], -1)
        Q1 = self.q1_layer(x)
        Q2 = self.q2_layer(x)
        return Q1, Q2


# Mạng PI
class SAC_Actor(nn.Module):
    def __init__(self, encoder: nn.Module, mu_layer: nn.Module, log_std_layer: nn.Module, log_std_max=2.0, log_std_min=-20.0):
        """Thiết lập Actor của SAC\n
        Yêu cầu encoder đầu vào là obs, đầu ra là đặc trưng (batch, dim).\n
        Yêu cầu log_std_layer và mu_layer đầu vào là x, đầu ra là (batch, act_dim) của độ lệch chuẩn và giá trị trung bình.\n
        """
        super().__init__()
        self.encoder_layer = deepcopy(encoder)
        self.mu_layer = deepcopy(mu_layer)
        self.log_std_layer = deepcopy(log_std_layer)
        self.LOG_STD_MAX = log_std_max
        self.LOG_STD_MIN = log_std_min

    def forward(self, obs, deterministic=False, with_logprob=True):
        feature = self.encoder_layer(obs)  # (batch, feature_dim)
        mu = self.mu_layer(feature)
        log_std = self.log_std_layer(feature)
        log_std = th.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = th.exp(log_std)
        # Phân phối chính sách
        dist = Normal(mu, std)
        if deterministic:
            u = mu
        else:
            u = dist.rsample()
        a = th.tanh(u)
        # Tính toán log của xác suất hành động
        if with_logprob:
            # 1. Công thức log xác suất của a từ u trong bài báo SAC:
            "logp_pi_a = (dist.log_prob(u) - th.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)"
            # 2. Công thức trong SAC gốc có a = tanh(u), dẫn đến gradient biến mất, ta mở rộng công thức tanh:
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)  # (batch, 1)
        else:
            logp_pi_a = None
        return a, logp_pi_a  # (batch, act_dim) và (batch, 1)

    def act(self, obs, deterministic=False) -> np.ndarray[any, float]:  # NOTE Không hỗ trợ không gian hành động hỗn hợp
        self.eval()
        with th.no_grad():
            a, _ = self.forward(obs, deterministic, False)
        self.train()
        return a.cpu().numpy().flatten()  # (act_dim, ) ndarray


# Thuật toán SAC-Auto
class SAC_Agent:
    """Thuật toán Soft Actor-Critic (arXiv: 1812)"""
   
    def __init__(
        self, 
        env: GymEnv,                # Môi trường gym hoặc tham số cfg
        *,
        gamma: float = 0.99,        # Hệ số chiết khấu γ
        alpha: float = 0.2,         # Hệ số nhiệt α
        batch_size: int = 128,      # Kích thước mẫu
        update_after: int = 1000,   # Bước bắt đầu huấn luyện, batch_size <= update_after <= memory_size

        lr_decay_period: int = None, # Chu kỳ giảm học, None nếu không giảm
        lr_critic: float = 1e-3,     # Học tỷ lệ Q
        lr_actor: float = 1e-3,      # Học tỷ lệ π
        tau: float = 0.005,          # Hệ số cập nhật mềm của target Q
        q_loss_cls = nn.MSELoss,     # Loại hàm mất mát Q (không có tác dụng khi use_per=True)
        grad_clip: float = None,     # Phạm vi cắt gradient của mạng Q, None nếu không cắt

        adaptive_alpha: bool = True,     # Có tự điều chỉnh nhiệt độ hay không
        target_entropy: float = None,    # Entropy mục tiêu cho hệ số nhiệt, mặc định: -dim(A)
        lr_alpha: float = 1e-3,          # Học tỷ lệ α
        alpha_optim_cls = th.optim.Adam, # Loại bộ tối ưu cho α

        device: DeviceLike = th.device("cuda" if th.cuda.is_available() else "cpu"), # Thiết bị tính toán
    ): 
        """
        Tham số:
            env (GymEnv): Instance môi trường Gym, hoặc lớp dữ liệu chứa observation_space và action_space.
        KwArgs:
            gamma (float): Tỷ lệ chiết khấu phần thưởng tích lũy. Mặc định là 0.99.
            alpha (float): Hệ số nhiệt ban đầu. Mặc định là 0.2.
            batch_size (int): Kích thước mẫu. Mặc định là 128.
            update_after (int): Số bước bắt đầu huấn luyện. Mặc định là 1000.
            lr_decay_period (int): Chu kỳ giảm học xuống còn 0.1 lần. Mặc định là None (không giảm).
            lr_critic (float): Học tỷ lệ của Q function. Mặc định là 0.001.
            lr_actor (float): Học tỷ lệ của Pi function. Mặc định là 0.001.
            tau (float): Hệ số cập nhật mềm của Q target. Mặc định là 0.005.
            q_loss_cls (TorchLossClass): Hàm mất mát của Q function. Mặc định là MSELoss.
            grad_clip (float): Phạm vi cắt gradient cho Q function. Mặc định là None (không cắt).
            adaptive_alpha (bool): Có tự điều chỉnh hệ số nhiệt hay không. Mặc định là True.
            target_entropy (float): Entropy mục tiêu của chính sách. Mặc định là -dim(A).
            lr_alpha (float): Học tỷ lệ của hệ số nhiệt. Mặc định là 0.001.
            alpha_optim_cls (TorchOptimizerClass): Bộ tối ưu cho hệ số nhiệt. Mặc định là Adam.
            device (DeviceLike): Thiết bị huấn luyện. Mặc định là cuda0.
        """
        assert isinstance(env.action_space, GymBox), "Không gian hành động của SAC-Auto chỉ có thể là Box"
        self.device = device
        # Tham số môi trường
        self.obs_space = env.observation_space
        self.act_space = env.action_space
        self.num_actions = np.prod(self.act_space.shape)
        # Khởi tạo tham số SAC
        self.gamma = gamma
        self.batch_size = int(batch_size)
        self.update_after = int(update_after)
        # Khởi tạo tham số DL
        self.lr_decay_period = lr_decay_period
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        # Khởi tạo ReplayBuffer
        self.__set_buffer = False
        self.buffer = BaseBuffer()
        # Khởi tạo mạng nơ-ron
        self.__set_nn = False
        self.actor = None
        self.q_critic = None
        self.target_q_critic = None
        self.actor_optimizer = None
        self.q_critic_optimizer = None
        # Thiết lập hàm mất mát
        self.grad_clip = grad_clip
        self.q_loss = q_loss_cls()
        # Có tự điều chỉnh α hay không
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            target_entropy = target_entropy or -self.num_actions  # Entropy mục tiêu = -dim(A)
            self.target_entropy = th.tensor(target_entropy, dtype=float, requires_grad=True, device=self.device)
            self.log_alpha = th.tensor(np.log(alpha), dtype=float, requires_grad=True, device=self.device)  # log_alpha không bị giới hạn > 0
            self.alpha_optimizer = alpha_optim_cls([self.log_alpha], lr = lr_alpha)
            self.lr_alpha = lr_alpha
        # Các tham số khác
        self.learn_counter = 0
    
    # 0. Giao diện Torch
    def to(self, device: DeviceLike):
        """Cài đặt thiết bị cho thuật toán"""
        assert self.__set_nn, "Chưa thiết lập mạng nơ-ron!"
        assert self.__set_buffer, "Chưa thiết lập ReplayBuffer!"
        self.device = device
        self.buffer.to(device)
        self.actor.to(device)
        self.q_critic.to(device)
        self.target_q_critic.to(device)
        if self.adaptive_alpha:
            self.log_alpha.to(device)
            self.target_entropy.to(device)
        return self

    def cuda(self, cuda_id=None):
        """Chuyển thiết bị thuật toán sang CUDA"""
        device = 'cpu' if not th.cuda.is_available() else 'cuda' if cuda_id is None else 'cuda:' + str(cuda_id)
        self.to(device)
        return self

    def cpu(self):
        """Chuyển thiết bị thuật toán sang CPU"""
        self.to('cpu')
        return self

    
    # 1. Giao diện IO
    def save(self, data_dir: PathLike):
        """Lưu trữ thuật toán"""
        assert self.__set_nn, "Chưa thiết lập mạng nơ-ron!"
        assert self.__set_buffer, "Chưa thiết lập ReplayBuffer!"
        data_dir = Path(data_dir)
        model_dir = data_dir/'state_dict'
        model_dir.mkdir(parents=True, exist_ok=True)
        th.save(self.actor.state_dict(), model_dir/'actor.pth')
        th.save(self.q_critic.state_dict(), model_dir/'critic.pth')
        th.save(self.target_q_critic.state_dict(), model_dir/'target_critic.pth')
        self.buffer.save(data_dir/'buffer')
        # Lưu trữ các tham số như hệ số nhiệt/ tối ưu hóa/ tham số thuật toán, mã bỏ qua
        
    def load(self, data_dir: PathLike):
        """Tải thuật toán"""
        assert self.__set_nn, "Chưa thiết lập mạng nơ-ron!"
        assert self.__set_buffer, "Chưa thiết lập ReplayBuffer!"
        data_dir = Path(data_dir)
        self.actor.load_state_dict(th.load(data_dir/'state_dict'/'actor.pth', map_location=self.device))
        self.q_critic.load_state_dict(th.load(data_dir/'state_dict'/'critic.pth', map_location=self.device))
        self.target_q_critic.load_state_dict(th.load(data_dir/'state_dict'/'target_critic.pth', map_location=self.device))
        self.buffer.load(data_dir/'buffer')
        # Tải các tham số như hệ số nhiệt/ tối ưu hóa/ tham số thuật toán, mã bỏ qua

    def export(
        self, 
        file: PathLike, 
        map_device: DeviceLike = 'cpu', 
        use_stochastic_policy: bool = True, 
        output_logprob: bool = False
    ):
        """Xuất mô hình chính sách ONNX (có thể xem đồ thị tính toán mô hình qua https://netron.app)\n
        Tham số:
            file (PathLike): Tên file mô hình.
            map_device (DeviceLike): Thiết bị tính toán mô hình. Mặc định là 'cpu'.
            use_stochastic_policy (bool): Có sử dụng mô hình chính sách ngẫu nhiên không. Mặc định là True.
            output_logprob (bool): Mô hình có tính toán entropy chính sách của SAC không. Mặc định là False.
        """
        assert self.__set_nn, "Chưa thiết lập mạng nơ-ron!"
        file = Path(file).with_suffix('.onnx')
        file.parents[0].mkdir(parents=True, exist_ok=True)
        # Thiết lập đầu vào/đầu ra
        device = deepcopy(self.device)
        self.to(map_device)
        obs_tensor = self.state_to_tensor(self.obs_space.sample())
        dummy_input = (obs_tensor, not use_stochastic_policy, output_logprob)  # BUG use_rnn cần thêm h hoặc h+C
        input_names, output_names = self._get_onnx_input_output_names(False)  # BUG use_rnn chưa triển khai
        dynamic_axes, axes_name = self._get_onnx_dynamic_axes(input_names, output_names)
        if output_logprob:
            output_names += ['logprob']
            dynamic_axes['logprob'] = axes_name
        # Triển khai mô hình
        self.actor.eval()
        th.onnx.export(
            self.actor, 
            dummy_input,
            file, 
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            export_params=True,
            verbose=False,
            opset_version=11,  # Phiên bản 11 trở đi mới hỗ trợ phép toán Normal
        )
        self.actor.train()
        self.to(device)

    def _get_onnx_input_output_names(self, use_rnn=False):
        """Lấy tên đầu vào/đầu ra cho onnx"""
        if isinstance(self.obs_space, GymDict):
            input_names = [str(k) for k in self.obs_space]
        elif isinstance(self.obs_space, GymTuple):
            input_names = ['observation'+str(i) for i in range(len(self.obs_space))]
        else:
            input_names = ['observation']
        output_names = ['action']  # NOTE Không hỗ trợ không gian hành động hỗn hợp
        if use_rnn:
            input_names += ['old_hidden']  # BUG Không hỗ trợ LSTM, cần thêm cell_state
            output_names += ['new_hidden']
        return input_names, output_names

    def _get_onnx_dynamic_axes(self, onnx_input_names, onnx_output_names):
        """Lấy các trục động cho onnx"""
        # Dạng dữ liệu: (batch_size, seq_len, *shape)
        # Dạng ẩn: (num_directions*num_layer, batch_size, hidden_size)
        if 'old_hidden' in onnx_input_names:
            data_axes_name = {0: 'batch_size', 1: 'seq_len'}
        else:
            data_axes_name = {0: 'batch_size'}
        axes_dict = {}
        for k in onnx_input_names + onnx_output_names:
            if k in {'old_hidden', 'new_hidden'}:
                axes_dict[k] = {1: 'batch_size'}
            else:
                axes_dict[k] = data_axes_name
        return axes_dict, data_axes_name


    # 2. Giao diện thiết lập mạng nơ-ron
    def set_nn(
        self, 
        actor: SAC_Actor, 
        critic: SAC_Critic, 
        *, 
        actor_optim_cls = th.optim.Adam, 
        critic_optim_cls = th.optim.Adam, 
        copy: bool = True
    ):
        """Thiết lập mạng nơ-ron, yêu cầu là các đối tượng SAC_Actor/SAC_Critic, hoặc các đối tượng lớp ngỗng có cấu trúc tương tự"""
        self.__set_nn = True
        self.actor = deepcopy(actor) if copy else actor
        self.actor.train().to(self.device)
        self.q_critic = deepcopy(critic) if copy else critic
        self.q_critic.train().to(self.device)  # Twin Q Critic
        self.target_q_critic = self._build_target(self.q_critic)
        self.actor_optimizer = actor_optim_cls(self.actor.parameters(), self.lr_actor)
        self.q_critic_optimizer = critic_optim_cls(self.q_critic.parameters(), self.lr_critic)


    # 3. Giao diện thiết lập bộ đệm trải nghiệm
    def set_buffer(self, buffer: BaseBuffer):
        """Thiết lập bộ đệm trải nghiệm, yêu cầu là đối tượng của lớp con BaseBuffer, hoặc đối tượng có cấu trúc tương tự"""
        self.__set_buffer = True
        self.buffer = buffer

    def store_memory(
        self, 
        transition: tuple[Obs, Act, float, Obs, bool], 
        terminal: bool = None, 
        **kwargs
    ):
        """Lưu trữ trải nghiệm\n
        Args:
            transition (tuple): Bộ tuple (s, a, r, s_, done), thứ tự không được thay đổi.
            terminal (bool): Tham số điều khiển cho các thuật toán RNN như DRQN/R2D2, điều khiển con trỏ thời gian trong bộ đệm.
            **kwargs: Các tham số điều khiển khác cho phương thức `push` của bộ đệm.
            Lưu ý: done biểu thị kết thúc thành công/thất bại/những trường hợp tử vong, khi đó không có trạng thái tiếp theo s_; terminal biểu thị kết thúc của một vòng (parameter `truncated` trong gym mới), có thể do hết thời gian hoặc vượt quá giới hạn, lúc này sẽ có trạng thái tiếp theo s_.
        """
        assert self.__set_buffer, "Chưa thiết lập ReplayBuffer!"
        self.buffer.push(transition, terminal, **kwargs)

    def replay_memory(self, batch_size: int, **kwargs):
        """Trải nghiệm quay lại\n
        Args:
            batch_size (int): Số lượng mẫu.
            **kwargs: Các tham số điều khiển cho phương thức `sample` của bộ đệm, ví dụ như khi sử dụng trải nghiệm ưu tiên (PER), cần truyền vào tham số `rate = learn_step / total_step` để cập nhật tham số alpha/beta của bộ đệm.
        Returns:
            batch = {'s': ObsBatch, 'a': ActBatch, 'r': FloatTensor, 's_': ObsBatch, 'done': FloatTensor, ...}\n
            Nếu là từ điển PER, các khóa cũng cần bao gồm 'IS_weight'.
        """
        return self.buffer.sample(batch_size, **kwargs)

    @property
    def buffer_len(self) -> int:
        """Dung lượng hiện tại của bộ đệm"""
        return len(self.buffer)

    @property
    def use_per(self) -> bool:
        """Có sử dụng trải nghiệm ưu tiên (PER) hay không"""
        return self.buffer.is_per

        
    # 4. Giao diện mô-đun ra quyết định
    def state_to_tensor(self, state: Obs) -> ObsBatch:
        """Chuyển trạng thái lên chiều và chuyển đổi thành Tensor"""
        return self.buffer.state_to_tensor(state, use_rnn=False) # (1, *state_shape) tensor trên GPU

    def select_action(self, state: Obs, *, deterministic=False, **kwargs) -> np.ndarray:
        """Chọn hành động -> [-1, 1]"""
        assert self.__set_nn, "Chưa thiết lập mạng nơ-ron!"
        state = self.state_to_tensor(state)
        return self.actor.act(state, deterministic) # (act_dim, ) ndarray

    def random_action(self) -> np.ndarray:
        """Chọn hành động ngẫu nhiên -> [-1, 1]"""
        action = self.act_space.sample()
        lb, ub = self.act_space.low, self.act_space.high
        action = 2 * (action - lb) / (ub - lb) - 1
        return np.clip(action, -1.0, 1.0) # (act_dim, ) ndarray


    # 5. Giao diện học Reinforcement Learning
    def learn(self, **kwargs) -> dict[str, Union[float, None]]:
        """Soft Actor-Critic\n
        1. Tối ưu hóa Critic
            min J(Q) = LOSS[ Q(s, a) - Q* ]\n
            Q* = r + (1-d) * γ * V(s_, a*)\n
            V(s_, a*) = Qt(s_, a*) - α*log π(a*|s_)\n
        2. Tối ưu hóa Actor
            min J(π) = -V(s, a^)\n
            V(s, a^) = α*log π(a^|s) - Q(s, a^)\n
        3. Tối ưu hóa Alpha
            min J(α) = -α * (log π(a^|s) + H0)\n
            min J(α) = -logα * (log π(a^|s) + H0) -> nhanh hơn\n
        """
        assert self.__set_nn, "Chưa thiết lập mạng nơ-ron!"
        if self.buffer_len < self.batch_size or self.buffer_len < self.update_after:    
            return {'q_loss': None, 'actor_loss': None, 'alpha_loss': None, 'q': None, 'alpha': None}
        self.learn_counter += 1
        
        ''' Trải nghiệm hồi tiếp '''
        batch = self.replay_memory(self.batch_size, **kwargs)  # trả về tensor trên GPU

        ''' Cập nhật Critic '''
        #* J(Q) = E_{s_t~D, a_t~D, s_t+1~D, a_t+1~π_t+1}[0.5*[ Q(s_t, a_t) - [r + (1-d)*γ* [ Q_tag(s_t+1,a_t+1) - α*logπ_t+1 ] ]^2 ]
        q_loss, Q_curr = self._compute_qloss(batch)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.q_critic.parameters(), self.grad_clip)
        self.q_critic_optimizer.step()

        ''' Cập nhật Actor '''
        #* J(π) = E_{s_t~D, a~π_t}[ α*logπ_t(a|π_t) - Q(s_t, a) ] 
        self._freeze_network(self.q_critic)
        a_loss, log_pi = self._compute_ploss(batch)
        self._optim_step(self.actor_optimizer, a_loss)
        self._unfreeze_network(self.q_critic)

        ''' Cập nhật Alpha '''
        #* J(α) = E_{a~π_t}[ -α * ( logπ_t(a|π_t) + H0 ) ]
        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()  # Thuận lợi hơn, tính toán nhanh hơn
            #alpha_loss = -(self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)).mean()  # Dùng cho tập episode lớn hơn, nhưng tính toán chậm hơn
            self._optim_step(self.alpha_optimizer, alpha_loss)
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item()  # dùng cho logging
        else:
            alpha_loss = None

        ''' Cập nhật target '''
        self._soft_update(self.target_q_critic, self.q_critic, self.tau)
        
        ''' Suy giảm học tốc độ '''
        if self.lr_decay_period:
            self._lr_decay(self.actor_optimizer)
            self._lr_decay(self.q_critic_optimizer)
            if self.adaptive_alpha:
                self._lr_decay(self.alpha_optimizer)
        
        return {'q_loss': q_loss.item(), 'actor_loss': a_loss.item(), 'alpha_loss': alpha_loss, 
                'q': Q_curr.mean().item(), 'alpha': self.alpha}

    
    # 6. Hàm mất mát SAC
    def _compute_qloss(self, batch) -> tuple[th.Tensor, th.Tensor]:
        """Tính toán hàm mất mát Q-Critic (liên tục) hoặc Q-Net (rời rạc), trả về Loss và Q giá trị hiện tại"""
        s, a, r, s_, done = batch["s"], batch["a"], batch["r"], batch["s_"], batch["done"]
        #* SAC: Q_targ = E[ r + (1-d) * γ * V_next ]
        #* SAC: V_next = E[ Q_next - α*logπ_next ]
        with th.no_grad():
            a_, log_pi_next = self.actor(s_)                # (m, act_dim), (m, 1) GPU không tính gradient
            Q1_next, Q2_next = self.target_q_critic(s_, a_) # (m, 1)
            Q_next = th.min(Q1_next, Q2_next)               
            Q_targ = r + (1.0 - done) * self.gamma * (Q_next - self.alpha*log_pi_next) # (m, 1)
        Q1_curr, Q2_curr = self.q_critic(s, a)              # (m, 1) GPU tính gradient

        if self.use_per:
            IS_weight = batch["IS_weight"]
            td_err1, td_err2 = Q1_curr - Q_targ, Q2_curr - Q_targ
            q_loss = (IS_weight * (td_err1 ** 2)).mean() + (IS_weight * (td_err2 ** 2)).mean() # () Chú ý: mean phải được thêm vào ngoài cùng!!!!！
            self.buffer.update_priorities(td_err1.detach().cpu().numpy().flatten()) # Cập nhật độ ưu tiên của buffer với lỗi td err: (m, ) ndarray
        else:
            q_loss = self.q_loss(Q1_curr, Q_targ) + self.q_loss(Q2_curr, Q_targ) # ()

        return q_loss, Q1_curr

    def _compute_ploss(self, batch) -> tuple[th.Tensor, th.Tensor]:
        """Tính toán hàm mất mát của Actor và logπ, trả về Loss và logπ"""
        state = batch["s"]
        new_action, log_pi = self.actor(state)    # (m, act_dim), (m, 1) GPU tính gradient
        Q1, Q2 = self.q_critic(state, new_action) # (m, 1) GPU không tính gradient 
        Q = th.min(Q1, Q2)                        # (m, 1) GPU không tính gradient
        a_loss = (self.alpha * log_pi - Q).mean()
        return a_loss, log_pi


    # 7. Các hàm chức năng
    @staticmethod
    def _soft_update(target_network: nn.Module, network: nn.Module, tau: float):
        """
        Cập nhật mềm mạng mục tiêu\n
        >>> for target_param, param in zip(target_network.parameters(), network.parameters()):
        >>>    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        """
        for target_param, param in zip(target_network.parameters(), network.parameters()):
            target_param.data.copy_( target_param.data * (1.0 - tau) + param.data * tau ) # Cập nhật mềm

    @staticmethod
    def _hard_update(target_network: nn.Module, network: nn.Module):
        """
        Cập nhật cứng mạng mục tiêu\n
        >>> target_network.load_state_dict(network.state_dict())
        """
        # for target_param, param in zip(target_network.parameters(), network.parameters()):
        #     target_param.data.copy_(param.data)
        target_network.load_state_dict(network.state_dict()) # Cập nhật cứng

    @staticmethod
    def _freeze_network(network: nn.Module):
        """
        Đóng băng mạng nơ-ron\n
        >>> for p in network.parameters():
        >>>     p.requires_grad = False
        """
        for p in network.parameters():
            p.requires_grad = False
        # requires_grad = False dùng để chuỗi mạng (cần truyền gradient lại) tính toán mất mát, như khi truyền gradient từ Q tới mạng Actor nhưng không cập nhật Critic
        # with th.no_grad() dùng cho mạng song song hoặc chuỗi mạng (không cần truyền gradient lại) tính toán mất mát, như khi Actor tính toán next_a, dùng next_a để tính Q nhưng không truyền gradient từ Q về Actor

    @staticmethod
    def _unfreeze_network(network: nn.Module):
        """
        Mở khóa mạng nơ-ron\n
        >>> for p in network.parameters():
        >>>     p.requires_grad = True
        """
        for p in network.parameters():
            p.requires_grad = True

    @staticmethod
    def _build_target(network: nn.Module):
        """
        Sao chép một mạng mục tiêu\n
        >>> target_network = deepcopy(network).eval()
        >>> for p in target_network.parameters():
        >>>     p.requires_grad = False
        """
        target_network = deepcopy(network).eval()
        for p in target_network.parameters():
            p.requires_grad = False
        return target_network

    @staticmethod
    def _set_lr(optimizer: th.optim.Optimizer, lr: float):
        """
        Điều chỉnh học suất của bộ tối ưu hóa\n
        >>> for g in optimizer.param_groups:
        >>>     g['lr'] = lr
        """
        for g in optimizer.param_groups:
            g['lr'] = lr

    def _lr_decay(self, optimizer: th.optim.Optimizer):
        """Giảm học suất (trong chu kỳ lr_decay_period, giảm xuống 0.1 lần học suất ban đầu, period là None/0 không giảm)
        >>> lr = 0.9 * lr_init * max(0, 1 - step / lr_decay_period) + 0.1 * lr_init
        >>> self._set_lr(optimizer, lr)
        """
        if self.lr_decay_period:
            lr_init = optimizer.defaults["lr"] # Đọc học suất khởi tạo của bộ tối ưu hóa
            lr = 0.9 * lr_init * max(0, 1 - self.learn_counter / self.lr_decay_period) + 0.1 * lr_init # Cập nhật lr
            self._set_lr(optimizer, lr) # Thay đổi học suất trong param_groups
            # LƯU Ý: Việc thay đổi học suất trong param_groups sẽ không thay đổi lr trong defaults

    @staticmethod
    def _optim_step(optimizer: th.optim.Optimizer, loss: th.Tensor):
        """
        Cập nhật trọng số mạng nơ-ron\n
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


