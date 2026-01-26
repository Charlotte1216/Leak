"""
networks.py
定义智能体使用的神经网络模型，包括 Actor-Critic 网络、LSTM、Transformer 等网络结构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic 网络（全连接版本）
    用于 PPO 等策略梯度算法
    """
    def __init__(self, input_dim, action_dim, hidden_dims=[128, 128], activation='relu'):
        super(ActorCriticNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # 选择激活函数
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            act_fn = nn.ReLU
        
        # 共享特征提取层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor 网络（输出动作概率分布）
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        # Critic 网络（输出状态价值）
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            act_fn(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入状态 [batch_size, input_dim] 或 [input_dim]
        Returns:
            action_probs: 动作概率分布 [batch_size, action_dim]
            value: 状态价值 [batch_size, 1]
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        features = self.shared_layers(x)
        action_logits = self.actor(features)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action_and_value(self, x, action=None):
        """
        获取动作和值，用于 PPO 训练
        """
        action_probs, value = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, action_log_prob, entropy, value


class LSTMActorCriticNetwork(nn.Module):
    """
    带 LSTM 的 Actor-Critic 网络
    用于处理时序依赖关系
    """
    def __init__(self, input_dim, action_dim, hidden_dim=128, lstm_hidden_dim=64, num_layers=1):
        super(LSTMActorCriticNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        
        # 输入特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM 层
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, num_layers, batch_first=True)
        
        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # LSTM 隐藏状态
        self.hidden_state = None
    
    def reset_hidden(self, batch_size=1):
        """重置 LSTM 隐藏状态"""
        device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to(device)
        )
    
    def forward(self, x, hidden=None):
        """
        前向传播
        Args:
            x: 输入状态 [batch_size, seq_len, input_dim] 或 [seq_len, input_dim] 或 [input_dim]
            hidden: LSTM 隐藏状态
        Returns:
            action_probs: 动作概率分布
            value: 状态价值
            new_hidden: 新的隐藏状态
        """
        # 处理输入维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, input_dim]
        elif len(x.shape) == 2:
            if x.shape[0] == self.input_dim:
                x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, input_dim]
            else:
                x = x.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        batch_size, seq_len, _ = x.shape
        
        # 特征提取
        features = self.feature_extractor(x)  # [batch_size, seq_len, hidden_dim]
        
        # LSTM
        if hidden is None:
            hidden = self.hidden_state if self.hidden_state is not None else None
        
        lstm_out, new_hidden = self.lstm(features, hidden)  # [batch_size, seq_len, lstm_hidden_dim]
        
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        # Actor 和 Critic
        action_logits = self.actor(lstm_out)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(lstm_out)
        
        return action_probs, value, new_hidden
    
    def get_action_and_value(self, x, action=None, hidden=None):
        """获取动作和值，用于 PPO 训练"""
        action_probs, value, new_hidden = self.forward(x, hidden)
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, action_log_prob, entropy, value, new_hidden


class TransformerActorCriticNetwork(nn.Module):
    """
    基于 Transformer 的 Actor-Critic 网络
    用于处理长期依赖关系
    """
    def __init__(self, input_dim, action_dim, hidden_dim=128, num_heads=4, num_layers=2, seq_len=10):
        super(TransformerActorCriticNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim))
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 历史状态缓冲区
        self.state_buffer = []
    
    def reset_buffer(self):
        """重置状态缓冲区"""
        self.state_buffer = []
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入状态 [batch_size, input_dim] 或 [input_dim]
        Returns:
            action_probs: 动作概率分布
            value: 状态价值
        """
        # 处理输入维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # [1, input_dim]
        
        batch_size = x.shape[0]
        
        # 更新状态缓冲区
        if not hasattr(self, 'state_buffer') or len(self.state_buffer) == 0:
            self.state_buffer = [x[0].detach().cpu().numpy()] * self.seq_len
        
        self.state_buffer.append(x[0].detach().cpu().numpy())
        if len(self.state_buffer) > self.seq_len:
            self.state_buffer.pop(0)
        
        # 构建序列
        seq = torch.tensor(np.array(self.state_buffer), dtype=torch.float32).to(x.device)
        seq = seq.unsqueeze(0).repeat(batch_size, 1, 1)  # [batch_size, seq_len, input_dim]
        
        # 嵌入和位置编码
        embedded = self.input_embedding(seq)  # [batch_size, seq_len, hidden_dim]
        embedded = embedded + self.pos_encoding[:, :seq.shape[1], :]
        
        # Transformer 编码
        transformer_out = self.transformer(embedded)  # [batch_size, seq_len, hidden_dim]
        
        # 取最后一个时间步的输出
        last_hidden = transformer_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Actor 和 Critic
        action_logits = self.actor(last_hidden)
        action_probs = F.softmax(action_logits, dim=-1)
        value = self.critic(last_hidden)
        
        return action_probs, value
    
    def get_action_and_value(self, x, action=None):
        """获取动作和值，用于 PPO 训练"""
        action_probs, value = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, action_log_prob, entropy, value


class DQNNetwork(nn.Module):
    """
    DQN 网络（用于 Q-learning）
    输出每个动作的 Q 值
    """
    def __init__(self, input_dim, action_dim, hidden_dims=[128, 128], activation='relu'):
        super(DQNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # 选择激活函数
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        else:
            act_fn = nn.ReLU
        
        # 构建网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入状态 [batch_size, input_dim] 或 [input_dim]
        Returns:
            q_values: 每个动作的 Q 值 [batch_size, action_dim]
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        q_values = self.network(x)
        return q_values


class LSTMDQNNetwork(nn.Module):
    """
    带 LSTM 的 DQN 网络
    """
    def __init__(self, input_dim, action_dim, hidden_dim=128, lstm_hidden_dim=64, num_layers=1):
        super(LSTMDQNNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        
        # 输入特征提取
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM 层
        self.lstm = nn.LSTM(hidden_dim, lstm_hidden_dim, num_layers, batch_first=True)
        
        # Q 值网络
        self.q_network = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # LSTM 隐藏状态
        self.hidden_state = None
    
    def reset_hidden(self, batch_size=1):
        """重置 LSTM 隐藏状态"""
        device = next(self.parameters()).device
        self.hidden_state = (
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to(device)
        )
    
    def forward(self, x, hidden=None):
        """
        前向传播
        Args:
            x: 输入状态
            hidden: LSTM 隐藏状态
        Returns:
            q_values: 每个动作的 Q 值
            new_hidden: 新的隐藏状态
        """
        # 处理输入维度
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            if x.shape[0] == self.input_dim:
                x = x.unsqueeze(0).unsqueeze(0)
            else:
                x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # 特征提取
        features = self.feature_extractor(x)
        
        # LSTM
        if hidden is None:
            hidden = self.hidden_state if self.hidden_state is not None else None
        
        lstm_out, new_hidden = self.lstm(features, hidden)
        lstm_out = lstm_out[:, -1, :]
        
        # Q 值
        q_values = self.q_network(lstm_out)
        
        return q_values, new_hidden
