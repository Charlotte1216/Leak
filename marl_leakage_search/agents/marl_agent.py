"""
marl_agent.py
单个智能体的行为封装，包括动作选择、策略网络、训练和经验回放
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional

from .networks import (
    ActorCriticNetwork,
    LSTMActorCriticNetwork,
    TransformerActorCriticNetwork,
    DQNNetwork,
    LSTMDQNNetwork
)


class ReplayBuffer:
    """
    经验回放缓冲区
    用于存储和重用过去的经验
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    @staticmethod
    def _to_plain(value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return value.numpy()
        return value
    
    def push(self, state, action, reward, next_state, done):
        """存储一个经验元组"""
        self.buffer.append(
            (
                self._to_plain(state),
                int(self._to_plain(action)),
                float(self._to_plain(reward)),
                self._to_plain(next_state),
                bool(self._to_plain(done)),
            )
        )
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class MARLAgent:
    """
    多智能体强化学习中的单个智能体
    支持 PPO 和 DQN 算法
    """
    def __init__(
        self,
        agent_id: int,
        state_dim: int,
        action_dim: int = 8,  # 8个方向：上下左右及四个对角线
        algorithm: str = 'ppo',  # 'ppo' 或 'dqn'
        network_type: str = 'ffnn',  # 'ffnn', 'lstm', 'transformer'
        config: Optional[Dict] = None
    ):
        """
        初始化智能体
        
        Args:
            agent_id: 智能体ID
            state_dim: 状态维度（例如：位置 + 浓度信息）
            action_dim: 动作维度（默认8个方向）
            algorithm: 使用的算法（'ppo' 或 'dqn'）
            network_type: 网络类型（'ffnn', 'lstm', 'transformer'）
            config: 配置字典
        """
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.algorithm = algorithm
        self.network_type = network_type
        
        # 从配置中获取参数
        config = config or {}
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.epsilon = config.get('epsilon', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.batch_size = config.get('batch_size', 64)
        self.tau = config.get('tau', 0.005)  # 软更新参数
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # PPO 特定参数
        self.ppo_clip = config.get('ppo_clip', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # 创建网络
        self._build_networks()
        
        # 优化器
        if self.algorithm == 'ppo':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        else:  # dqn
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # 经验回放缓冲区（主要用于 DQN）
        self.replay_buffer = ReplayBuffer(capacity=config.get('replay_buffer_size', 10000))
        
        # 训练统计
        self.training_stats = {
            'loss': [],
            'value_loss': [],
            'policy_loss': []
        }
    
    def _build_networks(self):
        """构建神经网络"""
        hidden_dims = [128, 128]
        
        if self.algorithm == 'ppo':
            if self.network_type == 'ffnn':
                self.policy_net = ActorCriticNetwork(
                    self.state_dim, self.action_dim, hidden_dims
                ).to(self.device)
            elif self.network_type == 'lstm':
                self.policy_net = LSTMActorCriticNetwork(
                    self.state_dim, self.action_dim
                ).to(self.device)
            elif self.network_type == 'transformer':
                self.policy_net = TransformerActorCriticNetwork(
                    self.state_dim, self.action_dim
                ).to(self.device)
            else:
                raise ValueError(f"Unknown network type: {self.network_type}")
        
        elif self.algorithm == 'dqn':
            if self.network_type == 'ffnn':
                self.q_net = DQNNetwork(
                    self.state_dim, self.action_dim, hidden_dims
                ).to(self.device)
                self.target_q_net = DQNNetwork(
                    self.state_dim, self.action_dim, hidden_dims
                ).to(self.device)
            elif self.network_type == 'lstm':
                self.q_net = LSTMDQNNetwork(
                    self.state_dim, self.action_dim
                ).to(self.device)
                self.target_q_net = LSTMDQNNetwork(
                    self.state_dim, self.action_dim
                ).to(self.device)
            else:
                raise ValueError(f"DQN does not support {self.network_type} network type")
            
            # 初始化目标网络
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            self.target_q_net.eval()
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态（位置、浓度信息等）
            training: 是否处于训练模式
        
        Returns:
            action: 选择的动作（0-7，对应8个方向）
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        if self.algorithm == 'ppo':
            # PPO: 使用策略网络采样动作
            with torch.no_grad():
                if self.network_type == 'lstm':
                    action_probs, _, _ = self.policy_net(state_tensor)
                else:
                    action_probs, _ = self.policy_net(state_tensor)
                
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                return action.item()
        
        else:  # dqn
            # DQN: epsilon-greedy 策略
            if training and random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)
            
            with torch.no_grad():
                if self.network_type == 'lstm':
                    q_values, _ = self.q_net(state_tensor)
                else:
                    q_values = self.q_net(state_tensor)
                
                action = q_values.argmax().item()
                return action
    
    def select_action_with_probs(self, state: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        选择动作并返回概率和值（用于 PPO）
        
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        if self.algorithm != 'ppo':
            raise ValueError("This method is only for PPO algorithm")
        
        with torch.no_grad():
            if self.network_type == 'lstm':
                action, log_prob, entropy, value, _ = self.policy_net.get_action_and_value(state_tensor)
            else:
                action, log_prob, entropy, value = self.policy_net.get_action_and_value(state_tensor)
            
            return action.item(), log_prob, value
    
    def train(self, batch: Optional[Dict] = None):
        """
        训练智能体
        
        Args:
            batch: 训练批次数据（用于 PPO）
                对于 PPO: {'states', 'actions', 'old_log_probs', 'rewards', 'values', 'advantages', 'returns'}
                对于 DQN: 从经验回放缓冲区采样
        """
        if self.algorithm == 'ppo':
            self._train_ppo(batch)
        else:  # dqn
            self._train_dqn()
    
    def _train_ppo(self, batch: Dict):
        """使用 PPO 算法训练"""
        if batch is None or len(batch['states']) == 0:
            return
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        
        total_loss = 0
        
        for epoch in range(self.ppo_epochs):
            # 前向传播
            if self.network_type == 'lstm':
                # PPO 按样本批训练时不复用 rollout 的隐藏状态，避免 batch 维度不匹配
                self.policy_net.hidden_state = None
                _, new_log_probs, entropy, values, _ = self.policy_net.get_action_and_value(
                    states, actions, hidden=None
                )
            else:
                _, new_log_probs, entropy, values = self.policy_net.get_action_and_value(
                    states, actions
                )
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO 裁剪
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # 熵损失
            entropy_loss = -entropy.mean()
            
            # 总损失
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # 更新统计
        self.training_stats['loss'].append(total_loss / self.ppo_epochs)
    
    def _train_dqn(self):
        """使用 DQN 算法训练"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 当前 Q 值
        if self.network_type == 'lstm':
            self.q_net.hidden_state = None
            current_q_values, _ = self.q_net(states, hidden=None)
        else:
            current_q_values = self.q_net(states)
        
        q_value = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标 Q 值
        with torch.no_grad():
            if self.network_type == 'lstm':
                self.target_q_net.hidden_state = None
                next_q_values, _ = self.target_q_net(next_states, hidden=None)
            else:
                next_q_values = self.target_q_net(next_states)
            
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        # 计算损失
        loss = F.mse_loss(q_value, target_q_value)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 软更新目标网络
        self._soft_update_target_network()
        
        # 更新 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # 更新统计
        self.training_stats['loss'].append(loss.item())
    
    def _soft_update_target_network(self):
        """软更新目标网络（用于 DQN）"""
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update_replay_buffer(self, state, action, reward, next_state, done):
        """更新经验回放缓冲区（用于 DQN）"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def save(self, filepath: str):
        """保存模型"""
        if self.algorithm == 'ppo':
            torch.save(self.policy_net.state_dict(), filepath)
        else:
            torch.save({
                'q_net': self.q_net.state_dict(),
                'target_q_net': self.target_q_net.state_dict(),
            }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        if self.algorithm == 'ppo':
            self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        else:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_net.load_state_dict(checkpoint['q_net'])
            self.target_q_net.load_state_dict(checkpoint['target_q_net'])
        
        if self.algorithm == 'ppo':
            self.policy_net.eval()
        else:
            self.q_net.eval()
            self.target_q_net.eval()
    
    def reset(self):
        """重置智能体状态（例如 LSTM 的隐藏状态）"""
        if self.network_type == 'lstm':
            if self.algorithm == 'ppo':
                self.policy_net.reset_hidden()
            else:
                self.q_net.reset_hidden()
                self.target_q_net.reset_hidden()
        elif self.network_type == 'transformer' and self.algorithm == 'ppo':
            self.policy_net.reset_buffer()
    
    def eval(self):
        """设置为评估模式"""
        if self.algorithm == 'ppo':
            self.policy_net.eval()
        else:
            self.q_net.eval()
            self.target_q_net.eval()
    
    def train_mode(self):
        """设置为训练模式"""
        if self.algorithm == 'ppo':
            self.policy_net.train()
        else:
            self.q_net.train()
            self.target_q_net.eval()  # 目标网络始终保持评估模式
