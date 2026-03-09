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
        self.aux_enabled = bool(config.get('aux_enabled', False))
        self.aux_weight = float(config.get('aux_weight', 0.0))
        self.aux_target_index = int(config.get('aux_target_index', 2))
        self.aux_hidden_dim = int(config.get('aux_hidden_dim', 128))
        self.lstm_hidden_dim = int(config.get('lstm_hidden_dim', 128))
        self.lstm_lstm_hidden_dim = int(config.get('lstm_lstm_hidden_dim', 64))
        self.lstm_num_layers = int(config.get('lstm_num_layers', 1))
        self.ppo_lstm_seq_training_enabled = bool(config.get('ppo_lstm_seq_training_enabled', False))
        self.ppo_lstm_seq_len = int(config.get('ppo_lstm_seq_len', 16))
        self.ppo_lstm_seq_stride = int(config.get('ppo_lstm_seq_stride', 4))
        
        # 创建网络
        self._build_networks()
        
        # 优化器
        if self.algorithm == 'ppo':
            params = list(self.policy_net.parameters())
            if self.aux_enabled:
                # Action-conditioned auxiliary predictor head: predicts next-step concentration.
                self.aux_predictor = nn.Sequential(
                    nn.Linear(self.state_dim + self.action_dim, self.aux_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.aux_hidden_dim, 1),
                ).to(self.device)
                params += list(self.aux_predictor.parameters())
            else:
                self.aux_predictor = None
            self.optimizer = optim.Adam(params, lr=self.lr)
        else:  # dqn
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
            self.aux_predictor = None
        
        # 经验回放缓冲区（主要用于 DQN）
        self.replay_buffer = ReplayBuffer(capacity=config.get('replay_buffer_size', 10000))
        
        # 训练统计
        self.training_stats = {
            'loss': [],
            'value_loss': [],
            'policy_loss': [],
            'aux_loss': [],
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
                    self.state_dim,
                    self.action_dim,
                    hidden_dim=self.lstm_hidden_dim,
                    lstm_hidden_dim=self.lstm_lstm_hidden_dim,
                    num_layers=self.lstm_num_layers,
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
                    self.state_dim,
                    self.action_dim,
                    hidden_dim=self.lstm_hidden_dim,
                    lstm_hidden_dim=self.lstm_lstm_hidden_dim,
                    num_layers=self.lstm_num_layers,
                ).to(self.device)
                self.target_q_net = LSTMDQNNetwork(
                    self.state_dim,
                    self.action_dim,
                    hidden_dim=self.lstm_hidden_dim,
                    lstm_hidden_dim=self.lstm_lstm_hidden_dim,
                    num_layers=self.lstm_num_layers,
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

    def _build_lstm_sequence_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        next_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert flat trajectory [T, D] into sequence batch [B, L, D] for LSTM PPO updates.
        Loss is computed on the last timestep of each sequence.
        """
        total_steps = int(states.shape[0])
        seq_len = max(1, min(int(self.ppo_lstm_seq_len), total_steps))
        stride = max(1, int(self.ppo_lstm_seq_stride))

        last_indices = list(range(seq_len - 1, total_steps, stride))
        if not last_indices or last_indices[-1] != total_steps - 1:
            last_indices.append(total_steps - 1)

        seq_states = []
        seq_actions = []
        seq_old_log_probs = []
        seq_advantages = []
        seq_returns = []
        seq_last_states = []
        seq_next_states = [] if next_states is not None else None

        for last_idx in last_indices:
            start_idx = max(0, last_idx - seq_len + 1)
            seq = states[start_idx:last_idx + 1]
            # Left-pad short prefix windows to fixed length so batching stays dense.
            if seq.shape[0] < seq_len:
                pad = seq[0].unsqueeze(0).repeat(seq_len - seq.shape[0], 1)
                seq = torch.cat([pad, seq], dim=0)

            seq_states.append(seq)
            seq_actions.append(actions[last_idx])
            seq_old_log_probs.append(old_log_probs[last_idx])
            seq_advantages.append(advantages[last_idx])
            seq_returns.append(returns[last_idx])
            seq_last_states.append(states[last_idx])
            if seq_next_states is not None:
                seq_next_states.append(next_states[last_idx])

        batch_states = torch.stack(seq_states, dim=0)
        batch_actions = torch.stack(seq_actions, dim=0)
        batch_old_log_probs = torch.stack(seq_old_log_probs, dim=0)
        batch_advantages = torch.stack(seq_advantages, dim=0)
        batch_returns = torch.stack(seq_returns, dim=0)
        batch_last_states = torch.stack(seq_last_states, dim=0)
        batch_next_states = torch.stack(seq_next_states, dim=0) if seq_next_states is not None else None

        return (
            batch_states,
            batch_actions,
            batch_old_log_probs,
            batch_advantages,
            batch_returns,
            batch_last_states,
            batch_next_states,
        )
    
    def _train_ppo(self, batch: Dict):
        """使用 PPO 算法训练"""
        if batch is None or len(batch['states']) == 0:
            return
        
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.LongTensor(batch['actions']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.device)
        returns = torch.FloatTensor(batch['returns']).to(self.device)
        next_states = None
        if 'next_states' in batch and batch['next_states'] is not None and len(batch['next_states']) > 0:
            next_states = torch.FloatTensor(batch['next_states']).to(self.device)

        train_states = states
        train_actions = actions
        train_old_log_probs = old_log_probs
        train_advantages = advantages
        train_returns = returns
        aux_states = states
        aux_next_states = next_states

        if self.network_type == 'lstm' and self.ppo_lstm_seq_training_enabled:
            (
                train_states,
                train_actions,
                train_old_log_probs,
                train_advantages,
                train_returns,
                aux_states,
                aux_next_states,
            ) = self._build_lstm_sequence_batch(
                states=states,
                actions=actions,
                old_log_probs=old_log_probs,
                advantages=advantages,
                returns=returns,
                next_states=next_states,
            )
        
        total_loss = 0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_aux_loss = 0.0
        
        for epoch in range(self.ppo_epochs):
            # 前向传播
            if self.network_type == 'lstm':
                # PPO 按样本批训练时不复用 rollout 的隐藏状态，避免 batch 维度不匹配
                self.policy_net.hidden_state = None
                _, new_log_probs, entropy, values, _ = self.policy_net.get_action_and_value(
                    train_states, train_actions, hidden=None
                )
            else:
                _, new_log_probs, entropy, values = self.policy_net.get_action_and_value(
                    train_states, train_actions
                )
            
            # 计算比率
            ratio = torch.exp(new_log_probs - train_old_log_probs)
            
            # PPO 裁剪
            surr1 = ratio * train_advantages
            surr2 = torch.clamp(ratio, 1 - self.ppo_clip, 1 + self.ppo_clip) * train_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(values.view(-1), train_returns.view(-1))
            
            # 熵损失
            entropy_loss = -entropy.mean()
            
            # 总损失
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            aux_loss_value = 0.0
            if (
                self.aux_enabled
                and self.aux_predictor is not None
                and aux_next_states is not None
                and 0 <= self.aux_target_index < aux_next_states.shape[1]
            ):
                action_onehot = F.one_hot(train_actions, num_classes=self.action_dim).float()
                aux_input = torch.cat([aux_states, action_onehot], dim=1)
                aux_pred = self.aux_predictor(aux_input).squeeze(-1)
                aux_target = aux_next_states[:, self.aux_target_index]
                aux_loss = F.mse_loss(aux_pred, aux_target)
                loss = loss + self.aux_weight * aux_loss
                aux_loss_value = float(aux_loss.item())
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            if self.aux_predictor is not None:
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.aux_predictor.parameters()),
                    0.5,
                )
            else:
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_aux_loss += aux_loss_value
        
        # 更新统计
        self.training_stats['loss'].append(total_loss / self.ppo_epochs)
        self.training_stats['policy_loss'].append(total_policy_loss / self.ppo_epochs)
        self.training_stats['value_loss'].append(total_value_loss / self.ppo_epochs)
        self.training_stats['aux_loss'].append(total_aux_loss / self.ppo_epochs)
    
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
            payload = {
                'policy_net': self.policy_net.state_dict(),
                'aux_enabled': bool(self.aux_enabled),
            }
            if self.aux_predictor is not None:
                payload['aux_predictor'] = self.aux_predictor.state_dict()
            torch.save(payload, filepath)
        else:
            torch.save({
                'q_net': self.q_net.state_dict(),
                'target_q_net': self.target_q_net.state_dict(),
            }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        if self.algorithm == 'ppo':
            checkpoint = torch.load(filepath, map_location=self.device)
            if isinstance(checkpoint, dict) and 'policy_net' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                if (
                    self.aux_predictor is not None
                    and 'aux_predictor' in checkpoint
                    and isinstance(checkpoint['aux_predictor'], dict)
                ):
                    self.aux_predictor.load_state_dict(checkpoint['aux_predictor'])
            else:
                # Backward compatibility with older PPO checkpoints (policy state_dict only).
                self.policy_net.load_state_dict(checkpoint)
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
