""" 
marl_trainer.py
多智能体强化学习训练器，实现 MAPPO 和 QMIX 训练逻辑
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from .marl_agent import MARLAgent


class QMIXMixer(nn.Module):
    """
    QMIX 混合网络
    将多个智能体的 Q 值合并成全局 Q 值
    """
    def __init__(self, num_agents, state_dim, hidden_dim=64):
        super(QMIXMixer, self).__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # 超网络：根据全局状态生成混合网络的权重
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 偏置
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, q_values: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        混合多个智能体的 Q 值
        
        Args:
            q_values: [batch_size, num_agents] 每个智能体的 Q 值
            global_state: [batch_size, state_dim] 全局状态
        
        Returns:
            q_total: [batch_size, 1] 全局 Q 值
        """
        batch_size = q_values.shape[0]
        
        # 第一层: [1, num_agents] x [num_agents, hidden_dim] -> [1, hidden_dim]
        w1 = torch.abs(self.hyper_w1(global_state))  # [batch_size, num_agents * hidden_dim]
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)  # [batch_size, num_agents, hidden_dim]
        b1 = self.hyper_b1(global_state)  # [batch_size, hidden_dim]

        q_values = q_values.unsqueeze(1)  # [batch_size, 1, num_agents]
        hidden = F.elu(torch.bmm(q_values, w1).squeeze(1) + b1)  # [batch_size, hidden_dim]

        # 第二层: [1, hidden_dim] x [hidden_dim, 1] -> [1, 1]
        w2 = torch.abs(self.hyper_w2(global_state)).unsqueeze(2)  # [batch_size, hidden_dim, 1]
        b2 = self.hyper_b2(global_state)  # [batch_size, 1]

        q_total = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2  # [batch_size, 1]
        
        return q_total


class MARLTrainer:
    """
    多智能体强化学习训练器
    支持 MAPPO 和 QMIX 算法
    """
    def __init__(
        self,
        agents: List[MARLAgent],
        env,
        algorithm: str = 'mappo',  # 'mappo' 或 'qmix'
        config: Optional[Dict] = None
    ):
        """
        初始化训练器
        
        Args:
            agents: 智能体列表
            env: 环境对象
            algorithm: 使用的算法（'mappo' 或 'qmix'）
            config: 配置字典
        """
        self.agents = agents
        self.env = env
        self.algorithm = algorithm
        self.num_agents = len(agents)
        
        # 从配置中获取参数
        config = config or {}
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # QMIX 特定组件
        if self.algorithm == 'qmix':
            # 检查所有智能体是否使用 DQN
            if not all(agent.algorithm == 'dqn' for agent in agents):
                raise ValueError("QMIX requires all agents to use DQN algorithm")
            
            # 创建混合网络
            global_state_dim = config.get('global_state_dim', agents[0].state_dim)
            self.mixer = QMIXMixer(self.num_agents, global_state_dim).to(self.device)
            self.target_mixer = QMIXMixer(self.num_agents, global_state_dim).to(self.device)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            
            # 混合网络优化器
            self.mixer_optimizer = torch.optim.Adam(self.mixer.parameters(), lr=float(config.get('mixer_lr', 3e-4)))
            self.tau = config.get('tau', 0.005)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'found_sources': []
        }

    def _estimate_bootstrap_value(
        self,
        agent: MARLAgent,
        states: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> float:
        """
        Estimate the bootstrap value for the final transition when a trajectory
        is truncated by the rollout horizon instead of a terminal environment state.
        """
        if len(dones) == 0 or bool(dones[-1]):
            return 0.0
        if next_states is None or len(next_states) == 0:
            return 0.0

        bootstrap_state = np.asarray(next_states[-1], dtype=np.float32)
        history_states = None
        if agent.network_type == 'lstm':
            state_history = np.asarray(states, dtype=np.float32)
            window_size = max(1, int(getattr(agent, 'ppo_lstm_seq_len', 16)))
            seq_parts = []
            if window_size > 1 and state_history.size > 0:
                seq_parts.append(state_history[-(window_size - 1):])
            seq_parts.append(bootstrap_state.reshape(1, -1))
            history_states = np.concatenate(seq_parts, axis=0)

        return agent.estimate_value(bootstrap_state, history_states=history_states)
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算广义优势估计 (GAE)
        
        Returns:
            advantages: 优势值
            returns: 回报值
        """
        advantages = np.zeros(len(rewards))
        returns = np.zeros(len(rewards))
        
        gae = 0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:

                next_val = values[step + 1] if step + 1 < len(values) else next_value
                delta = rewards[step] + self.gamma * next_val - values[step]
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def train_step_mappo(self, trajectories: List[Dict]) -> Dict:
        """
        执行一个 MAPPO 训练步骤
        
        Args:
            trajectories: 轨迹列表，每个元素包含：
                - 'states': 状态序列
                - 'actions': 动作序列
                - 'rewards': 奖励序列
                - 'dones': 完成标志序列
                - 'old_log_probs': 旧的对数概率
                - 'values': 价值估计
        
        Returns:
            训练统计信息
        """
        all_losses = []
        
        # 为每个智能体准备训练数据
        for agent_idx, agent in enumerate(self.agents):
            traj = trajectories[agent_idx]
            
            states = np.asarray(traj['states'], dtype=np.float32)
            next_states = np.asarray(traj.get('next_states', []), dtype=np.float32)
            actions = np.asarray(traj['actions'])
            rewards = np.asarray(traj['rewards'], dtype=np.float32)
            dones = np.asarray(traj['dones'], dtype=bool)
            old_log_probs = np.asarray(traj['old_log_probs'], dtype=np.float32)
            values = np.asarray(traj['values'], dtype=np.float32)
            
            # 对被 rollout 长度截断的轨迹进行真实 bootstrap，而不是默认置 0。
            next_value = self._estimate_bootstrap_value(agent, states, next_states, dones)
            
            # 计算 GAE
            values_list = list(values) + [next_value]
            advantages, returns = self.compute_gae(rewards, values_list[:-1], dones, next_value)
            
            # 准备批次数据
            batch = {
                'states': states,
                'next_states': next_states,
                'actions': actions,
                'old_log_probs': old_log_probs,
                'rewards': rewards,
                'values': values,
                'advantages': advantages,
                'returns': returns
            }
            
            # 训练智能体
            agent.train(batch)
            
            # 记录损失
            if len(agent.training_stats['loss']) > 0:
                all_losses.append(agent.training_stats['loss'][-1])
        
        return {
            'loss': np.mean(all_losses) if all_losses else 0.0,
            'num_agents': self.num_agents
        }
    
    def train_step_qmix(self, batch: Dict) -> Dict:
        """
        执行一个 QMIX 训练步骤
        
        Args:
            batch: 批次数据，包含：
                - 'states': [batch_size, seq_len, state_dim] 每个智能体的状态
                - 'actions': [batch_size, seq_len, num_agents] 动作
                - 'rewards': [batch_size, seq_len] 奖励
                - 'next_states': [batch_size, seq_len, state_dim] 下一个状态
                - 'dones': [batch_size, seq_len] 完成标志
                - 'global_states': [batch_size, seq_len, global_state_dim] 全局状态
                - 'next_global_states': [batch_size, seq_len, global_state_dim] 下一个全局状态
        
        Returns:
            训练统计信息
        """
        states = batch['states']  # [batch_size, seq_len, num_agents, state_dim]
        actions = batch['actions']  # [batch_size, seq_len, num_agents]
        rewards = batch['rewards']  # [batch_size, seq_len]
        next_states = batch['next_states']  # [batch_size, seq_len, num_agents, state_dim]
        dones = batch['dones']  # [batch_size, seq_len]
        global_states = batch['global_states']  # [batch_size, seq_len, global_state_dim]
        next_global_states = batch['next_global_states']  # [batch_size, seq_len, global_state_dim]
        
        batch_size, seq_len = rewards.shape
        
        # 计算每个智能体的 Q 值
        q_values = []
        target_q_values = []
        
        for agent_idx, agent in enumerate(self.agents):
            agent_states = states[:, :, agent_idx, :]  # [batch_size, seq_len, state_dim]
            agent_actions = actions[:, :, agent_idx]  # [batch_size, seq_len]
            agent_next_states = next_states[:, :, agent_idx, :]  # [batch_size, seq_len, state_dim]
            
            # 当前 Q 值
            agent_states_flat = agent_states.reshape(-1, agent.state_dim)
            agent_actions_flat = agent_actions.reshape(-1)
            
            if agent.network_type == 'lstm':
                q_vals, _ = agent.q_net(agent_states_flat)
            else:
                q_vals = agent.q_net(agent_states_flat)
            
            # q_vals = q_vals.gather(1, agent_actions_flat.unsqueeze(1))
            q_vals = q_vals.gather(1, agent_actions_flat.unsqueeze(1).to(torch.long))
            q_values.append(q_vals.view(batch_size, seq_len, 1))
            
            # 目标 Q 值
            agent_next_states_flat = agent_next_states.reshape(-1, agent.state_dim)
            
            with torch.no_grad():
                if agent.network_type == 'lstm':
                    next_q_vals, _ = agent.target_q_net(agent_next_states_flat)
                else:
                    next_q_vals = agent.target_q_net(agent_next_states_flat)
                
                next_q_vals = next_q_vals.max(1)[0]
                target_q_values.append(next_q_vals.view(batch_size, seq_len, 1))
        
        # 合并 Q 值
        q_values = torch.cat(q_values, dim=2)  # [batch_size, seq_len, num_agents]
        target_q_values = torch.cat(target_q_values, dim=2)  # [batch_size, seq_len, num_agents]
        
        # 计算全局 Q 值
        global_states_flat = global_states.reshape(-1, global_states.shape[-1])
        q_values_flat = q_values.reshape(-1, self.num_agents)
        q_total = self.mixer(q_values_flat, global_states_flat).view(batch_size, seq_len)
        
        # 计算目标全局 Q 值
        next_global_states_flat = next_global_states.reshape(-1, next_global_states.shape[-1])
        target_q_values_flat = target_q_values.reshape(-1, self.num_agents)
        with torch.no_grad():
            target_q_total = self.target_mixer(target_q_values_flat, next_global_states_flat).view(batch_size, seq_len)
        
        # 计算目标
        if torch.is_tensor(rewards):
            rewards_tensor = rewards.to(self.device, dtype=torch.float32)
        else:
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        if torch.is_tensor(dones):
            dones_tensor = dones.to(self.device, dtype=torch.float32)
        else:
            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        targets = rewards_tensor + (1 - dones_tensor) * self.gamma * target_q_total
        
        # 计算损失
        loss = F.mse_loss(q_total, targets)
        
        # 反向传播
        self.mixer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), 10.0)
        self.mixer_optimizer.step()
        
        # 训练各个智能体
        for agent_idx, agent in enumerate(self.agents):
            agent_states = states[:, :, agent_idx, :]
            agent_actions = actions[:, :, agent_idx]
            agent_next_states = next_states[:, :, agent_idx, :]
            
            # 将数据添加到经验回放缓冲区
            for t in range(seq_len):
                for b in range(batch_size):
                    agent.update_replay_buffer(
                        agent_states[b, t].cpu().numpy(),
                        agent_actions[b, t].item(),
                        float(rewards[b, t].item()) if torch.is_tensor(rewards) else float(rewards[b, t]),
                        agent_next_states[b, t].cpu().numpy(),
                        bool(dones[b, t].item()) if torch.is_tensor(dones) else bool(dones[b, t])
                    )
            
            # 训练智能体
            agent.train()
        
        # 软更新目标网络
        self._soft_update_target_mixer()
        
        return {
            'loss': loss.item(),
            'q_total_mean': q_total.mean().item()
        }
    
    def _soft_update_target_mixer(self):
        """软更新 QMIX 目标混合网络"""
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, num_episodes: int, max_steps_per_episode: int = 1000):
        """
        训练多个智能体
        
        Args:
            num_episodes: 训练回合数
            max_steps_per_episode: 每个回合的最大步数
        """
        for episode in range(num_episodes):
            # 重置环境
            observations = self.env.reset()
            
            # 重置智能体状态
            for agent in self.agents:
                agent.reset()
                agent.train_mode()
            
            # 存储轨迹数据
            if self.algorithm == 'mappo':
                trajectories = [defaultdict(list) for _ in range(self.num_agents)]
            else:  # qmix
                batch_data = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'next_states': [],
                    'dones': [],
                    'global_states': [],
                    'next_global_states': []
                }
            
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_steps_per_episode):
                # 选择动作
                actions = []
                action_log_probs = []
                values = []
                
                for agent_idx, agent in enumerate(self.agents):
                    obs = observations[agent_idx] if isinstance(observations, list) else observations
                    
                    if self.algorithm == 'mappo':
                        action, log_prob, value = agent.select_action_with_probs(obs)
                        actions.append(action)
                        action_log_probs.append(log_prob.cpu().numpy())
                        values.append(value.cpu().numpy())
                    else:  # qmix
                        action = agent.select_action(obs, training=True)
                        actions.append(action)
                
                # 执行动作
                next_observations, rewards, dones, info = self.env.step(actions)
                
                # 存储数据
                if self.algorithm == 'mappo':
                    for agent_idx in range(self.num_agents):
                        obs = observations[agent_idx] if isinstance(observations, list) else observations
                        trajectories[agent_idx]['states'].append(obs)
                        trajectories[agent_idx]['actions'].append(actions[agent_idx])
                        trajectories[agent_idx]['rewards'].append(rewards[agent_idx])
                        trajectories[agent_idx]['dones'].append(dones[agent_idx])
                        trajectories[agent_idx]['old_log_probs'].append(action_log_probs[agent_idx])
                        trajectories[agent_idx]['values'].append(values[agent_idx])
                else:  # qmix
                    # 获取全局状态（这里需要根据实际环境实现）
                    global_state = self._get_global_state(observations)
                    next_global_state = self._get_global_state(next_observations)
                    
                    batch_data['states'].append(observations)
                    batch_data['actions'].append(actions)
                    batch_data['rewards'].append(np.mean(rewards))  # 或者使用全局奖励
                    batch_data['next_states'].append(next_observations)
                    batch_data['dones'].append(any(dones) if isinstance(dones, list) else dones)
                    batch_data['global_states'].append(global_state)
                    batch_data['next_global_states'].append(next_global_state)
                
                episode_reward += np.mean(rewards) if isinstance(rewards, list) else rewards
                episode_length += 1
                
                observations = next_observations
                
                # 检查是否结束
                if isinstance(dones, list):
                    if all(dones):
                        break
                elif dones:
                    break
            
            # 训练
            if self.algorithm == 'mappo':
                stats = self.train_step_mappo(trajectories)
            else:  # qmix
                # 将批次数据转换为张量
                batch = self._prepare_qmix_batch(batch_data)
                stats = self.train_step_qmix(batch)
            
            # 记录统计信息
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            self.training_stats['losses'].append(stats.get('loss', 0.0))
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_stats['episode_rewards'][-10:])
                avg_length = np.mean(self.training_stats['episode_lengths'][-10:])
                print(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.2f}, "
                      f"Loss: {stats.get('loss', 0.0):.4f}")
    
    def _get_global_state(self, observations):
        """
        从局部观测构建全局状态
        这里需要根据实际环境实现
        """
        if isinstance(observations, list):
            # 简单拼接所有智能体的观测
            return np.concatenate(observations)
        else:
            return observations
    
    def _prepare_qmix_batch(self, batch_data: Dict) -> Dict:
        """准备 QMIX 批次数据"""
        # 转换为张量
        batch = {}
        batch['states'] = np.array(batch_data['states'])  # [seq_len, num_agents, state_dim]
        batch['actions'] = np.array(batch_data['actions'])  # [seq_len, num_agents]
        batch['rewards'] = np.array(batch_data['rewards'])  # [seq_len]
        batch['next_states'] = np.array(batch_data['next_states'])  # [seq_len, num_agents, state_dim]
        batch['dones'] = np.array(batch_data['dones'])  # [seq_len]
        batch['global_states'] = np.array(batch_data['global_states'])  # [seq_len, global_state_dim]
        batch['next_global_states'] = np.array(batch_data['next_global_states'])  # [seq_len, global_state_dim]
        
        # 添加批次维度
        for key in batch:
            batch[key] = torch.FloatTensor(batch[key]).unsqueeze(0).to(self.device)
        
        return batch
    
    def save_models(self, save_dir: str):
        """保存所有模型"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for agent_idx, agent in enumerate(self.agents):
            agent.save(os.path.join(save_dir, f"agent_{agent_idx}.pth"))
        
        if self.algorithm == 'qmix':
            torch.save(self.mixer.state_dict(), os.path.join(save_dir, "mixer.pth"))
    
    def load_models(self, load_dir: str):
        """加载所有模型"""
        import os
        
        for agent_idx, agent in enumerate(self.agents):
            agent.load(os.path.join(load_dir, f"agent_{agent_idx}.pth"))
        
        if self.algorithm == 'qmix':
            self.mixer.load_state_dict(torch.load(os.path.join(load_dir, "mixer.pth"), map_location=self.device))
