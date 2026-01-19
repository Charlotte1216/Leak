marl_leakage_search/
│
├── envs/
│   ├── __init__.py
│   ├── plume_model.py          # 高斯烟羽模型
│   ├── vortex_model.py         # 卡门涡街扰动模型
│   ├── concentration_field.py  # plume + vortex 总浓度场
│   ├── uav_dynamics.py         # UAV 动力学 & 能耗
│   └── marl_env.py             # Gym-style MARL 环境
│
├── agents/
│   ├── __init__.py
│   ├── networks.py             # Actor / Critic / LSTM / Transformer
│   ├── marl_agent.py           # 单个 agent 封装
│   └── marl_trainer.py         # MAPPO / QMIX 训练逻辑
│
├── configs/
│   ├── env_config.yaml         # 环境参数（源、涡街、地图）
│   ├── agent_config.yaml       # 网络结构、LSTM 长度等
│   └── train_config.yaml       # 学习率、batch、episode
│
├── scripts/
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 测试与指标统计
│   └── visualize.py            # 浓度场 & UAV 轨迹可视化
│
├── utils/
│   ├── logger.py               # 日志、tensorboard
│   ├── metrics.py              # T_total, E_loc, Recall 等指标
│   └── seed.py                 # 随机种子控制
│
├── experiments/
│   ├── baseline_single_agent/  # 对比实验
│   ├── no_vortex/              # 消融实验
│   └── no_memory/              # 消融实验
│
├── requirements.txt
└── README.md
