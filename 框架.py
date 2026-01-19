# marl_leakage_search/
# │
# ├── envs/
# │   ├── __init__.py
# │   ├── plume_model.py          # 高斯烟羽模型
# │   ├── vortex_model.py         # 卡门涡街扰动模型
# │   ├── concentration_field.py  # plume + vortex 总浓度场
# │   ├── uav_dynamics.py         # UAV 动力学 & 能耗
# │   └── marl_env.py             # Gym-style MARL 环境
# │
# ├── agents/
# │   ├── __init__.py
# │   ├── networks.py             # Actor / Critic / LSTM / Transformer
# │   ├── marl_agent.py           # 单个 agent 封装
# │   └── marl_trainer.py         # MAPPO / QMIX 训练逻辑
# │
# ├── configs/
# │   ├── env_config.yaml         # 环境参数（源、涡街、地图）
# │   ├── agent_config.yaml       # 网络结构、LSTM 长度等
# │   └── train_config.yaml       # 学习率、batch、episode
# │
# ├── scripts/
# │   ├── train.py                # 训练入口
# │   ├── evaluate.py             # 测试与指标统计
# │   └── visualize.py            # 浓度场 & UAV 轨迹可视化
# │
# ├── utils/
# │   ├── logger.py               # 日志、tensorboard
# │   ├── metrics.py              # T_total, E_loc, Recall 等指标
# │   └── seed.py                 # 随机种子控制
# │
# ├── experiments/
# │   ├── baseline_single_agent/  # 对比实验
# │   ├── no_vortex/              # 消融实验
# │   └── no_memory/              # 消融实验
# │
# ├── requirements.txt
# └── README.md

import os

# 定义项目结构
structure = {
    "marl_leakage_search": {
        "envs": [
            "__init__.py",
            "plume_model.py",  # 高斯烟羽模型
            "vortex_model.py",  # 卡门涡街扰动模型
            "concentration_field.py",  # plume + vortex 总浓度场
            "uav_dynamics.py",  # UAV 动力学 & 能耗
            "marl_env.py"  # Gym-style MARL 环境
        ],
        "agents": [
            "__init__.py",
            "networks.py",  # Actor / Critic / LSTM / Transformer
            "marl_agent.py",  # 单个 agent 封装
            "marl_trainer.py"  # MAPPO / QMIX 训练逻辑
        ],
        "configs": [
            "env_config.yaml",  # 环境参数（源、涡街、地图）
            "agent_config.yaml",  # 网络结构、LSTM 长度等
            "train_config.yaml"  # 学习率、batch、episode
        ],
        "scripts": [
            "train.py",  # 训练入口
            "evaluate.py",  # 测试与指标统计
            "visualize.py"  # 浓度场 & UAV 轨迹可视化
        ],
        "utils": [
            "__init__.py",
            "logger.py",  # 日志、tensorboard
            "metrics.py",  # T_total, E_loc, Recall 等指标
            "seed.py"  # 随机种子控制
        ],
        "experiments": {
            "baseline_single_agent": [],
            "no_vortex": [],
            "no_memory": []
        },
        "requirements.txt": None,
        "README.md": None
    }
}


def create_structure(base_path, structure):
    """递归创建项目结构"""
    for name, content in structure.items():
        current_path = os.path.join(base_path, name)

        if isinstance(content, dict):
            # 创建子目录
            os.makedirs(current_path, exist_ok=True)
            create_structure(current_path, content)
        elif isinstance(content, list):
            # 创建目录和文件
            os.makedirs(current_path, exist_ok=True)
            for item in content:
                file_path = os.path.join(current_path, item)
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 写入基础注释
                    if item.endswith('.py'):
                        f.write(f'"""\n{item}\n"""\n\n')
                    elif item.endswith('.yaml'):
                        f.write(f'# {item}\n\n')
        elif content is None:
            # 创建空文件
            with open(current_path, 'w', encoding='utf-8') as f:
                if name == "requirements.txt":
                    f.write("# 项目依赖库\n\n")
                elif name == "README.md":
                    f.write("# MARL 气体泄漏源搜索项目\n\n")


# 执行创建
if __name__ == "__main__":
    create_structure(".", structure)
    print("✅ 项目结构创建完成！")