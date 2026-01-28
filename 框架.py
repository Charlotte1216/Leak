# # marl_leakage_search/
# # │
# # ├── envs/
# # │   ├── __init__.py
# # │   ├── plume_model.py          # 高斯烟羽模型
# # │   ├── vortex_model.py         # 卡门涡街扰动模型
# # │   ├── concentration_field.py  # plume + vortex 总浓度场
# # │   ├── uav_dynamics.py         # UAV 动力学 & 能耗
# # │   └── marl_env.py             # Gym-style MARL 环境
# # │
# # ├── agents/
# # │   ├── __init__.py
# # │   ├── networks.py             # Actor / Critic / LSTM / Transformer
# # │   ├── marl_agent.py           # 单个 agent 封装
# # │   └── marl_trainer.py         # MAPPO / QMIX 训练逻辑
# # │
# # ├── configs/
# # │   ├── env_config.yaml         # 环境参数（源、涡街、地图）
# # │   ├── agent_config.yaml       # 网络结构、LSTM 长度等
# # │   └── train_config.yaml       # 学习率、batch、episode
# # │
# # ├── scripts/
# # │   ├── train.py                # 训练入口
# # │   ├── evaluate.py             # 测试与指标统计
# # │   └── visualize.py            # 浓度场 & UAV 轨迹可视化
# # │
# # ├── utils/
# # │   ├── logger.py               # 日志、tensorboard
# # │   ├── metrics.py              # T_total, E_loc, Recall 等指标
# # │   └── seed.py                 # 随机种子控制
# # │
# # ├── experiments/
# # │   ├── baseline_single_agent/  # 对比实验
# # │   ├── no_vortex/              # 消融实验
# # │   └── no_memory/              # 消融实验
# # │
# # ├── requirements.txt
# # └── README.md

# import os

# # 定义项目结构
# structure = {
#     "marl_leakage_search": {
#         "envs": [
#             "__init__.py",
#             "plume_model.py",  # 高斯烟羽模型
#             "vortex_model.py",  # 卡门涡街扰动模型
#             "concentration_field.py",  # plume + vortex 总浓度场
#             "uav_dynamics.py",  # UAV 动力学 & 能耗
#             "marl_env.py"  # Gym-style MARL 环境
#         ],
#         "agents": [
#             "__init__.py",
#             "networks.py",  # Actor / Critic / LSTM / Transformer
#             "marl_agent.py",  # 单个 agent 封装
#             "marl_trainer.py"  # MAPPO / QMIX 训练逻辑
#         ],
#         "configs": [
#             "env_config.yaml",  # 环境参数（源、涡街、地图）
#             "agent_config.yaml",  # 网络结构、LSTM 长度等
#             "train_config.yaml"  # 学习率、batch、episode
#         ],
#         "scripts": [
#             "train.py",  # 训练入口
#             "evaluate.py",  # 测试与指标统计
#             "visualize.py"  # 浓度场 & UAV 轨迹可视化
#         ],
#         "utils": [
#             "__init__.py",
#             "logger.py",  # 日志、tensorboard
#             "metrics.py",  # T_total, E_loc, Recall 等指标
#             "seed.py"  # 随机种子控制
#         ],
#         "experiments": {
#             "baseline_single_agent": [],
#             "no_vortex": [],
#             "no_memory": []
#         },
#         "requirements.txt": None,
#         "README.md": None
#     }
# }


# def create_structure(base_path, structure):
#     """递归创建项目结构"""
#     for name, content in structure.items():
#         current_path = os.path.join(base_path, name)

#         if isinstance(content, dict):
#             # 创建子目录
#             os.makedirs(current_path, exist_ok=True)
#             create_structure(current_path, content)
#         elif isinstance(content, list):
#             # 创建目录和文件
#             os.makedirs(current_path, exist_ok=True)
#             for item in content:
#                 file_path = os.path.join(current_path, item)
#                 with open(file_path, 'w', encoding='utf-8') as f:
#                     # 写入基础注释
#                     if item.endswith('.py'):
#                         f.write(f'"""\n{item}\n"""\n\n')
#                     elif item.endswith('.yaml'):
#                         f.write(f'# {item}\n\n')
#         elif content is None:
#             # 创建空文件
#             with open(current_path, 'w', encoding='utf-8') as f:
#                 if name == "requirements.txt":
#                     f.write("# 项目依赖库\n\n")
#                 elif name == "README.md":
#                     f.write("# MARL 气体泄漏源搜索项目\n\n")


# # 执行创建
# if __name__ == "__main__":
#     create_structure(".", structure)
#     print("✅ 项目结构创建完成！")

import os
from pathlib import Path

def generate_directory_tree(directory_path, max_depth=None, show_files=True, indent='    '):
    """
    生成文件夹目录树结构
    
    参数:
    directory_path: 目标文件夹路径
    max_depth: 最大遍历深度，None表示无限制
    show_files: 是否显示文件
    indent: 缩进字符
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        print(f"错误：路径 '{directory_path}' 不存在")
        return
    if not directory_path.is_dir():
        print(f"错误：'{directory_path}' 不是一个文件夹")
        return
    
    def _generate_tree(path, current_depth=0, prefix=''):
        if max_depth is not None and current_depth > max_depth:
            return ''
        
        tree_str = ''
        if current_depth == 0:
            tree_str += f"{path.name}/\n"
        else:
            tree_str += f"{prefix}└── {path.name}/\n"
        
        try:
            items = sorted(list(path.iterdir()), key=lambda x: (x.is_file(), x.name.lower()))
            
            for i, item in enumerate(items):
                is_last = (i == len(items) - 1)
                new_prefix = prefix + ('    ' if is_last else '│   ')
                
                if item.is_dir():
                    tree_str += _generate_tree(item, current_depth + 1, new_prefix)
                elif show_files and item.is_file():
                    connector = '└── ' if is_last else '├── '
                    tree_str += f"{new_prefix}{connector}{item.name}\n"
                    
        except PermissionError:
            tree_str += f"{prefix}    └── [权限拒绝]\n"
        
        return tree_str
    
    return _generate_tree(directory_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='生成文件夹目录结构')
    parser.add_argument('path', nargs='?', default='.', help='目标文件夹路径（默认为当前目录）')
    parser.add_argument('-d', '--depth', type=int, help='最大显示深度')
    parser.add_argument('-f', '--no-files', action='store_true', help='不显示文件，只显示文件夹')
    parser.add_argument('-o', '--output', help='将结果保存到文件')
    
    args = parser.parse_args()
    
    tree = generate_directory_tree(
        args.path, 
        max_depth=args.depth, 
        show_files=not args.no_files
    )
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(tree)
        print(f"目录结构已保存到: {args.output}")
    else:
        print(tree)