import numpy as np
import matplotlib.pyplot as plt
from concentration_field import ConcentrationField
from plume_model import GaussianPlumeModel2D
from vortex_model import KarmanVortexStreet
import matplotlib.patches as patches


def main():
    # 1. 定义模型参数
    plume_params = {
        "L": 100.0,  # downstream attenuation length scale
        "sigma_y": 5.0  # lateral dispersion coefficient
    }

    vortex_params = {
        "A": 0.3,  # perturbation amplitude
        "omega": 0.2,  # shedding frequency
        "kx": 0.3,  # streamwise wave number
        "ky": 2.0,  # cross-stream wave number
        "decay": 0.05,  # downstream attenuation
        "sigma_w0": 2.0,  # initial wake width
        "beta": 0.05,  # wake spreading rate
        "phase": 0.0
    }

    # 2. 设置源和障碍物
    sources = [
        {"x": 0, "y": 20, "Q": 20.0},
        {"x": 10, "y": 70, "Q": 15.0},
        {"x": 0, "y": 80, "Q": 15.0}
    ]

    obstacles = [
        # {"x": 30, "y": 50, "radius": 1.0},  # example obstacle
        {"x": 5, "y": 20, "radius": 1.0},  # example obstacle
        {"x": 15, "y": 75, "radius": 3.0}  # example obstacle
    ]

    # 3. 创建浓度场对象
    concentration_model = ConcentrationField(
        sources=sources,
        obstacles=obstacles,
        wind_speed=1.0,
        plume_params=plume_params,
        vortex_params=vortex_params,
        keep_plume_behind_obstacle=True,  # 关闭后方烟羽
        noise_std=0.05  # 可选：加入噪声
    )

    # 4. 网格设置
    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    X, Y = np.meshgrid(x, y)

    # 5. 可视化：高斯烟羽、涡街扰动、最终浓度场
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 5.1 可视化高斯烟羽（平均场）
    C_plume = concentration_model.plume.calculate_concentration(X, Y, sources)
    c1 = axs[0].imshow(C_plume, extent=[0, 100, 0, 100], origin='lower', aspect='auto', cmap='viridis')
    axs[0].set_title("Gaussian Plume (Mean Field)")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    fig.colorbar(c1, ax=axs[0])

    # 标出源头点
    source_x = [source["x"] for source in sources]
    source_y = [source["y"] for source in sources]
    axs[0].scatter(source_x, source_y, color='red', label='Sources', edgecolor='black', zorder=5)
    axs[0].legend()

    # 5.2 可视化涡街扰动
    C_vortex = np.zeros_like(X)
    for obs in obstacles:
        C_vortex += concentration_model.vortex.perturbation(X, Y, t=0.0, obstacle=obs)
    c2 = axs[1].imshow(C_vortex, extent=[0, 100, 0, 100], origin='lower', aspect='auto', cmap='coolwarm')
    axs[1].set_title("Vortex Perturbation")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    fig.colorbar(c2, ax=axs[1])

    # 标出障碍物
    obstacle_x = [obs["x"] for obs in obstacles]
    obstacle_y = [obs["y"] for obs in obstacles]
    for obs in obstacles:
        circle = patches.Circle((obs["x"], obs["y"]), obs["radius"], color='blue', alpha=0.3, label='Obstacle Radius')
        axs[1].add_patch(circle)
    # axs[1].scatter(obstacle_x, obstacle_y, color='blue', label='Obstacles', edgecolor='black', zorder=5)
    # axs[1].legend()

    # 5.3 最终浓度场（高斯烟羽 + 涡街扰动）
    C_total = concentration_model.concentration(X, Y, t=0.0)
    c3 = axs[2].imshow(C_total, extent=[0, 100, 0, 100], origin='lower', aspect='auto', cmap='plasma')
    axs[2].set_title("Total Concentration Field")
    axs[2].set_xlabel("X")
    axs[2].set_ylabel("Y")
    fig.colorbar(c3, ax=axs[2])

    # 标出源头点和障碍物
    axs[2].scatter(source_x, source_y, color='red', label='Sources', edgecolor='black', zorder=5)
    axs[2].scatter(obstacle_x, obstacle_y, color='blue', label='Obstacles', edgecolor='black', zorder=5)
    axs[2].legend()


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
