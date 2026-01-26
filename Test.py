import os
import tempfile
import unittest

import numpy as np

from marl_leakage_search.envs.marl_env import PlumeEnv


def _write_dummy_field(dir_path: str, name: str = "field_0.npz") -> str:
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 10, 11)
    X, Y = np.meshgrid(x, y, indexing="xy")
    concentration = np.exp(-((X - 5) ** 2 + (Y - 5) ** 2) / 10.0)

    sources = np.array([[5.0, 5.0, 10.0]], dtype=float)
    obstacles = np.array([[3.0, 3.0, 1.0]], dtype=float)
    wind_speed = np.array([1.5], dtype=float)

    path = os.path.join(dir_path, name)
    np.savez_compressed(
        path,
        concentration_field=concentration,
        sources=sources,
        obstacles=obstacles,
        wind_speed=wind_speed,
        x=x,
        y=y,
    )
    return path


class TestPlumeEnv(unittest.TestCase):
    def test_reset_step_observe_render(self):
        field_dir = r"C:\\Users\\Charlotte\\NewStart\\EXP\\GaosiLeak\\marl_leakage_search\\envs\\generated_fields"
        self.assertTrue(os.path.isdir(field_dir), f"Field directory not found: {field_dir}")

        env = PlumeEnv(field_dir=field_dir, num_agents=2, seed=None, uav_params={"max_speed": 2.0})

        obs = env.reset()
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0].shape[0], 4)

        next_obs, rewards, done, info = env.step([0, 3])
        self.assertEqual(len(next_obs), 2)
        self.assertEqual(len(rewards), 2)
        self.assertIn("found_sources", info)

        concentrations = env.observe()
        self.assertEqual(len(concentrations), 2)

        render_dir = os.path.join(os.getcwd(), "debug_renders")
        os.makedirs(render_dir, exist_ok=True)
        render_path = os.path.join(render_dir, "render.png")
        env.render(save_path=render_path, show=False)
        print("saved to:", render_path)


if __name__ == "__main__":
    unittest.main()
