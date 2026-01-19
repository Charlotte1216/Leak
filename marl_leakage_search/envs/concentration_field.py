"""
concentration_field.py
浓度场接口
"""
import numpy as np


class ConcentrationField2D:
    """
    Combined concentration field:
    Gaussian plume (mean field) + Kármán vortex perturbation
    """

    def __init__(
        self,
        plume_model,
        vortex_model=None,
        sources=None,
        obstacles=None,
        clamp_nonnegative=True
    ):
        """
        Parameters
        ----------
        plume_model : GaussianPlume2D
        vortex_model : KarmanVortexStreet or None
        sources : list of dict
            [{'x': float, 'y': float, 'Q': optional}]
        obstacles : list of dict
            [{'x': float, 'y': float, 'radius': float}]
        """
        self.plume = plume_model
        self.vortex = vortex_model
        self.sources = sources if sources is not None else []
        self.obstacles = obstacles if obstacles is not None else []
        self.clamp_nonnegative = clamp_nonnegative

    def concentration(self, x, y, t=0.0):
        """
        Compute total concentration at (x, y, t).
        """
        # Mean plume concentration
        C = self.plume.multi_source_concentration(x, y, self.sources)

        # Vortex perturbation
        if self.vortex is not None and self.obstacles:
            for obs in self.obstacles:
                C += self.vortex.perturbation(x, y, t, obstacle=obs)

        if self.clamp_nonnegative:
            C = np.maximum(C, 0.0)

        return C


