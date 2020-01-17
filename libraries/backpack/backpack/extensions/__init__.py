"""
BackPACK Extensions
"""

from .curvmatprod import CMP
from .firstorder import BatchGrad, BatchL2Grad, SumGradSquared, Variance
from .secondorder import (HBP,  KFAC, KFLR, KFRA, 
                                KFAC2, KFLR2, KFRA2,  # Huh
                                DiagGGN, DiagGGNExact, DiagGGNMC, DiagHessian)
