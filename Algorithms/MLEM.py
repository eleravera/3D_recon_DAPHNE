import numpy as np
from Algorithms.IterativeReconstruction import IterativeReconstruction


class MLEM(IterativeReconstruction):
    """!@brief  
    Implements the Maxmimum Likelihood Estimation Maximization algorithm (MLEM).  
    L. A. Shepp and Y. Vardi, "Maximum Likelihood Reconstruction for Emission Tomography,"
    in IEEE Transactions on Medical Imaging, vol. 1, no. 2, pp. 113-122, Oct. 1982.
    """

    def __init__(self):
        super().__init__()
        self._name = "MLEM"

    
    def PerfomSingleIteration(self):
        """!@brief 
            Implements the update rule for MLEM
        """
        # forward projection
        proj = self.ForwardProjection(self._image)
        # this avoid 0 division
        nnull = proj != 0
        # comparison with experimental measures (ratio)
        proj[nnull] = self._projection_data[nnull] / proj[nnull]
        #  backprojection
        tmp = self.BackProjection(proj)
        # apply sensitivity correction and update current estimate 
        self._image = self._image * self._S * tmp

    def __EvaluateSensitivity(self):
        """!@brief
             Backproject a vector filled with 1: the obtained image is often called
            sensitivity image
        """
        self._S = self.BackProjection(np.ones(self.GetNumberOfProjections()))
        nnull = self._S != 0
        self._S[nnull] = 1 / self._S[nnull]

    def EvaluateWeightingFactors(self):
        """!@brief
            Compute all the weighting factors needed for the update rule
        """
        self.__EvaluateSensitivity()
