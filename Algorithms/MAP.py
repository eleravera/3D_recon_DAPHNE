import numpy as np
from Algorithms.IterativeReconstruction import IterativeReconstruction
from scipy.optimize import curve_fit

from Misc.Preview import Visualize3dImage
from matplotlib import pyplot as plt

class MAP(IterativeReconstruction):
    """!@brief  
    
    """

    def __init__(self):
        super().__init__()
        self._name = "MAP"
        
        #QUI CI METTO GLI ATTRIBUTI DI MAP, CHE AGGIUNGO IO.         
        self.plane_sums = None  # Attribute to store the sum of each plane along the axis 0 of the matrix (corresponding to the z-axis)
        self.centroids = None   #Attribute to store the centroids for each plane
        self.fit_params = None  # To store fit parameters as [(ax, bx), (ay, by)]
        self.derivative = None
        self.new_sensitivity = None
        
    
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
        
        
        centroids = self._CalculateCentroids()
        fit_param = self._FitCentroids(p0=[(1, 1), (1, 1)])
        #print(centroids, fit_param)
        self.derivative, x_terms, y_terms, z_i, der_min, der_max = self._CalculateDerivatives()
        print(f'The deritivative.min() and derivative.max() are: {self.derivative.min():.3e} and {self.derivative.max():.3e}')
        print("centroids shape:", centroids.shape) 

        filename = "../Reconstruction/derivative_check.txt"
        
        data = np.column_stack((centroids[:,0], centroids[:,1], x_terms, y_terms, z_i, der_min, der_max))
        # Scrive in modalità appendendo al file
        with open(filename, "a") as f:
            f.write("#centroids, x_term1\tx_term2\ty_term\tz_i\tder_min\tder_max\n")  # Intestazione colonne
            np.savetxt(f, data, fmt=["%.4f", "%.4f", "%.4f", "%.4f", "%d", "%.3e", "%.3e"], delimiter="\t")


        #print('centroids.shape, x_terms.shape, y_terms.shape, z_i.shape', centroids.shape, x_terms.shape, y_terms.shape, z_i.shape)
        #       
        #self._EvaluatePriorCorrection()

        
        # apply sensitivity correction and update current estimate 
        nnull = self._S != 0
                
        sensitivity = np.zeros(self._S.shape)
        sensitivity[nnull] =  1 / (self._S[nnull]) #+self.derivative[nnull])
        
        #sensitivity_new = np.zeros(self._S.shape)
        #sensitivity_new[nnull] =  1 / (self._S[nnull] + self.derivative[nnull])
        # 
        #new_denom = (self._S + self.derivative)
        
        #difference = sensitivity - sensitivity_new
        #sum_ = self._S + self.derivative
        #Visualize3dImage(sum_.astype(np.float64), slice_axis=0, symmetric_colorbar=False, title ='Sum')
        #plt.figure()
        #plt.hist(np.concatenate(np.concatenate(self._S.astype(np.float64))), bins = 100)
        
        #Visualize3dImage(self._S.astype(np.float64), slice_axis=0, symmetric_colorbar=False, title ='S')
        #Visualize3dImage(self.derivative.astype(np.float64), slice_axis=0, symmetric_colorbar=True, title ='derivate')
        #Visualize3dImage(new_denom.astype(np.float64), slice_axis=0, symmetric_colorbar=False, title ='new_denom')
        
        #plt.figure()
        #plt.hist(np.concatenate(np.concatenate(new_denom.astype(np.float64))), bins = 100)
        #plt.title('new denom')
        
        #plt.show()
                
        self._image = self._image * sensitivity * tmp        
        
        
        




    def __EvaluateSensitivity(self):
        """!@brief
             Backproject a vector filled with 1: the obtained image is often called
            sensitivity image
        """
        self._S = self.BackProjection(np.ones(self.GetNumberOfProjections()))
        nnull = self._S != 0
        
        
        #self._S[nnull] = 1 / self._S[nnull]

    def EvaluateWeightingFactors(self):
        """!@brief
            Compute all the weighting factors needed for the update rule
        """
        self.__EvaluateSensitivity()
        #np.savez('/home/eleonora/3D_recon_DAPHNE/Algorithms/sensitivity.npz', matrixname = self._S)
        

        

    def _CalculatePlaneSums(self):
        """
            Computes the sum of each plane along axis 0 and stores it in self.plane_sums.
        """
        self.plane_sums = np.sum(self._image, axis=(1, 2))  # Sum of each plane (i.e., along axis 1 and 2)
        return 


    def _CalculateCentroids(self):
        """
            Computes the centroids for each plane along axis 0 and stores them in self.centroids.
            The axis 0 is selected because the distribution is supposed to be linear along this axis.
        """
        self._CalculatePlaneSums()  # Ensure plane sums are calculated first

        x_coords = np.arange(self._image.shape[2])  # X-axis coordinates
        y_coords = np.arange(self._image.shape[1])  # Y-axis coordinates

        # Compute centroids for each plane along axis 0
        self.centroids = np.array([
            (                
                np.sum(x_coords * np.sum(self._image[i, :, :], axis=1)) / self.plane_sums[i], #centroid on y
                np.sum(y_coords * np.sum(self._image[i, :, :], axis=0)) / self.plane_sums[i]  #centroid on x
                
            
            ) if self.plane_sums[i] > 0 else (np.nan, np.nan)  # Handle empty planes
            for i in range(self._image.shape[0])
        ])

        return self.centroids #[centroids on y, on x]
    
    
    
    
    def _FitCentroids(self, p0=None):
        """
            Fits the centroid positions with a linear model using scipy curve_fit.
            p0 parameters can be passed as: p0=[(ax, bx), (ay, by)]. 
        """
        if self.centroids is None:
            raise ValueError("Centroids have not been calculated. Run __CalculateCentroids first.")

        # Extract valid data (ignore NaN values)
        z_vals = np.arange(len(self.centroids))  # z-coordinates (plane indices)
        valid_mask = ~np.isnan(self.centroids[:, 0]) & ~np.isnan(self.centroids[:, 1])  # Ignore NaNs
        z_vals = z_vals[valid_mask]
        x_vals = self.centroids[:, 0][valid_mask]
        y_vals = self.centroids[:, 1][valid_mask]

        # Define a linear model: x(z) = a * z + b, y(z) = c * z + d
        def linear_model(z, a, b):
            return a * z + b

        if p0 is None: 
            # Initial guess: a=0, b=50 (center of the matrix)
            p0_x = [0, 50]
            p0_y = [0, 50]
        else:
            p0_x = p0[0]
            p0_y = p0[1]

        # Perform curve fitting
        params_x, _ = curve_fit(linear_model, z_vals, x_vals, p0=p0_x)
        params_y, _ = curve_fit(linear_model, z_vals, y_vals, p0=p0_y)

        # Store fit parameters as [(ax, bx), (ay, by)]
        self.fit_params = (params_x, params_y)

        return self.fit_params
        
        
        
        
    def _CalculateDerivatives(self):
        """
            Calculates the derivatives of the prior based on the centroids and fit parameters.
            The result is a 3D matrix where the derivatives are calculated over x, y, and z axes.
        """
        # Get coordinates
        x_coords = np.arange(self._image.shape[2])  # X-axis coordinates
        y_coords = np.arange(self._image.shape[1])  # Y-axis coordinates
        z_coords = np.arange(self._image.shape[0])  # Z-axis coordinates (planes)

        # Initialize the 3D derivative matrix
        derivative_matrix = np.zeros(self._image.shape)  # 3D matrix (z, y, x)
        
        
        x_terms = [] #solo per l'output di debugging su txt
        y_terms =[] #solo per l'output di debuggin su txt
        z_i = [] # solo per l'output di debuggin su txt
        der_min = []
        der_max = []
        
        # Calculate the derivatives for each plane (along z, x, y)
        for i in range(self._image.shape[0]):  # Loop over each plane
            # Compute the x_term and y_term for the derivatives along z
            x_term = 2 * (self.centroids[i, 0] - z_coords[i] * self.fit_params[0][0] - self.fit_params[0][1]) * \
                     (x_coords - self.centroids[i, 0]) / self.plane_sums[i]

            y_term = 2 * (self.centroids[i, 1] - z_coords[i] * self.fit_params[1][0] - self.fit_params[1][1]) * \
                     (y_coords - self.centroids[i, 1]) / self.plane_sums[i]
                                          
            
            # Now store the x_term and y_term across the 3D matrix
            # The derivative in the x direction for each plane is spread across the x-dimension
            # Likewise, the derivative in the y direction for each plane is spread across the y-dimension

            # The terms depend on z, so we will broadcast the terms along the other axes
            #derivative_matrix[i, :, :] = np.outer(y_term, x_term)  # Broadcasting over y and x dimensions
            
            
            
            X, Y = np.meshgrid(x_term, y_term)
            
            derivative_matrix[i, :, : ] = X + Y 
            
            x_terms.append(z_coords[i] * self.fit_params[0][0] + self.fit_params[0][1])
            y_terms.append(z_coords[i] * self.fit_params[1][0] + self.fit_params[1][1])
            z_i.append(z_coords[i])
            der_min.append((X + Y).min())
            der_max.append((X + Y).max())
            
        derivative_matrix = derivative_matrix#/(np.pi**100)  #Because the normalization of the prior is pi^N

        x_terms = np.array(x_terms)
        y_terms = np.array(y_terms)
        z_i = np.array(z_i)
        der_min = np.array(der_min)
        der_max = np.array(der_max)


        return derivative_matrix, x_terms, y_terms, z_i, der_min, der_max
        
        
          

    def _EvaluatePriorCorrection(self):
        #put together all the contributions, in order to substitute this in the PerformSingleIteration function.
        self.new_sensitivity = 1 / (1/self._S + self.derivative)

        return

