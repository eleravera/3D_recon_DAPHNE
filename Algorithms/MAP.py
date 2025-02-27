import numpy as np
import os
from Algorithms.IterativeReconstruction import IterativeReconstruction
from scipy.optimize import curve_fit

from Misc.Preview import Visualize3dImage
from matplotlib import pyplot as plt

LAMBDA = 8.75e+2 #TODO

class MAP(IterativeReconstruction):
    """!@brief  
    
    """

    def __init__(self):
        super().__init__()
        self._name = "MAP"
        
        self.plane_sums = None  # Attribute to store the sum of each plane along the axis 0 of the matrix (corresponding to the z-axis)
        self.centroids = None   #Attribute to store the centroids for each plane
        self.fit_params = None  # To store fit parameters as [(ax, bx), (ay, by)]
        self.derivative = None
        self.new_sensitivity = None
        self._Ssum = None
        
        self.sensitivity_file = None #USed to save the system matrix and the derivatives 
        
    
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
        
        #Calculates element for prior correction
        centroids = self._CalculateCentroids()
        fit_param = self._FitCentroids(p0=[(1, 1), (1, 1)])
        self.derivative, x_terms, y_terms, z_i, der_min, der_max = self._CalculateDerivatives()

        #if iteration <20 MLEM, else MAP
        if self._iteration < 20:
            self.new_sensitivity = np.zeros(self._S.shape)
            nnull = self._S != 0
            self.new_sensitivity[nnull] = 1 / self._S[nnull]
        else: 
            self._EvaluatePriorCorrection()
            self.new_sensitivity = np.zeros(self._Ssum.shape)
            nnull = self._Ssum != 0
            self.new_sensitivity[nnull] = 1 / self._Ssum[nnull]
        
        self.save_derivative_matrix()
        self.visualization_per_step()
        self.print_some_output()
        
        self._image = self._image * self.new_sensitivity * tmp   
        


    def __EvaluateSensitivity(self):
        """!@brief
             Backproject a vector filled with 1: the obtained image is often called
            sensitivity image
        """
        #ATTENZIONE: sto richiamando sensitivity il denominatore
        self._S = self.BackProjection(np.ones(self.GetNumberOfProjections()))
        self.sensitivity_file = os.path.join(os.path.dirname(self._output_file_name), "sensitivity.npz") 
        np.savez(self.sensitivity_file, system_matrix=self._S)
        #print(f"File created: {self.sensitivity_file}, and System matrix saved")
        


    def EvaluateWeightingFactors(self):
        """!@brief
            Compute all the weighting factors needed for the update rule
        """
        self.__EvaluateSensitivity()

        

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
                                          
            X, Y = np.meshgrid(x_term, y_term)
            
            derivative_matrix[i, :, : ] = X + Y 
            
            x_terms.append(z_coords[i] * self.fit_params[0][0] + self.fit_params[0][1])
            y_terms.append(z_coords[i] * self.fit_params[1][0] + self.fit_params[1][1])
            z_i.append(z_coords[i])
            der_min.append((X + Y).min())
            der_max.append((X + Y).max())
            
        derivative_matrix = derivative_matrix

        x_terms = np.array(x_terms)
        y_terms = np.array(y_terms)
        z_i = np.array(z_i)
        der_min = np.array(der_min)
        der_max = np.array(der_max)


        return derivative_matrix, x_terms, y_terms, z_i, der_min, der_max
        
        
          

    def _EvaluatePriorCorrection(self):
        #put together all the contributions, in order to substitute this in the PerformSingleIteration function.
        self._Ssum = self._S + self.derivative * LAMBDA
        return


"""
*******************************************************************************************************
Some usefull functions for debugging
*******************************************************************************************************
"""

    def save_derivative_matrix(self): 
       #Save the derivatives
       data = np.load(self.sensitivity_file, allow_pickle=True)
       existing_data = {key: data[key] for key in data.files}      
       key = f"derivative_it{self._iteration+1}"
       existing_data[key] = self.derivative
       np.savez(self.sensitivity_file, **existing_data)
       #print(f"Matrix added to file: {self.sensitivity_file}")
       return 
       
    def visualization_per_step(self):
        if (self._iteration > 20):
            Visualize3dImage(self.derivative.astype('float64'), slice_axis=0, symmetric_colorbar=False, title ='derivative')
            Visualize3dImage(self._Ssum.astype('float64'), slice_axis=0, symmetric_colorbar=False, title ='_S + derivative * LAMBDA')
            Visualize3dImage(self.new_sensitivity.astype('float64'), slice_axis=0, symmetric_colorbar=False, title ='new sensitivity')
            plt.show()
            
            
    def print_some_output(self):        
        print(f'new_sensitivity.min = {self.new_sensitivity.min()}, new_sensitivity.max = {self.new_sensitivity.max()}')        
        print(f'derivative.min = {self.derivative.min()}, derivative.max = {self.derivative.max()}')
        
