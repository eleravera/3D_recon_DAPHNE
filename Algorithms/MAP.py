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
        z_vals = np.arange(self._image.shape[0])
        centroids = self._CalculateCentroids()
        
        CORDINATE_LANEX = 150
        P_X, P_Y = 90, -8
        #_ = self._FitCentroids(z_vals, centroids, p0=[(1, 1), (1, 1)])
        #print('self.fit_params: ', self.fit_params)
        _ = self._FitCentroids(np.append(z_vals, CORDINATE_LANEX), np.vstack([centroids, [P_X, P_Y]]), p0=[(1, 1), (1, 1)])
        #print('self.fit_params: ', self.fit_params)

        self.derivative, x_terms, y_terms, z_i, der_min, der_max = self._CalculateDerivatives()

        #_ = self._DefineRelationsForFitParam(z_vals) #used just to check the fit parameters values
        #der_centroids = self._CalulateCentroidDerivate() # la shape è (2, 100)
        #der_fitparam = self._CalulateFitParamDerivate(z_vals, der_centroids) # sono 4 numeri. 
        #derivative_matrix, old_x, old_y, new_x, new_y = self._CalulateDerivativesP1(z_vals, der_centroids, der_fitparam)
        #print('ratio x , ratio y: ', new_x/old_x,  new_y/old_y )
        #print('der_centroids[0,:], der_fitparam[0], der_fitparam[1]: ', der_centroids[0,:], der_fitparam[0], der_fitparam[1])
        

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
        
        #derivative1 = derivative_matrix
        #self.save_derivative_matrix()
        #self.visualization_per_step()
        #self.print_some_output()
        
        #print('centroids: ', self.centroids)
        
        self._image = self._image * self.new_sensitivity * tmp   
        


    def __EvaluateSensitivity(self):
        """!@brief
             Backproject a vector filled with 1: the obtained image is often called
            sensitivity image
        """
        #ATTENZIONE: sto richiamando ._S il denominatore
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
    
    
    
    
    def _FitCentroids(self, z_vals, centroids, p0=None):
        """
            Fits the centroid positions with a linear model using scipy curve_fit. Fit performed using a linear model: x(z) = a * z + b
            p0 parameters can be passed as: p0=[(ax, bx), (ay, by)]. 
            z_vals = z-coordinates (plane indices)
        """
        if self.centroids is None:
            raise ValueError("Centroids have not been calculated. Run __CalculateCentroids first.")

        # Extract valid data (ignore NaN values)
        valid_mask = ~np.isnan(centroids[:, 0]) & ~np.isnan(centroids[:, 1])  # Ignore NaNs
        z_vals = z_vals[valid_mask]
        x_vals = centroids[:, 0][valid_mask]
        y_vals = centroids[:, 1][valid_mask]

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
        self.fit_params = (params_x, params_y)

        return self.fit_params
        
        
        
    def _DefineRelationsForFitParam(self, z_vals): 
        """NON VIENE DAVVERO USATA - SOLO PER AVERE UN RISCONTRO DI VALORI TRA FIT E CALCOLO. 
        Computes the fitting parameters ax, bx, ay, and by.  
        These parameters define the relationship between centroid coordinates  
        and the depth values (z_vals) of the image.

        Args:
            z_vals (numpy array): Array containing depth values.

        Returns:
            tuple: Computed values (ax, bx, ay, by).
        """
        ax_num = self._image.shape[0] * np.sum(z_vals * self.centroids[:, 0]) - np.sum(z_vals) * np.sum(self.centroids[:, 0]) 
        ax_denom = self._image.shape[0] * np.sum(z_vals * z_vals) - np.sum(z_vals)**2
        ax = ax_num/ax_denom

        ay_num = self._image.shape[0] * np.sum(z_vals * self.centroids[:, 1]) - np.sum(z_vals) * np.sum(self.centroids[:, 1]) 
        ay_denom = self._image.shape[0] * np.sum(z_vals * z_vals) - np.sum(z_vals)**2
        ay = ay_num/ay_denom
        
        bx = (np.sum(self.centroids[:,0]) - ax * np.sum(z_vals)) / self._image.shape[0]
        by = (np.sum(self.centroids[:,1]) - ay * np.sum(z_vals)) / self._image.shape[0]       
        
        #print('\n\n ax, ay, bx, by, self param: ', ax, ay, bx, by, self.fit_params)
        """PER RICORDARTI LA CORRISPONDENZA:
        ax = self.fit_params[0][0]
        bx = self.fit_params[0][1]
        ay = self.fit_params[1][0]
        by = self.fit_params[1][1]"""
        return ax, bx, ay, by   
        
        
    def _CalulateCentroidDerivate(self):
        """
        Computes the derivatives of centroid coordinates with respect to the z-axis.  
        The derivatives are calculated for each plane in the image.

        Returns:
            numpy array: Array containing the derivatives of the x and y centroids.
        """
        x_coords = np.arange(self._image.shape[2])  # X-axis coordinates
        y_coords = np.arange(self._image.shape[1])  # Y-axis coordinates

        x_terms = [] 
        y_terms =[] 
        for i in range(self._image.shape[0]):  # Loop over each plane
            # Compute the x_term and y_term for the derivatives along z
            x_term = (x_coords - self.centroids[i, 0]) / self.plane_sums[i]

            y_term = (y_coords - self.centroids[i, 1]) / self.plane_sums[i]
        
        return np.array([np.array(x_term), np.array(y_term)])
         

    def _CalulateFitParamDerivate(self, z_vals, der_centroids):
        """
        Computes the derivatives of the fitting parameters (ax, bx, ay, by)  
        with respect to the centroid derivatives.

        Args:
            z_vals (numpy array): Array containing depth values.
            der_centroids (numpy array): Array of centroid derivatives.

        Returns:
            numpy array: Array containing the computed derivatives  
                         [der_ax, der_bx, der_ay, der_by].
        """
        cost = 1/(self._image.shape[0] * np.sum(z_vals*z_vals) - (np.sum(z_vals))**2)
        der_ax = cost * ( self._image.shape[0] * np.sum(z_vals * der_centroids[0,:]) - np.sum(z_vals) * np.sum(der_centroids[0,:]) )
        der_ay = cost * ( self._image.shape[0] * np.sum(z_vals * der_centroids[1,:]) - np.sum(z_vals) * np.sum(der_centroids[1,:]) )

        der_bx = ( np.sum(der_centroids[0,:]) - der_ax * np.sum(z_vals) ) / self._image.shape[0] 
        der_by = ( np.sum(der_centroids[1,:]) - der_ay * np.sum(z_vals) ) / self._image.shape[0]                

        return  np.array([der_ax, der_bx, der_ay, der_by])
        
        
    def _CalulateDerivativesP1(self, z_vals, der_centroids, der_fitparam):
        # Get coordinates
        x_coords = np.arange(self._image.shape[2])  # X-axis coordinates
        y_coords = np.arange(self._image.shape[1])  # Y-axis coordinates

        # Initialize the 3D derivative matrix
        derivative_matrix = np.zeros(self._image.shape)  # 3D matrix (z, y, x)
        
        additional_term_x = []
        additional_term_y = []
        old_term_x = []
        old_term_y = []
        
        # Calculate the derivatives for each plane (along z, x, y)
        for i in range(self._image.shape[0]):  # Loop over each plane
            # Compute the x_term and y_term for the derivatives along z
            x_term = 2 * (self.centroids[i, 0] - z_vals[i] * self.fit_params[0][0] - self.fit_params[0][1]) * \
                     ( der_centroids[0,i] - z_vals[i] * der_fitparam[0] - der_fitparam[1] )                    
            


            y_term = 2 * (self.centroids[i, 1] - z_vals[i] * self.fit_params[1][0] - self.fit_params[1][1]) * \
                     ( der_centroids[1,i] - z_vals[i] * der_fitparam[2] - der_fitparam[3] )  
            
            additional_term_x.append(- z_vals[i] * der_fitparam[0] - der_fitparam[1] ) 
            additional_term_y.append(- z_vals[i] * der_fitparam[2] - der_fitparam[3] ) 
            old_term_x.append(der_centroids[0,i])
            old_term_y.append(der_centroids[1,i])
                                          
            X, Y = np.meshgrid(x_term, y_term)
            derivative_matrix[i, :, : ] = X + Y 

        return derivative_matrix, np.array(old_term_x), np.array(old_term_y), np.array(additional_term_x), np.array(additional_term_y)
    
        
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
        if True: #(self._iteration > 20):
            Visualize3dImage(self.derivative.astype('float64'), slice_axis=0, symmetric_colorbar=False, title ='derivative')
            #Visualize3dImage(self._Ssum.astype('float64'), slice_axis=0, symmetric_colorbar=False, title ='_S + derivative * LAMBDA')
            #Visualize3dImage(self.new_sensitivity.astype('float64'), slice_axis=0, symmetric_colorbar=False, title ='new sensitivity')
            plt.show()
            
            
    def print_some_output(self):        
        print(f'new_sensitivity.min = {self.new_sensitivity.min()}, new_sensitivity.max = {self.new_sensitivity.max()}')        
        print(f'derivative.min = {self.derivative.min()}, derivative.max = {self.derivative.max()}')
        
