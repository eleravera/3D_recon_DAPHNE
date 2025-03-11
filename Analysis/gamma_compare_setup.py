import sys
sys.path.append("../")
import argparse
import ast
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from Misc.Preview import Visualize3dImage, Visualize3dImageWithProfile, Visualize3dImageWithProfileAndHistogram
import Analysis.AnalyisiFunctions as Analysis
import Analysis.compare_setup as compare_setup

if __name__ == "__main__":
    
    
    input_matrix = np.load('/home/eleonora/3D_recon_DAPHNE/Data/doseDistribution_flip.npz')['doseDistr'].astype(np.float64)
    normalized_input = Analysis.normalize_matrix(input_matrix, per_slice=False)
    max_dose_ref = np.max(normalized_input)
    threshold_ref = 0.1 * max_dose_ref
    mask_reference = normalized_input >= threshold_ref 
    
    log_file = '/home/eleonora/3D_recon_DAPHNE/Reconstruction/log.txt'
    iterations, pitch, views, pixels, input_files, folder_path = compare_setup.read_log_file(log_file)


    criteria = '[3.0, 3.0, 0.0]'
    directories = ['2025-01-29_12-12-05', '2025-01-29_20-13-02', '2025-01-29_23-36-28', '2025-01-30_00-25-25', '2025-01-30_02-53-25', '2025-01-30_04-29-08']
    directories = [os.path.join(os.path.dirname(log_file), d) for d in directories]
    indices = [np.where(folder_path == d)[0][0] for d in directories]

    views = views[indices]
    pixels = pixels[indices]
    pitch = pitch[indices]
    
    passing_rate = []
    dose_contribution = [] 
    space_contribution = []
    
    for d in directories:
        
       gamma_file = os.path.join(d, 'gamma_analysis_new_dose_criteria.npz') 
       data = np.load(gamma_file)
      
       gamma_matrix = data[criteria]
       gamma_matrix[gamma_matrix == np.inf] = -1#np.nan  # -1
       pr = np.sum((gamma_matrix >= 0) * (gamma_matrix <= 1) * mask_reference) / np.sum(mask_reference) * 100 
       passing_rate.append(pr)
       
       sc = data[f"space_contribution_{criteria}"]
       sc[space_contribution == np.inf] = -1
       dc = data[f"dose_contribution_{criteria}"]
       dc[dose_contribution == np.inf] = -1
       
       mean_dc = np.mean(dc[(dc != -1) * mask_reference])                   
       mean_sc = np.mean(sc[(sc != -1) * mask_reference])
       
       dose_contribution.append(mean_dc)
       space_contribution.append(mean_sc)
       
    print(dose_contribution, space_contribution, passing_rate)   
      
