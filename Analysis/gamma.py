#python -i gamma.py /home/eleonora/3D_recon_DAPHNE/Data/cylindrical_phantom.npz /home/eleonora/3D_recon_DAPHNE/Reconstruction/2024-12-21_12-43-27/

import sys
sys.path.append("../")
import argparse
import ast
import os
import numpy as np
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from Misc.Preview import Visualize3dImage, Visualize3dImageWithProfile, Visualize3dImageWithProfileAndHistogram
import Analysis.AnalyisiFunctions as Analysis



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elabora i dati di ricostruzione e salva le statistiche.")
    parser.add_argument('input_file', type=str, help="Percorso del file di input contenente la matrice originale (ad esempio, 'cylindrical_phantom.npz').")
    parser.add_argument('recon_dir', type=str, help="Directory contenente i file di ricostruzione.")
    parser.add_argument('--sigma', type=int, default=None, help="Sigma for the Gaussian filter (must be an integer).")
    args = parser.parse_args()    

    input_matrix = np.load(args.input_file)['doseDistr'].astype(np.float64)

    #NOn so se va bene questo if. 
    if args.sigma is not None: 
        input_matrix = gaussian_filter(input_matrix, sigma = args.sigma/6, order = 0)

    latest_recon_file = os.path.join(args.recon_dir, 'reco.npz') if os.path.isfile(os.path.join(args.recon_dir, 'reco.npz')) else None
    reconstructed_matrix = Analysis.load_highest_iteration_matrix(latest_recon_file).astype(np.float64)
        
    gamma_file = os.path.join(args.recon_dir, 'gamma_analysis.npz') if os.path.isfile(os.path.join(args.recon_dir, 'gamma_analysis.npz')) else None
    
    gamma_matrices= []
    
    fig_gamma, ax_gamma = plt.subplots(1, 1)
    ax_gamma.set(xlabel = "g", ylabel= 'passing rate', title = 'passing rate')

    
    if gamma_file is not None:
        try:
            data = np.load(gamma_file)
            for key in data.keys():
                gamma_matrix = data[key]
                gamma_matrix[gamma_matrix == np.inf] = -1#np.nan
                gamma_matrices.append(gamma_matrix)
                dose_threshold = ast.literal_eval(key)[2]
                

                Visualize3dImageWithProfileAndHistogram(
                    gamma_matrix, slice_axis=0, profile_axis=0, title=f"Gamma - {key}",
                    log_scale_hist = True, symmetric_colorbar= False)
                
                max_dose = np.max(input_matrix)
                dose_threshold *= max_dose
                mask_reference = input_matrix >= dose_threshold
                mask_verification = reconstructed_matrix >= dose_threshold
                valid_mask = mask_reference & mask_verification  # Maschera per punti validi


                gamma_hist = np.histogram(np.concatenate(np.concatenate(gamma_matrix)), bins = np.linspace(0, int(np.nanmax(gamma_matrix))+1, 2*(int(np.nanmax(gamma_matrix))+1)+1))
                passing_rate_hist = passing_rate = np.cumsum(gamma_hist[0])/np.sum(valid_mask)*100
                midpoints = (gamma_hist[1][:-1] + gamma_hist[1][1:]) / 2
                
                ax_gamma.plot(midpoints, passing_rate_hist, '-o', label = key)
                
                passing_rate = np.sum((gamma_matrix > 0)*(gamma_matrix <= 1)) / np.sum(valid_mask) * 100    
                pixel_with_signal = np.sum(valid_mask)*100/100**3 
                print(f'Criteria {key} give a passing rate: {passing_rate:.2f}% -- pixel with signal {pixel_with_signal:.2f}%')
        except Exception as e:
            print(f"Errore durante il caricamento o l'elaborazione di '{gamma_file}': {e}")
    else:
        print("Il file 'gamma_analysis.npz' non è stato trovato.")
        
        
    ax_gamma.legend()    
        
    plt.show()
        
