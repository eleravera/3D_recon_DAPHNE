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

do_plot = False

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
    
    fig_gamma, ax_gamma = plt.subplots(1, 1)
    ax_gamma.set(xlabel="g", ylabel='passing rate', title='passing rate')

    gamma_matrices = []

    if gamma_file is not None:
        try:
            data = np.load(gamma_file)
            
            filtered_keys = []
            for key in data.keys():
                try:             
                    value = ast.literal_eval(key)  # Converte la stringa in un oggetto Python
                    if isinstance(value, list) and len(value) == 3 and all(isinstance(i, (int, float)) for i in value):
                        filtered_keys.append(key)
                except (ValueError, SyntaxError):  # Se la conversione fallisce, significa che la chiave non è nel formato desiderato
                    pass

        
            spatial_contributions = []
            dose_contributions = []
            for key in filtered_keys:               
                gamma_matrix = data[key]
                gamma_matrix[gamma_matrix == np.inf] = -1  # np.nan
                gamma_matrices.append(gamma_matrix)
                dose_threshold = ast.literal_eval(key)[2]
                
                
                if  key == "[1.0, 1.0, 0.005]":
                    Visualize3dImageWithProfileAndHistogram(
                        gamma_matrix, slice_axis=0, profile_axis=0, title=f"Gamma - {key}",
                        log_scale_hist=True, symmetric_colorbar=False)
                
                max_dose = np.max(reconstructed_matrix)
                dose_threshold = dose_threshold * max_dose
                mask_reference = input_matrix >= dose_threshold
                mask_verification = reconstructed_matrix >= dose_threshold
                valid_mask = mask_verification
                
                gamma_hist = np.histogram(np.concatenate(np.concatenate(gamma_matrix)), 
                                          bins=np.linspace(0, int(np.nanmax(gamma_matrix)) + 1, 
                                                           2 * (int(np.nanmax(gamma_matrix)) + 1) + 1))
                passing_rate_hist = np.cumsum(gamma_hist[0]) / np.sum(valid_mask) * 100
                midpoints = (gamma_hist[1][:-1] + gamma_hist[1][1:]) / 2
                
                #if ast.literal_eval(key)[2] == 0.1:
                ax_gamma.plot(midpoints, passing_rate_hist, '-o', label=key)
                
                passing_rate = np.sum((gamma_matrix >= 0) * (gamma_matrix <= 1)) / np.sum(valid_mask) * 100    
                pixel_with_signal = np.sum(valid_mask) * 100 / 100 ** 3 
                print(f'\nCriteria {key} give a passing rate: {passing_rate:.2f}% -- pixel with signal {pixel_with_signal:.2f}%')
                print('dose_threshold in this case: ', dose_threshold, ast.literal_eval(key)[2])
                
                # Seconda parte del codice nel loop
                try:
                    space_contribution = data[f"space_contribution_{key}"]
                    space_contribution[space_contribution == np.inf] = -1
                    dose_contribution = data[f"dose_contribution_{key}"]
                    dose_contribution[dose_contribution == np.inf] = -1
                    
                    if (do_plot == True) and (key == "[1.0, 1.0, 0.]"):

                        Visualize3dImageWithProfileAndHistogram(
                            space_contribution, slice_axis=0, profile_axis=0, title=f"space_contribution - {key}",
                            log_scale_hist=True, symmetric_colorbar=False)
                        
                        Visualize3dImageWithProfileAndHistogram(
                            dose_contribution, slice_axis=0, profile_axis=0, title=f"dose_contribution - {key}",
                        log_scale_hist=True, symmetric_colorbar=False)
                    
                        plt.figure()
                        _ = plt.hist(dose_contribution[dose_contribution != -1], bins=100)
                        plt.xlabel(f'dose contribution {key}')
                        
                        plt.figure()
                        _ = plt.hist(space_contribution[space_contribution != -1], bins=10)
                        plt.xlabel(f'spatial contribution {key}')

                    mean_dose_contribution = np.mean(dose_contribution[dose_contribution != -1])                   
                    mean_spatial_contribution = np.mean(space_contribution[space_contribution != -1])
                    spatial_contributions.append(mean_spatial_contribution)
                    dose_contributions.append(mean_dose_contribution)
                    print(f'The mean dose contribution is {mean_dose_contribution:.2f}, ' 
                          f'and the mean spatial contribution is {mean_spatial_contribution:.2f}')
                except KeyError as e:
                    print(f"KeyError: {e} not found in the dataset.")

        except Exception as e:
            print(f"Errore durante il caricamento o l'elaborazione di '{gamma_file}': {e}")
    else:
        print("Il file 'gamma_analysis.npz' non è stato trovato.")

    ax_gamma.legend()




    key = "[1.0, 1.0, 0.0]"
    gamma_matrixa_all = data[key]
    gamma_matrixa_all[gamma_matrixa_all == np.inf] = -1

    mask_all = (gamma_matrixa_all >= 0) * (gamma_matrixa_all <= 1)
    passing_rate = np.sum(mask_all) / 10000  
                
    print('\n\npassing rate: ', passing_rate)

    
    key = "[1.0, 1.0, 0.005]"
    gamma_matrixa_1 = data[key]    
    gamma_matrixa_1[gamma_matrixa_1 == np.inf] = -1    
    
    max_dose = 0.005 * np.max(reconstructed_matrix)
    mask_verification = reconstructed_matrix >= max_dose
    
    mask_1 = (gamma_matrixa_1 >= 0) * (gamma_matrixa_1 <= 1)
    passing_rate = np.sum(mask_all) / np.sum(mask_verification)*100
    
    print('\n\npassing rate: ', passing_rate)   
    
    
        
    plt.show()
        
