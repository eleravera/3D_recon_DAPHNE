#python -i gamma.py /home/eleonora/3D_recon_DAPHNE/Data/cylindrical_phantom.npz /home/eleonora/3D_recon_DAPHNE/Reconstruction/2024-12-21_12-43-27/

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


    normalized_input = Analysis.normalize_matrix(input_matrix, per_slice=False)
    normalized_reconstructed = Analysis.normalize_matrix(reconstructed_matrix, per_slice=False)

        
    """    
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

    ax_gamma.legend()"""


    #######################################################################



    max_dose_ref = np.max(normalized_input)
    max_dose_eval = np.max(normalized_reconstructed)
    if not np.isclose(max_dose_ref, max_dose_eval, rtol=0.1):
        print("Attenzione: il massimo della dose di riferimento e di verifica differiscono significativamente.")
    
    
    
    keys = ["[1.0, 1.0, 0.0]", "[2.0, 2.0, 0.0]", "[3.0, 3.0, 0.0]", "[1.0, 3.0, 0.0]", "[3.0, 1.0, 0.0]", "[1.0, 2.0, 0.0]", "[2.0, 1.0, 0.0]", "[3.0, 2.0, 0.0]", "[2.0, 3.0, 0.0]"]
    dose_thresholds = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]


    gamma_file = os.path.join(args.recon_dir, 'gamma_analysis_new_dose_criteria.npz') if os.path.isfile(os.path.join(args.recon_dir, 'gamma_analysis.npz')) else None
    data = np.load(gamma_file)


    dose_criteria = []
    space_criteria = []
    thresholds = []
    prate_ref = []
    prate_val = []
    pvoxel_ref = []
    pvoxel_val = []
    dose_cont = []
    space_cont = []
    
    
    for k in keys:
        print(f'\n*********')
        
        gamma_matrix = data[k]
        gamma_matrix[gamma_matrix == np.inf] = -1#np.nan  # -1

        for d in dose_thresholds:
            threshold_ref = d * max_dose_ref
            threshold_eval = d * max_dose_eval       
            mask_reference = normalized_input >= threshold_ref
            mask_evaluated = normalized_reconstructed >= threshold_eval

                    
            passing_rate_thref = np.sum((gamma_matrix >= 0) * (gamma_matrix <= 1) * mask_reference) / np.sum(mask_reference) * 100    
            pixel_with_signal_thref = np.sum(mask_reference) * 100 / 100 ** 3 
            passing_rate_theval = np.sum((gamma_matrix >= 0) * (gamma_matrix <= 1) * mask_evaluated) / np.sum(mask_evaluated) * 100    
            pixel_with_signal_theval = np.sum(mask_evaluated) * 100 / 100 ** 3 

            print(f'\nCriteria {k} - threshold {d*100:.1f}% on reference give a passing rate: {passing_rate_thref:.2f}% -- pixel with signal {pixel_with_signal_thref:.2f}%')
            print(f'Criteria {k} - threshold {d*100:.1f}% on evaluated give a passing rate: {passing_rate_theval:.2f}% -- pixel with signal {pixel_with_signal_theval:.2f}%')
                    
            space_contribution = data[f"space_contribution_{k}"]
            space_contribution[space_contribution == np.inf] = -1
            dose_contribution = data[f"dose_contribution_{k}"]
            dose_contribution[dose_contribution == np.inf] = -1

            mean_dose_contribution = np.mean(dose_contribution[(dose_contribution != -1) * mask_reference])                   
            mean_spatial_contribution = np.mean(space_contribution[(space_contribution != -1) * mask_reference])
            print(f'The mean dose contribution is {mean_dose_contribution:.2f}, ' 
                          f'and the mean spatial contribution is {mean_spatial_contribution:.2f}')



            dose_criteria.append(ast.literal_eval(k)[0])
            space_criteria.append(ast.literal_eval(k)[1])
            thresholds.append(d)
            prate_ref.append(passing_rate_thref)
            prate_val.append(passing_rate_theval)
            pvoxel_ref.append(pixel_with_signal_thref)
            pvoxel_val.append(pixel_with_signal_theval)
            dose_cont.append(mean_dose_contribution)
            space_cont.append(mean_spatial_contribution)
            
            
    dose_criteria, space_criteria, thresholds, prate_ref, prate_val, pvoxel_ref, pvoxel_val, dose_cont, space_cont = np.array(dose_criteria), np.array(space_criteria), np.array(thresholds), np.array(prate_ref), np.array(prate_val), np.array(pvoxel_ref), np.array(pvoxel_val), np.array(dose_cont), np.array(space_cont)
            
            
    #VOXEL COUNTED VS THRESHOLD    
    plt.figure()
    plt.plot(thresholds, pvoxel_ref, 'o', label='pvoxel ref')
    plt.plot(thresholds, pvoxel_val, 'o', label='pvoxel val')
    plt.xlabel(" threshold")
    plt.ylabel("%")
    plt.grid()
    plt.legend()
    
    
    unique_pairs = set(zip(dose_criteria, space_criteria))


    #PASSING RATE VS THRESHOLD    
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))  # 3 righe, 1 colonna
    for dose, space in unique_pairs:
        indices = [i for i, (d, s) in enumerate(zip(dose_criteria, space_criteria)) if d == dose and s == space]
        filtered_thresholds = [thresholds[i] for i in indices]
        filtered_prate_ref = [prate_ref[i] for i in indices]
        filtered_prate_val = [prate_val[i] for i in indices]
        difference = np.array(filtered_prate_ref) - np.array(filtered_prate_val)

        axes[0].plot(filtered_thresholds, filtered_prate_ref, 'o--', label=f"Dose: {dose}, Space: {space}")
        axes[1].plot(filtered_thresholds, filtered_prate_val, 'o--', label=f"Dose: {dose}, Space: {space}")
        axes[2].plot(filtered_thresholds, difference, 'o--', label=f"Dose: {dose}, Space: {space}")

    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Passing Rate ref")
    axes[0].set_title("Passing Rate ref vs Threshold")
    axes[0].legend()
    axes[0].grid()

    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Passing Rate val")
    axes[1].set_title("Passing Rate val vs Threshold")
    axes[1].legend()
    axes[1].grid()

    axes[2].set_xlabel("Threshold")
    axes[2].set_ylabel("Difference")
    axes[2].set_title("Difference (ref - val) vs Threshold")
    axes[2].legend()
    axes[2].grid()

    plt.tight_layout()
    
    
    #SPATIAL AND DOSE CONTRIBUTIONS VS THRESHOLD
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))  # 3 righe, 1 colonna
    for dose, space in unique_pairs:
        indices = [i for i, (d, s) in enumerate(zip(dose_criteria, space_criteria)) if d == dose and s == space]
        filtered_thresholds = [thresholds[i] for i in indices]
        spatial_contribution = [space_cont[i] for i in indices]
        dose_contribution = [dose_cont[i] for i in indices]

        axes[0].plot(filtered_thresholds, spatial_contribution, 'o--', label=f"Dose: {dose}, Space: {space}")
        axes[1].plot(filtered_thresholds, dose_contribution, 'o--', label=f"Dose: {dose}, Space: {space}")

    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Spatial contribution")
    axes[0].set_title("")
    axes[0].legend()
    axes[0].grid()

    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Dose contribution")
    axes[1].set_title("")
    axes[1].legend()
    axes[1].grid()


    plt.tight_layout()    
        
        
    mask = thresholds == 0.1
    plt.figure()
    plt.plot(dose_criteria[mask], prate_ref[mask], 'o')
    
    plt.figure()
    plt.plot(space_criteria[mask], prate_ref[mask], 'o')
    
    
    
    
    
    
    
    # Creazione della matrice vuota 3x3
    prate_matrix = np.full((3, 3), np.nan)  # Inizializza una matrice 3x3 con NaN

    # Mappatura per associare dose e space ai rispettivi indici
    dose_mapping = {1.0: 0, 2.0: 1, 3.0: 2}
    space_mapping = {1.0: 0, 2.0: 1, 3.0: 2}

    # Riempimento della matrice con i valori di prate_ref
    for dose, space, prate in zip(dose_criteria, space_criteria, prate_ref):
        dose_idx = dose_mapping[dose]
        space_idx = space_mapping[space]
        prate_matrix[dose_idx, space_idx] = prate

    # Creazione del grafico
    plt.figure(figsize=(6, 6))

    # Uso di imshow per visualizzare la matrice come un'immagine
    # Usando 'coolwarm' per la colorazione e NaN verranno lasciati vuoti
    cax = plt.imshow(prate_matrix, interpolation='nearest', aspect='auto')

    # Aggiungi una colorbar
    plt.colorbar(cax, label='Passing Rate')

    # Etichette degli assi
    plt.xticks(np.arange(3), ['1', '2', '3'])
    plt.yticks(np.arange(3), ['1', '2', '3'])

    # Etichette degli assi
    plt.xlabel('Space Criteria')
    plt.ylabel('Dose Criteria')

    # Titolo
    plt.title('2D Histogram of Passing Rate')
        
    
    plt.show()
        
