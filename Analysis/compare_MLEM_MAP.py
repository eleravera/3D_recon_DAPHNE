import sys
import os
import argparse
sys.path.append("../")
from matplotlib import pyplot as plt
import numpy as np
from Misc.Preview import Visualize3dImage, Visualize3dImageWithProfile, Visualize3dImageWithProfileAndHistogram
import Analysis.AnalyisiFunctions as Analysis


def calculate_relative_diff(directory, input_matrix, mask):
    relative_diffs = []
    dev_relative_diffs = []
    iterations = []
                
    reco_data = np.load(directory + '/reco.npz')  # Carica il file .npz
    for filename in reco_data.files:  # Usa le chiavi del file .npz
        recon_matrix = open_and_normalize(directory + '/reco.npz', filename)  # Estrarre la matrice associata alla chiave
        
                
        relative_diff = (input_matrix - recon_matrix) / (input_matrix + recon_matrix) * 2
        mean_relative_diff = np.mean(relative_diff[mask])        
        relative_diffs.append(mean_relative_diff)
        dev_relative_diffs.append(np.std(relative_diff[mask]))
        iterations.append(filename)

    return relative_diffs, dev_relative_diffs, iterations


def open_and_normalize(path, key): 
    data = np.load(path)
    matrix = data[key]
    normalize_matrix = Analysis.normalize_matrix(matrix)
    return normalize_matrix 


if __name__ == "__main__":
    INPUT_FILE = '../Data/traccia.npz'
    SELECTED_ITER = 'iter49'
    
   
    
    parser = argparse.ArgumentParser(description="Compara MLEM e MAP")
    parser.add_argument('--MLEM', type=str, default=None, help="Directory contenente i file di ricostruzione con MLEM.")
    parser.add_argument('--MAP', type=str, default=None, help="Directory contenente i file di ricostruzione con MAP.")
    parser.add_argument('--lamda', type=float, default=1.e+4, help="Valore di lambda per la ricostruzione.")
    args = parser.parse_args()

    
    #if args.MLEM or args.MAP is None:
    #    exit. 
    
    
    #OPEN THE INPUT MATRIX
    input_matrix = open_and_normalize(INPUT_FILE, 'doseDistr')    
    #set a threshold to ...
    max_input = np.max(input_matrix)
    threshold_mask = input_matrix > 0.03 * max_input
    
   
    recon_matrix_MLEM = open_and_normalize(args.MLEM+'/reco.npz', SELECTED_ITER)
    recon_matrix_MAP = open_and_normalize(args.MAP+'/reco.npz', SELECTED_ITER)


    plt.figure()
    max_bin = max(input_matrix.max(), recon_matrix_MAP.max(), recon_matrix_MLEM.max())
    bins = np.linspace(0, max_bin, 100)
    plt.hist(np.concatenate(np.concatenate(input_matrix)), bins = bins, histtype='step', label='Input')
    plt.hist(np.concatenate(np.concatenate(recon_matrix_MLEM)), bins = bins, histtype='step', label='MLEM')
    plt.hist(np.concatenate(np.concatenate(recon_matrix_MAP)), bins = bins, histtype='step', label='MAP')    
    plt.xlabel('Signal intensity [a.u.]')
    plt.yscale('log')
    plt.title(f'$\lambda$ = {args.lamda:2e}')
    plt.legend()  
    #plt.savefig(f"/home/eleonora/3D_recon_DAPHNE/Reconstruction/prior/histo_{int(args.lamda):d}.png", dpi=300, bbox_inches="tight")




    rel_diff_MLEM, std_MLEM, it_MLEM = calculate_relative_diff(args.MLEM, input_matrix, threshold_mask)
    rel_diff_MAP, std_MAP, it_MAP = calculate_relative_diff(args.MAP, input_matrix, threshold_mask)
        

    #RELATIVE DIFFERENCE, SIGMA AND MEAN
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    axes[0].plot(np.flip(rel_diff_MLEM), 'o--', label='MLEM')
    axes[0].plot(np.flip(rel_diff_MAP), 'o--', label='MAP')
    axes[0].set_title(f'Mean, $\lambda$ = {args.lamda}')
    axes[0].legend()
    axes[0].grid()
    axes[1].plot(np.flip(std_MLEM), 'o--', label='MLEM')
    axes[1].plot(np.flip(std_MAP), 'o--', label='MAP')
    axes[1].set_title(f'St. dev., $\lambda$ = {args.lamda:2e}')
    axes[1].legend()
    axes[1].grid()
    plt.tight_layout()
    #plt.savefig(f"/home/eleonora/3D_recon_DAPHNE/Reconstruction/prior/rel_diff_{int(args.lamda):d}.png", dpi=300, bbox_inches="tight")


    d = np.abs(np.flip(rel_diff_MLEM)-np.flip(rel_diff_MAP))
    index = np.argwhere(d == d.max())
    diff_percent_on_mean = np.max(d[20:])/np.mean(np.flip(rel_diff_MLEM)[20:])*100
    

    d = np.abs(np.flip(std_MLEM)-np.flip(std_MAP))
    index1 = np.argwhere(d == d.max())
    diff_percent_on_std = np.max(d[20:])/np.mean(np.flip(std_MLEM)[20:])*100

    print(f'\n\ndiff_percent_on_mean = {diff_percent_on_mean} and diff_percent_on_std = {diff_percent_on_std}, @ index = {index}, {index1}\n\n')

    diff = recon_matrix_MLEM - recon_matrix_MAP
    Visualize3dImageWithProfileAndHistogram(diff, slice_axis=0, symmetric_colorbar=True, title = 'MLEM-MAP')
        
    rel_diff_MLEM = np.where(threshold_mask, (input_matrix - recon_matrix_MLEM) / (input_matrix + recon_matrix_MLEM) * 200, np.nan)
    
    rel_diff_MAP = np.where(threshold_mask, (input_matrix - recon_matrix_MAP) / (input_matrix + recon_matrix_MAP) * 200, np.nan)

    #Visualize3dImageWithProfileAndHistogram(recon_matrix_MLEM.astype('float64'), slice_axis=0, symmetric_colorbar=False, title ='MLEM')
    #Visualize3dImageWithProfileAndHistogram(rel_diff.astype('float64'), slice_axis=0, symmetric_colorbar=True, title ='rel_diff MLEM %')
    #Visualize3dImageWithProfileAndHistogram(rel_diff.astype('float64'), slice_axis=0, symmetric_colorbar=True, title ='rel_diff MAP %')
    

    plt.figure()
    max_bin = max(np.nanmax(rel_diff_MAP), np.nanmax(rel_diff_MLEM))  
    bins = np.linspace(0, max_bin, 100)
    plt.hist(np.concatenate(np.concatenate(rel_diff_MLEM)), bins = bins, histtype='step', label='MLEM')
    plt.hist(np.concatenate(np.concatenate(rel_diff_MAP)), bins = bins, histtype='step', label='MAP')    
    plt.xlabel('Signal intensity [a.u.]')
    plt.yscale('log')
    plt.title(f'Relative difference $\lambda$ = {args.lamda:2e}')
    plt.legend()   
            
    """
        Visualize3dImageWithProfileAndHistogram(diff, slice_axis=0, symmetric_colorbar=True)
        Visualize3dImageWithProfileAndHistogram(input_matrix-recon_matrix_MLEM, slice_axis=0, symmetric_colorbar=True, title ='input-MLEM')
        Visualize3dImageWithProfileAndHistogram(input_matrix-recon_matrix, slice_axis=0, symmetric_colorbar=True, title ='input-MAP')
        Visualize3dImageWithProfileAndHistogram(recon_matrix.astype('float64'), slice_axis=0, symmetric_colorbar=False, title ='MAP')
    """

    plt.show()
