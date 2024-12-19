from matplotlib import pyplot as plt
import numpy as np
import os
from datetime import datetime
import plotly.graph_objects as go
import sys


class SaveOutput:
    def __init__(self, basePath='../Reconstruction/', logFilePath=None):
        # Crea una directory dinamica all'inizializzazione
        self.outputDir = self.createOutputDirectory(basePath)
        self.outputFile = os.path.join(self.outputDir, "logging.txt")
        # Configura il file di log
        self.logFilePath = logFilePath or os.path.join(self.outputDir, "log.txt")
        # Reindirizza stdout
        sys.stdout = self.Tee(self.outputFile)

    def createOutputDirectory(self, basePath):
        """Crea una cartella dinamica basata su data e ora."""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        outputDirectory = os.path.join(basePath, timestamp)
        os.makedirs(outputDirectory, exist_ok=True)
        return outputDirectory


    def log_execution(self, input_file, views, pixels, pitch, iterations):
        """Scrive informazioni sul log di esecuzione."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_entry = (
            f"{timestamp}\t\t{input_file}\t\t{views}\t\t"
            f"{pixels}\t\t{pitch:.2f}\t\t{iterations}\n"
        )
        with open(self.logFilePath, "a") as log_file:
            log_file.write(log_entry)
            
            
    def PlotThreeSlices(self, inputImage, slices=(50, 50, 50), title="", savefig=False):   
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        im0 = ax[0].imshow(inputImage[slices[0], :, :], cmap="viridis")
        im1 = ax[1].imshow(inputImage[:, slices[1], :], cmap="viridis")
        im2 = ax[2].imshow(inputImage[:, :, slices[2]], cmap="viridis")
        
        ax[0].set_title(f"Slice {slices[0]}")
        ax[1].set_title(f"Slice {slices[1]}")
        ax[2].set_title(f"Slice {slices[2]}")
        
        cbar = fig.colorbar(im2, ax=ax, location='bottom', shrink=0.8)
        cbar.set_label("Intensity")
        fig.suptitle(title, fontsize=16)

        if savefig:
            i = 1
            while os.path.exists(f"{self.outputDir}/figure{i}.png"):
                i += 1
            fig.savefig(f"{self.outputDir}/figure{i}.png")    
        return


    def PlotProjections(self, sinogramList, savefig=False):
        max_subfigures = 9
        
        len_sino1 = sinogramList[1]._data.shape[1]
        
        if len_sino1 <= 2:
            rows, cols = 3, 1
        elif len_sino1 <= 4:
            rows, cols = 3, 1
        elif len_sino1 <= 6:
            rows, cols = 3, 2
        else:
            rows, cols = 3, 3
        
        num_plots = rows * cols
        
        images_to_plot = [sinogramList[0]._data[:, 0, ::-1]]
        
        if len_sino1 > 0:
            step = max(1, len_sino1 // (num_plots - 1))  # Determina il passo per selezionare le immagini
            selected_indices = list(range(0, len_sino1, step))[:num_plots - 1]
            images_to_plot += [sinogramList[1]._data[:, i, ::-1] for i in selected_indices]
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
        axes = axes.flatten()
      
        for i, img in enumerate(images_to_plot):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Projection {i + 1}")
        
        for j in range(len(images_to_plot), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        if savefig:
            i = 1
            while os.path.exists(f"{self.outputDir}/figure{i}.png"):
                i += 1
            fig.savefig(f"{self.outputDir}/figure{i}.png") 


    class Tee:
        """Classe per duplicare l'output su terminale e file."""
        def __init__(self, file_name, mode="w"):
            self.file = open(file_name, mode)
            self.stdout = sys.stdout  # Mantiene un riferimento a stdout originale

        def write(self, message):
            self.file.write(message)  # Scrive su file
            self.stdout.write(message)  # Scrive su terminale

        def flush(self):
            self.file.flush()
            self.stdout.flush()

    def close(self):
        """Chiude il file e ripristina stdout."""
        sys.stdout.file.close()
        






#####
    
       

def SaveMatrix(matrix, filename, key="data"):
    """
    Salva una matrice in un file .npz con una chiave specificata.
    - Crea automaticamente le directory mancanti nel percorso.
    
    Args:
        matrix (np.ndarray): La matrice da salvare.
        filename (str): Nome del file (con o senza estensione).
        key (str): La chiave con cui la matrice sarà salvata nel file .npz.
    """
    # Assicura che il file abbia l'estensione .npz
    if not filename.endswith(".npz"):
        filename += ".npz"
    
    # Crea la directory se non esiste
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # Controlla se il file esiste già
    if os.path.exists(filename):
        # Carica i dati esistenti
        existing_data = np.load(filename)
        # Crea un nuovo dizionario con i dati esistenti più la nuova matrice
        new_data = {key: matrix}
        new_data.update(existing_data)  # Unisce i dati esistenti con la nuova matrice
        
        # Salva tutto nel file .npz
        np.savez(filename, **new_data)
    else:
        # Se il file non esiste, salva normalmente la matrice
        np.savez(filename, **{key: matrix})

    
    
    

   
