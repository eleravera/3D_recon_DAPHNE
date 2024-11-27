import numpy as np
import imageio
import os
import pickle
from Misc.DataTypes import point_dtype
from skimage.measure import block_reduce



def FindPolarAngle(v,range_deg=180.0):
    """!@brief
       Evaluate theta 
    """
    return np.rad2deg(np.arctan2(v[0],v[1])) % range_deg

def SafeDivision(A, B):
    """!@brief 
        This function handles the 0 division problem
    """
    eps = 1e-6
    if B == 0:
        return A / (eps)
    return A / B


def CheckParameters(obj,param_name):
    if not hasattr(obj, param_name):
        raise Exception("{} is not defined".format(param_name))
        
    
    
def RotatePointAlongZ(p, angle):
    """!@brief 
    Rotate a rec-array of point_dtype along Z axis
    """
    angle_rad = np.deg2rad(angle)
    shape_0 = max(p.shape[0], angle.shape[0])
    r = np.zeros(shape_0, dtype=point_dtype)
    r["x"] = np.cos(angle_rad) * p["x"] - np.sin(angle_rad) * p["y"]
    r["y"] = np.sin(angle_rad) * p["x"] + np.cos(angle_rad) * p["y"]
    r["z"] = p["z"]
    return r

def RotatePointAlongY(p, angle):
    """!@brief 
    Rotate a rec-array of point_dtype along Y axis
    """
    angle_rad = np.deg2rad(angle)
    shape_0 = max(p.shape[0], angle.shape[0])
    r = np.zeros(shape_0, dtype=point_dtype)
    r["z"] = np.cos(angle_rad) * p["z"] - np.sin(angle_rad) * p["x"]
    r["x"] = np.sin(angle_rad) * p["z"] + np.cos(angle_rad) * p["x"]
    r["y"] = p["y"]
    return r

def RotatePointAlongX(p, angle):
    """!@brief 
    #DEVO CAMBIARE Z, X, Y 
    Rotate a rec-array of point_dtype along X axis
    """
    angle_rad = np.deg2rad(angle)
    shape_0 = max(p.shape[0], angle.shape[0])
    r = np.zeros(shape_0, dtype=point_dtype)
    r["y"] = np.cos(angle_rad) * p["y"] - np.sin(angle_rad) * p["z"]
    r["z"] = np.sin(angle_rad) * p["y"] + np.cos(angle_rad) * p["z"]
    r["x"] = p["x"]
    return r   


def ReadImage(filename):
    """!@brief 
     Read a formatted image from disk using imageio module
    """
    img = imageio.imread(filename)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img





def WriteImage(img, filename):
    """!@brief 
        Write a formatted image from disk using imageio module
    """
    imageio.imwrite(filename, img)




def EvaluatePoint(p, v, size):    
    """!@brief 
    Evaluate the coordinates of a point starting using the formula p+v*size
    @param p:
    @param v:
    @param size: 
     
    """
    return p + v * size





def CheckExt(filename, ext):
    """!@brief 
        Check if a given filename has an extension
    """
    if filename.endswith(ext):
        return filename
    return filename + ext





def DownscaleImage(img,downscale_factors):
    """!@brief 
       Downscale img in each dimension and return the downlscaed image
    """
    return block_reduce(img, downscale_factors, func=np.mean)


def Pickle(obj, filename, ext):
    """!@brief 
        Serialize an object using the pickle library. 
        If the filename does not contain the extension 
        it is added prior to saving file  
    """
    # Estrai la directory dal percorso
    directory = os.path.dirname(filename)
    
    # Crea la directory se non esiste
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(CheckExt(filename, ext), "wb") as f:
        pickle.dump(obj, f)





def Unpickle(filename):
    """!@brief 
        De-serialize an object using the pickle library
    """
    with open(filename, "rb") as f:
        return pickle.load(f)



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

def OutputFileName(directory, input_file_name, d1, iterations, views_nb, fan_angle):
    output_file_name = os.path.splitext(os.path.basename(input_file_name))[0] + '_d1' + str(int(d1)) + '_i' + str(int(iterations)) + '_p' + str(int(views_nb)) + '_t' + str(int(fan_angle))
    output_file_name = directory + output_file_name
    return output_file_name

    

