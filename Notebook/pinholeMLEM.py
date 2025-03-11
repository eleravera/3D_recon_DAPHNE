#python3 -i pinholeMLEM.py --input_file ../Data/cylindrical_phantom.npz --iterations 10 --long_views 16 --pixel_nb 200 --pixel_pitch 0.5 --do_plot 

import sys
sys.path.append("../") # this to be able to include all the object contained in the modules
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from Misc.Utils import Unpickle,ReadImage,DownscaleImage
from Misc.Preview import Visualize3dImage
import Misc.OutputDir as saveOutput
from Algorithms.SinogramGenerator_3D_long_transv import SinogramGenerator_3D
from Geometry.ExperimentalSetupCT_3D_long_transv import ExperimentalSetupCT_3D, Mode, DetectorShape
from Algorithms.MLEM import MLEM
from Algorithms.MAP import MAP
from Misc.DataTypes import voxel_dtype



#input_file_name = '/home/eleonora/eFLASH_3D_Sim-build/Dose_Map_30mm/spettro_9MeV/doseDistribution.npz'
#SET THE INFORMATION
#INPUT_FILE = '../Data/cylindrical_phantom.npz' # questo è quello transpose(input_img, (2, 1, 0)) rispetto a cilinder.npz

parser = argparse.ArgumentParser(description="Set experimental setup parameters and reconstruction settings.")
parser.add_argument('--input_file', type=str, required=True, help="Path to the input .npz file.")
parser.add_argument('--iterations', type=int, default=50, help="Number of iterations for the reconstruction algorithm.")
parser.add_argument('--long_views', type=int, default=8, help="Number of longituinal views.")
parser.add_argument('--pixel_nb', type=int, default=100, help="Number of pixels per slice.")
parser.add_argument('--pixel_pitch', type=float, default=1.0, help="Pixel pitch in mm.")
parser.add_argument('--do_plot', action='store_true', help="Enable plotting of the results.")

args = parser.parse_args()
INPUT_FILE = args.input_file
ITERATIONS = args.iterations
VIEWS = args.long_views
PIXEL_NB = args.pixel_nb
PIXEL_PITCH = args.pixel_pitch
DO_PLOT = args.do_plot

output_manager = saveOutput.SaveOutput(basePath="../Reconstruction/", logFilePath="../Reconstruction/log.txt")
output_manager.log_execution(INPUT_FILE, VIEWS, PIXEL_NB, PIXEL_PITCH, ITERATIONS)

#---------------------------------------------------------------------------

print('\nDETECTOR INFO: ')
# create a CT experimental setup
my_experimental_setup = ExperimentalSetupCT_3D()
my_experimental_setup.mode = Mode.CONEBEAM
my_experimental_setup._detector_number = 2 #one longitudinal and one transversal

# Detector 
my_experimental_setup.pixels_per_slice_nb=PIXEL_NB
my_experimental_setup.detector_slice_nb=PIXEL_NB
my_experimental_setup.pixel_size=PIXEL_PITCH
my_experimental_setup.slice_pitch_mm=my_experimental_setup.pixel_size
my_experimental_setup.detector_shape=DetectorShape.PLANAR
 
fov_side_mm = 100 # fov size in mm
my_experimental_setup.image_matrix_size_mm = np.array([fov_side_mm,fov_side_mm,fov_side_mm])
my_experimental_setup.voxel_size_mm = np.array([1,1,1]) # fov size in mm
my_experimental_setup.gantry_angles_nb = VIEWS # number of rotation of the gantry
my_experimental_setup.angular_range_deg = 360 # range of the rotation

#
k = 0
my_experimental_setup.detector_size = my_experimental_setup.pixels_per_slice_nb * my_experimental_setup.slice_pitch_mm#mm
sad = fov_side_mm*(fov_side_mm+np.sqrt(2)*k) / (my_experimental_setup.detector_size-np.sqrt(2)*fov_side_mm)
sdd = sad + fov_side_mm/np.sqrt(2) + k
theta = 2 * np.degrees(np.arctan(my_experimental_setup.detector_size / 2 / sdd))
print('det size: ', my_experimental_setup.detector_size)

# sources 
my_experimental_setup.sdd_mm=sdd
my_experimental_setup.sad_mm=sad
my_experimental_setup.fan_angle_deg=theta

# compute the geometry
my_experimental_setup.Update()
print(my_experimental_setup.GetInfo())


#info about inputfile
print('\nINPUT FILE INFO:')
print('Input file: %s' %INPUT_FILE)
input_img = np.load(INPUT_FILE)['doseDistr'].astype(np.float64)
print('Input data shape: ', input_img.shape)
output_manager.PlotThreeSlices(input_img, savefig=True)

print('\nPROJECTIONS:')
s=SinogramGenerator_3D(my_experimental_setup)
sino_list=s.GenerateObjectSinogram(input_img,transponse_image=0)
np.savez(output_manager.outputDir+'/projections.npz', projection_t=sino_list[0]._data, projection_l=sino_list[1]._data)
output_manager.PlotProjections(sino_list, savefig=True)

#Calculate the projection arrays:
projections_t = np.concatenate(np.concatenate(np.transpose(sino_list[0]._data, axes=(2,1,0)), axis=1))
projections_l = np.concatenate(np.concatenate(np.transpose(sino_list[1]._data, axes=(2,1,0)), axis=1))
projections = np.concatenate((projections_t, projections_l))
print('projections_t.shape, projectionsl.shape, projections.shape:, ', projections_t.shape, projections_l.shape, projections.shape)

print('\n')
algorithm="MLEM"
niter=ITERATIONS
initial_value=1

it = eval( algorithm+ "()")
it.SetExperimentalSetup(my_experimental_setup)
it.SetNumberOfIterations(niter)
it.SetProjectionData(projections)
initial_guess=np.full(it.GetNumberOfVoxels(),initial_value, dtype=voxel_dtype) 
it.SetImageGuess(initial_guess)
it.SetOutputBaseName(output_manager.outputDir+'/reco.npz') 
output_img = it.Reconstruct()

output_manager.PlotThreeSlices(output_img.astype('float64'), savefig=True)


if DO_PLOT == True:
    Visualize3dImage(input_img.astype(np.float64), slice_axis=0,_cmap='gray') #This figure is not saved 
    Visualize3dImage(output_img.astype(np.float64), slice_axis=0,_cmap='gray') #This figure is not saved 
    plt.show()

#output_manager.close()

