import numpy as np
import vtk
from Misc import Utils
from Misc.Utils import CheckParameters
from Geometry.ExperimentalSetup import ExperimentalSetup, DetectorType, DetectorShape
from Misc.DataTypes import proj_extrema_dtype, point_dtype,bin_dtype
from Misc.VisualizationUtils import RenderSceneJupyter, par_text_color
from Geometry.ExperimentalSetup import Mode
from Misc.VisualizationUtils import (
    CreateCube,
    CreateAutoOrientedeCube,
    CreateAxesActor,
    CreateTR,
    CreateCornerAnnotation,
    CreateLabel,
    CreateSphere,
    CreateLine
)

from Misc.VisualizationUtils import (
    par_background_color,
    par_src_color,
    par_pixel_color,
)

default_pixel_size_CT = np.array([3, 3, 3])
default_src_size_CT = 3

class ExperimentalSetupCT_3D(ExperimentalSetup):
    """!@brief Implements a setup containing a  CT detector.
        A CT detector consists of a number of sensitive elements (pixels) and one or more sources 
    """

    def __init__(self):
        self.detector_type = DetectorType.CT

    def SetMode(self, mode):
        """!@brief
        Mode can be either: parallel beam, cone beam or fan beam
        """
        if mode==1:
            self.mode = Mode.PARALLELBEAM
        elif mode==2:
            self.mode = Mode.FANBEAM
        elif mode==3:
            self.mode = Mode.CONEBEAM
        else:
            raise Exception("Mode not supported")

    def SetDetectorShape(self, detector_shape):
        """!@brief
            Geometry can be either: planar , arc.
        """
        if detector_shape==1:
            self.detector_shape = DetectorShape.PLANAR
        elif detector_shape==2:
            self.detector_shape = DetectorShape.ARC
        else:
            raise Exception("Detector shape not supported")
    
    def __Validate(self):
        """!@brief 
            Check if all the parameters needed where provided 
        """
        CheckParameters(self,"pixels_per_slice_nb")
        CheckParameters(self,"detector_slice_nb")
        CheckParameters(self,"gantry_angles_nb")
        CheckParameters(self,"image_matrix_size_mm")
        CheckParameters(self,"voxel_size_mm")
        CheckParameters(self,"angular_range_deg")
        CheckParameters(self,"slice_pitch_mm")
        if self.mode == Mode.PARALLELBEAM:
            pass
        elif  self.mode == Mode.FANBEAM or self.mode==Mode.CONEBEAM:
            CheckParameters(self,"detector_shape")
            CheckParameters(self,"sdd_mm")
            CheckParameters(self,"sad_mm")
            CheckParameters(self,"fan_angle_deg")

    
    def PlacePixels(self,detector_shape):
        """!@brief
            Create the pixel for fan and cone beam geometries
        """ 
        self._pixel_pos_mm=np.zeros(self._detector_pixel_nb * self._detector_number, dtype=point_dtype)
        
        if detector_shape==DetectorShape.PLANAR:
            # evaluate the size of the detector according to the fan size
            detector_size = 2*self.sdd_mm*np.tan(np.deg2rad(self.fan_angle_deg/2))
            print('detector_size: %.f mm    #calculated as 2*sdd*tan(fan_angle/2)' %detector_size)
            pixel_size = detector_size/self.pixels_per_slice_nb #default_pixel_size_CT[0] 
            #self.pixels_per_slice_nb = int(detector_size/pixel_size)
            #print('Updated pixels_per_slice_nb: ', self.pixels_per_slice_nb)
            pixel_slice_x_pos, self._detector_pitch_mm =np.linspace(
                                                        (-detector_size/2+pixel_size/2),
                                                        ( detector_size/2-pixel_size/2),
                                                        self.pixels_per_slice_nb,
                                                        retstep=True)
            
            
            
            pixel_slice_x_pos = np.arange(-self.pixels_per_slice_nb+1, self.pixels_per_slice_nb+1, 2)/2*self.slice_pitch_mm
            pixel_slice_y_pos, pixel_slice_z_pos = pixel_slice_x_pos, pixel_slice_x_pos

            if (detector_size*0.5 > pixel_slice_x_pos.max()): 
                print('The detector should be bigger! You can either: \n- increase the pixels_per_slice_nb \n-decrease the fan_angle_deg \n-decrease the slice_pitch_mm\n')
                #print('detector_size/2: ', detector_size*0.5)
                #print('pixel_slice_x_pos.max(): ', pixel_slice_x_pos.max())

            #self._detector_slice_z_coordinates=np.arange(-self.detector_slice_nb+1,self.detector_slice_nb+1,2)/2*self.slice_pitch_mm
            
            if self._detector_number==1:
                # create all the pixels self._source_pos_mm[x], self._source_pos_mm[y], self._source_pos_mm[z]: [0.] [0.] [0.]
                self._pixel_pos_mm['x'] = np.tile(pixel_slice_x_pos,self.detector_slice_nb)
                self._pixel_pos_mm['y'] = self.sdd_mm-self.sad_mm
                self._pixel_pos_mm['z'] = np.repeat(self._detector_slice_z_coordinates,self.pixels_per_slice_nb)
                #print('pixel_slice_x_pos: ', pixel_slice_x_pos)
            
            elif self._detector_number==2: 
                i = self._detector_pixel_nb
                # create all the pixels self._source_pos_mm[x], self._source_pos_mm[y], self._source_pos_mm[z]: [0.] [0.] [0.]
                self._pixel_pos_mm['x'][:i] = np.tile(pixel_slice_x_pos,self.detector_slice_nb)
                self._pixel_pos_mm['y'][:i] = self.sdd_mm-self.sad_mm
                self._pixel_pos_mm['z'][:i] = np.repeat(self._detector_slice_z_coordinates,self.pixels_per_slice_nb)
                
                self._pixel_pos_mm['x'][i:] = self.sdd_mm-self.sad_mm 
                self._pixel_pos_mm['y'][i:] = np.repeat(self._detector_slice_y_coordinates,self.pixels_per_slice_nb)
                self._pixel_pos_mm['z'][i:] = np.tile(pixel_slice_z_pos,self.detector_slice_nb)

            elif self._detector_number==3:
                i = self._detector_pixel_nb
                # create all the pixels self._source_pos_mm[x], self._source_pos_mm[y], self._source_pos_mm[z]: [0.] [0.] [0.]
                self._pixel_pos_mm['x'][:i] = np.tile(pixel_slice_x_pos,self.detector_slice_nb)
                self._pixel_pos_mm['y'][:i] = self.sdd_mm-self.sad_mm
                self._pixel_pos_mm['z'][:i] = np.repeat(self._detector_slice_z_coordinates,self.pixels_per_slice_nb)
                
                self._pixel_pos_mm['x'][i:2*i] = self.sdd_mm-self.sad_mm 
                self._pixel_pos_mm['y'][i:2*i] = np.repeat(self._detector_slice_y_coordinates,self.pixels_per_slice_nb)
                self._pixel_pos_mm['z'][i:2*i] = np.tile(pixel_slice_z_pos,self.detector_slice_nb)

                self._pixel_pos_mm['x'][2*i:] = np.repeat(self._detector_slice_x_coordinates,self.pixels_per_slice_nb)
                self._pixel_pos_mm['y'][2*i:] = np.tile(pixel_slice_y_pos,self.detector_slice_nb)
                self._pixel_pos_mm['z'][2*i:] = self.sdd_mm-self.sad_mm
        
        elif self.detector_shape==DetectorShape.ARC:
            angular_steps,angular_pitch=np.linspace(-self.fan_angle_deg/2,
                                       self.fan_angle_deg/2,
                                       self.pixels_per_slice_nb,
                                       retstep=True)
            pixel_slice_x_pos=self.sdd_mm*np.sin(np.deg2rad(angular_steps))
            pixel_slice_y_pos=self.sdd_mm*np.cos(np.deg2rad(angular_steps))-self.sad_mm
            self._pixel_pos_mm['x']=np.tile(pixel_slice_x_pos,self.detector_slice_nb)
            self._pixel_pos_mm['y']=np.tile(pixel_slice_y_pos,self.detector_slice_nb)
            self._pixel_pos_mm['z']=np.repeat(self._detector_slice_z_coordinates,self.pixels_per_slice_nb)
            self._detector_pitch_mm=angular_pitch*self.sdd_mm

          
    def PlaceSources(self):
        """!@brief
            Create the sources for fan and cone beam geometries
        """ 
        if self.mode==Mode.FANBEAM:
            self._source_pos_mm=np.zeros(self.detector_slice_nb, dtype=point_dtype)
            self._source_pos_mm['z']=self._detector_slice_z_coordinates
            self._source_pos_mm['x']=0
            self._source_pos_mm['y']=-self.sad_mm
        
        elif self.mode==Mode.CONEBEAM:
            self._source_pos_mm=np.zeros(self._detector_number, dtype=point_dtype)
            
            if self._detector_number==1:
                self._source_pos_mm['x']=0
                self._source_pos_mm['y']=-self.sad_mm
                self._source_pos_mm['z']=0

            elif self._detector_number==2:
                self._source_pos_mm['x'][0]=0
                self._source_pos_mm['y'][0]=-self.sad_mm
                self._source_pos_mm['z'][0]=0
                
                self._source_pos_mm['x'][1]=-self.sad_mm
                self._source_pos_mm['y'][1]=0         
                self._source_pos_mm['z'][1]=0
            
            elif self._detector_number==3:
                self._source_pos_mm['x'][0]=0
                self._source_pos_mm['y'][0]=-self.sad_mm
                self._source_pos_mm['z'][0]=0
                
                self._source_pos_mm['x'][1]=-self.sad_mm
                self._source_pos_mm['y'][1]=0
                self._source_pos_mm['z'][1]=0

                self._source_pos_mm['x'][2]=0
                self._source_pos_mm['y'][2]=0        
                self._source_pos_mm['z'][2]=-self.sad_mm
        
            #print('self._source_pos_mm[x], self._source_pos_mm[y], self._source_pos_mm[z]:', self._source_pos_mm['x'], self._source_pos_mm['y'], self._source_pos_mm['z'])

        

    def PlaceSourcesAndPixelsParallelBeam(self):
        """!@brief
            Place the srcs and the pixels for parallel beam geometry
        """
        # pixel position
        self._pixel_pos_mm = np.zeros(self._detector_pixel_nb * self._detector_number, dtype=point_dtype)
        # sources position
        self._source_pos_mm = np.zeros(self._detector_pixel_nb * self._detector_number, dtype=point_dtype)
        
        # gantry angle 0 is when detector is moving along x axis
        c0,self._detector_pitch_mm =np.linspace(
            (-self.image_matrix_size_mm[0] + self.voxel_size_mm[0]) / 2,
            ( self.image_matrix_size_mm[0] - self.voxel_size_mm[0]) / 2,
            self.pixels_per_slice_nb,
            retstep=True,)

        c1,self._detector_pitch_mm =np.linspace(
            (-self.image_matrix_size_mm[1] + self.voxel_size_mm[1]) / 2,
            ( self.image_matrix_size_mm[1] - self.voxel_size_mm[1]) / 2,
            self.pixels_per_slice_nb,
            retstep=True,)
        
        c2,self._detector_pitch_mm =np.linspace(
            (-self.image_matrix_size_mm[2] + self.voxel_size_mm[2]) / 2,
            ( self.image_matrix_size_mm[2] - self.voxel_size_mm[2]) / 2,
            self.pixels_per_slice_nb,
            retstep=True,)        
        
        #SOURCE-DETECTOR POSITIONS 
        if self._detector_number==1: 
            self._source_pos_mm["x"] = np.tile(c0,self.detector_slice_nb)
            self._pixel_pos_mm["x"] =  self._source_pos_mm["x"]
            # this value is irrelevant in this geometry as long as is bigger that the TR y extension
            self._source_pos_mm["y"] = -self.image_matrix_size_mm[0]
            self._pixel_pos_mm["y"] =   self.image_matrix_size_mm[0]
            self._pixel_pos_mm["z"] =   np.repeat(self._detector_slice_z_coordinates,self.pixels_per_slice_nb)
            self._source_pos_mm["z"]=   self._pixel_pos_mm["z"]

        elif self._detector_number==2: 
            i = self._detector_pixel_nb
            self._source_pos_mm["x"][0:i] = np.tile(c0,self.detector_slice_nb)
            print('self._pixel_pos_mm["x"].shape, self._source_pos_mm["x"].shape: ', self._pixel_pos_mm["x"].shape, self._source_pos_mm["x"].shape)
            self._pixel_pos_mm["x"][0:i] =  self._source_pos_mm["x"][:i]
            # this value is irrelevant in this geometry as long as is bigger that the TR y extension
            self._source_pos_mm["y"][0:i] = -self.image_matrix_size_mm[0]
            self._pixel_pos_mm["y"][0:i] =   self.image_matrix_size_mm[0]
            self._pixel_pos_mm["z"][0:i] =   np.repeat(self._detector_slice_z_coordinates,self.pixels_per_slice_nb)
            self._source_pos_mm["z"][0:i]=   self._pixel_pos_mm["z"][:i]

            self._source_pos_mm["x"][i:] = np.tile(c2,self.detector_slice_nb)
            self._pixel_pos_mm["x"][i:] =  self._source_pos_mm["x"][i:]
            # this value is irrelevant in this geometry as long as is bigger that the TR y extension
            self._source_pos_mm["z"][i:] = -self.image_matrix_size_mm[0]
            self._pixel_pos_mm["z"][i:] =   self.image_matrix_size_mm[0]
            self._pixel_pos_mm["y"][i:] =   np.repeat(self._detector_slice_y_coordinates,self.pixels_per_slice_nb)
            self._source_pos_mm["y"][i:] =   self._pixel_pos_mm["y"][i:]

        elif self._detector_number==3: 
            i = self._detector_pixel_nb
            self._source_pos_mm["x"][0:i] = np.tile(c0,self.detector_slice_nb)
            self._pixel_pos_mm["x"][0:i] =  self._source_pos_mm["x"][:i]
            # this value is irrelevant in this geometry as long as is bigger that the TR y extension
            self._source_pos_mm["y"][0:i] = -self.image_matrix_size_mm[0]
            self._pixel_pos_mm["y"][0:i] =   self.image_matrix_size_mm[0]
            self._pixel_pos_mm["z"][0:i] =   np.repeat(self._detector_slice_z_coordinates,self.pixels_per_slice_nb)
            self._source_pos_mm["z"][0:i]=   self._pixel_pos_mm["z"][:i]

            self._source_pos_mm["x"][i:2*i] = np.tile(c2,self.detector_slice_nb)
            self._pixel_pos_mm["x"][i:2*i] =  self._source_pos_mm["x"][i:2*i]
            # this value is irrelevant in this geometry as long as is bigger that the TR y extension
            self._source_pos_mm["z"][i:2*i] = -self.image_matrix_size_mm[0]
            self._pixel_pos_mm["z"][i:2*i] =   self.image_matrix_size_mm[0]
            self._pixel_pos_mm["y"][i:2*i] =   np.repeat(self._detector_slice_y_coordinates,self.pixels_per_slice_nb)
            self._source_pos_mm["y"][i:2*i] =   self._pixel_pos_mm["y"][i:2*i]

            self._source_pos_mm["y"][2*i:] = np.tile(c1,self.detector_slice_nb)
            self._pixel_pos_mm["y"][2*i:] =  self._source_pos_mm["y"][2*i:]
            # this value is irrelevant in this geometry as long as is bigger that the TR y extension
            self._source_pos_mm["x"][2*i:] = -self.image_matrix_size_mm[1]
            self._pixel_pos_mm["x"][2*i:] =   self.image_matrix_size_mm[1]
            self._pixel_pos_mm["z"][2*i:] =   np.repeat(self._detector_slice_z_coordinates,self.pixels_per_slice_nb)
            self._source_pos_mm["z"][2*i:]=   self._pixel_pos_mm["z"][2*i:]            

    
    def PlaceSourcesAndPixelsFanBeam(self):
        """!@brief
            Place the srcs and the pixels for parallel beam geometry
        """
        self.PlaceSources()
        self.PlacePixels(self.detector_shape)

    def PlaceSourcesAndPixelsConeBeam(self):
        '''@!brief
            Place the srcs and the pixels for cone beam geometry
        '''
        self.PlaceSources()
        self.PlacePixels(self.detector_shape)
    
    def GenerateProjectionsExtremaFanBeam(self):    
        '''!@brief
            Create all the projection extrema for fan beam geometry
        '''    
        self._projections_extrema=np.zeros(self._detector_pixel_nb*self.gantry_angles_nb,
                                           dtype=proj_extrema_dtype)
        # create an array containing the angle of each projection
        g_angles = np.repeat(self._grantry_angles, self._detector_pixel_nb)
        #
        i=0
        srcs_gantry0  =np.zeros(self._detector_pixel_nb,dtype=point_dtype)
        pixels_grantry0=np.zeros(self._detector_pixel_nb,dtype=point_dtype)
        for s in range(len(self._source_pos_mm)):
            for p in range(self.pixels_per_slice_nb): 
                srcs_gantry0[i]=self._source_pos_mm[s]
                pixels_grantry0[i]=self._pixel_pos_mm[p+s*self.pixels_per_slice_nb]
                i+=1
        srcs_gantry0=  np.tile(srcs_gantry0,   self.gantry_angles_nb)
        pixels_grantry0=np.tile(pixels_grantry0, self.gantry_angles_nb)
        self._projections_extrema["p0"] = Utils.RotatePointAlongZ(srcs_gantry0,  g_angles)
        self._projections_extrema["p1"] = Utils.RotatePointAlongZ(pixels_grantry0,g_angles)
        

    def GenerateProjectionsExtremaConeBeam(self):
        '''!@brief
            Create all the projection extrema for cone beam geometry
        '''   

        number_of_projections = self._detector_number * self._detector_pixel_nb * self.gantry_angles_nb
        self._projections_extrema = np.zeros(number_of_projections,
                                            dtype=proj_extrema_dtype)
        
        g_angles = np.repeat(self._grantry_angles, self._detector_pixel_nb) # create an array containing the angle of each projection
        
        print('self._detector_number, self._detector_pixel_nb, self.gantry_angles_nb: %s, %s, %s', self._detector_number, self._detector_pixel_nb, self.gantry_angles_nb)
        print('g_angles.shape: %s, g_angles: %s', g_angles.shape, np.unique(g_angles))        
        
        # 
        srcs_gantry0    = np.zeros(self._detector_number * self._detector_pixel_nb, dtype=point_dtype)
        pixels_grantry0 = np.zeros(self._detector_number * self._detector_pixel_nb, dtype=point_dtype)
        i=0
        
        print('_pixel_pos_mm.shape: ', self._pixel_pos_mm.shape, '#this is the length of the inner loop')
        print('self._source_pos_mm: ', self._source_pos_mm, '#this is the position of the source for ')

        if self._detector_number==1:
            for p in self._pixel_pos_mm: 
                srcs_gantry0[i]=self._source_pos_mm
                pixels_grantry0[i]=p
                i+=1
            srcs_gantry0    = np.tile(srcs_gantry0,    self.gantry_angles_nb)
            pixels_grantry0 = np.tile(pixels_grantry0, self.gantry_angles_nb)
            self._projections_extrema["p0"] = Utils.RotatePointAlongZ(srcs_gantry0,  g_angles)
            self._projections_extrema["p1"] = Utils.RotatePointAlongZ(pixels_grantry0, g_angles)
            
        elif self._detector_number==2:
            metà = int(len(self._pixel_pos_mm)*0.5)
            for p in self._pixel_pos_mm[:metà]: 
                srcs_gantry0[i]=self._source_pos_mm[0]
                pixels_grantry0[i]=p
                i+=1
            for p in self._pixel_pos_mm[metà:]: 
                srcs_gantry0[i]=self._source_pos_mm[1]
                pixels_grantry0[i]=p
                i+=1
                
            n = self._detector_pixel_nb    
            l = self._detector_pixel_nb * self.gantry_angles_nb
            print('l,n: ', l, n)
            srcs_gantry0_1    = np.tile(srcs_gantry0[:n],    self.gantry_angles_nb)
            pixels_grantry0_1 = np.tile(pixels_grantry0[:n], self.gantry_angles_nb)
            self._projections_extrema["p0"][:l] = Utils.RotatePointAlongZ(srcs_gantry0_1,  g_angles)
            self._projections_extrema["p1"][:l] = Utils.RotatePointAlongZ(pixels_grantry0_1, g_angles)
            srcs_gantry0_2   = np.tile(srcs_gantry0[n:],    self.gantry_angles_nb)
            pixels_grantry0_2 = np.tile(pixels_grantry0[n:], self.gantry_angles_nb)
            self._projections_extrema["p0"][l:] = Utils.RotatePointAlongZ(srcs_gantry0_2,  g_angles)
            self._projections_extrema["p1"][l:] = Utils.RotatePointAlongZ(pixels_grantry0_2, g_angles)
        
        elif self._detector_number==3:
            metà = int(len(self._pixel_pos_mm)/3)
            
            for p in self._pixel_pos_mm[:metà]: 
                srcs_gantry0[i]=self._source_pos_mm[0]
                pixels_grantry0[i]=p
                i+=1
            for p in self._pixel_pos_mm[metà:metà*2]: 
                srcs_gantry0[i]=self._source_pos_mm[1]
                pixels_grantry0[i]=p
                i+=1

            for p in self._pixel_pos_mm[metà*2:]: 
                srcs_gantry0[i]=self._source_pos_mm[2]
                pixels_grantry0[i]=p
                i+=1

            n = self._detector_pixel_nb    
            l = self._detector_pixel_nb * self.gantry_angles_nb
            srcs_gantry0_1    = np.tile(srcs_gantry0[:n],    self.gantry_angles_nb)
            pixels_grantry0_1 = np.tile(pixels_grantry0[:n], self.gantry_angles_nb)
            self._projections_extrema["p0"][:l] = Utils.RotatePointAlongZ(srcs_gantry0_1,  g_angles)
            self._projections_extrema["p1"][:l] = Utils.RotatePointAlongZ(pixels_grantry0_1, g_angles)
            srcs_gantry0_2    = np.tile(srcs_gantry0[n:2*n],    self.gantry_angles_nb)
            pixels_grantry0_2 = np.tile(pixels_grantry0[n:2*n], self.gantry_angles_nb)
            self._projections_extrema["p0"][l:2*l] = Utils.RotatePointAlongZ(srcs_gantry0_2,  g_angles)
            self._projections_extrema["p1"][l:2*l] = Utils.RotatePointAlongZ(pixels_grantry0_2, g_angles)
            srcs_gantry0_3    = np.tile(srcs_gantry0[2*n:],    self.gantry_angles_nb)
            pixels_grantry0_3 = np.tile(pixels_grantry0[2*n:], self.gantry_angles_nb)
            self._projections_extrema["p0"][2*l:] = Utils.RotatePointAlongY(srcs_gantry0_3,  g_angles)
            self._projections_extrema["p1"][2*l:] = Utils.RotatePointAlongY(pixels_grantry0_3, g_angles)
        
        print('srcs_gantry0, pixels_grantry0: ', srcs_gantry0, pixels_grantry0)
        #print('self._projections_extrema["p0"], self._projections_extrema["p1"]: ', self._projections_extrema["p0"], self._projections_extrema["p1"])
        #print('self._projections_extrema["p0"]: ', np.unique(self._projections_extrema["p0"]))
        #print('self._projections_extrema["p1"]: ', np.unique(self._projections_extrema["p1"]))
        
        """            srcs = np.tile(self._source_pos_mm[0:i], self.gantry_angles_nb)
                    pixels = np.tile(self._pixel_pos_mm[0:i], self.gantry_angles_nb)
                    # apply the rotation
                    self._projections_extrema["p0"][:l] = Utils.RotatePointAlongZ(srcs, g_angles)
                    self._projections_extrema["p1"][:l] = Utils.RotatePointAlongZ(pixels, g_angles)
        """

        print('p0 unique, counts: ', np.unique(self._projections_extrema["p0"], return_counts=True))
        print('p1 unique X, counts: ' , np.unique(self._projections_extrema["p1"]['x'], return_counts=True))
        print('p1 unique Y , counts: ' , np.unique(self._projections_extrema["p1"]['y'], return_counts=True))
        print('p1 unique Z, counts: ' , np.unique(self._projections_extrema["p1"]['z'], return_counts=True))

        print('shape p0 and p1: ',  self._projections_extrema["p0"].shape, self._projections_extrema["p1"].shape)
    
    
        
    def GenerateProjectionsExtremaParallelBeam(self):
        """!@brief
            Generate the coordinates of the projection extrema that will be used by the reconstruction algorithms.
        """
        self._projections_extrema = np.zeros(
            self.gantry_angles_nb * self._detector_pixel_nb * self._detector_number,
            dtype=proj_extrema_dtype,)

        i = self._detector_pixel_nb
        l = self._detector_pixel_nb * self.gantry_angles_nb
        
        # create an array containing the angle of each projection
        g_angles = np.repeat(self._grantry_angles, self._detector_pixel_nb)

        if self._detector_number==1:
            # create an array containing the coordinates of the src and pixels before the rotation
            srcs = np.tile(self._source_pos_mm, self.gantry_angles_nb)
            pixels = np.tile(self._pixel_pos_mm, self.gantry_angles_nb)
            # apply the rotation
            self._projections_extrema["p0"] = Utils.RotatePointAlongZ(srcs, g_angles)
            self._projections_extrema["p1"] = Utils.RotatePointAlongZ(pixels, g_angles)
            

        elif self._detector_number==2: 
            # create an array containing the coordinates of the src and pixels before the rotation
            srcs = np.tile(self._source_pos_mm[0:i], self.gantry_angles_nb)
            pixels = np.tile(self._pixel_pos_mm[0:i], self.gantry_angles_nb)
            # apply the rotation
            self._projections_extrema["p0"][:l] = Utils.RotatePointAlongZ(srcs, g_angles)
            self._projections_extrema["p1"][:l] = Utils.RotatePointAlongZ(pixels, g_angles)

            # create an array containing the coordinates of the src and pixels before the rotation
            srcs = np.tile(self._source_pos_mm[i:], self.gantry_angles_nb)
            pixels = np.tile(self._pixel_pos_mm[i:], self.gantry_angles_nb)
            # apply the rotation
            self._projections_extrema["p0"][l:] = Utils.RotatePointAlongZ(srcs, g_angles)
            self._projections_extrema["p1"][l:] = Utils.RotatePointAlongZ(pixels, g_angles)

        elif self._detector_number==3:
            srcs = np.tile(self._source_pos_mm[0:i], self.gantry_angles_nb)
            pixels = np.tile(self._pixel_pos_mm[0:i], self.gantry_angles_nb)
            # apply the rotation
            self._projections_extrema["p0"][:l] = Utils.RotatePointAlongZ(srcs, g_angles)
            self._projections_extrema["p1"][:l] = Utils.RotatePointAlongZ(pixels, g_angles)

            # create an array containing the coordinates of the src and pixels before the rotation
            srcs = np.tile(self._source_pos_mm[i:2*i], self.gantry_angles_nb)
            pixels = np.tile(self._pixel_pos_mm[i:2*i], self.gantry_angles_nb)
            # apply the rotation
            self._projections_extrema["p0"][l:2*l] = Utils.RotatePointAlongZ(srcs, g_angles)
            self._projections_extrema["p1"][l:2*l] = Utils.RotatePointAlongZ(pixels, g_angles)    
    
            # create an array containing the coordinates of the src and pixels before the rotation
            srcs = np.tile(self._source_pos_mm[2*i:], self.gantry_angles_nb)
            pixels = np.tile(self._pixel_pos_mm[2*i:], self.gantry_angles_nb)
            # apply the rotation
            self._projections_extrema["p0"][2*l:] = Utils.RotatePointAlongX(srcs, g_angles)
            self._projections_extrema["p1"][2*l:] = Utils.RotatePointAlongX(pixels, g_angles)    

        print('p0: ', self._projections_extrema["p0"])
        print('p1: ', self._projections_extrema["p1"])
    
    def GenSinogramIndices(self):
        """!@brief
            Generate the sinogram indices associated to each projection
        """
        x=np.repeat(range(0,self.detector_slice_nb),self.pixels_per_slice_nb)
        self._bins=np.zeros(self._detector_pixel_nb*self.gantry_angles_nb,dtype=bin_dtype)
        self._bins['s']    =np.tile(range(0,self.pixels_per_slice_nb),self.detector_slice_nb*self.gantry_angles_nb)
        self._bins['theta']=np.repeat(range(0,self.gantry_angles_nb), self._detector_pixel_nb)
        self._bins['slice']=np.tile(x,self.gantry_angles_nb)
        self._radial_step_mm=self._detector_pitch_mm
    
    def Update(self):
        """!@brief
            Run all the function to compute the geometry of the CT experimental setup
        """
        self.__Validate()
        self.CalculateTRSizeInVoxels()
        # calculate the total number of pixel and slice of the detector
        self._detector_pixel_nb=self.pixels_per_slice_nb*self.detector_slice_nb
        # calculate the slice coordinates along z
        self._detector_slice_z_coordinates=np.arange(-self.detector_slice_nb+1,self.detector_slice_nb+1,2)/2*self.slice_pitch_mm
        # calculate the slice coordinates along y
        self._detector_slice_y_coordinates=np.arange(-self.detector_slice_nb+1,self.detector_slice_nb+1,2)/2*self.slice_pitch_mm
        
        # calculate the slice coordinates along x 
        self._detector_slice_x_coordinates=np.arange(-self.detector_slice_nb+1,self.detector_slice_nb+1,2)/2*self.slice_pitch_mm

        # generate the gantry angles 
        self._grantry_angles = np.arange(
            0,
            self.angular_range_deg,
            self.angular_range_deg / self.gantry_angles_nb,
        )
        # 
        if self.mode == Mode.PARALLELBEAM:
            self.PlaceSourcesAndPixelsParallelBeam()
            self.GenerateProjectionsExtremaParallelBeam()
        elif self.mode==Mode.FANBEAM:
            self.PlaceSourcesAndPixelsFanBeam()
            self.GenerateProjectionsExtremaFanBeam()
        elif self.mode==Mode.CONEBEAM:
            self.PlaceSourcesAndPixelsConeBeam()
            self.GenerateProjectionsExtremaConeBeam()
        self._number_of_projections=self._projections_extrema.shape[0]
        self._z_range_mm=np.max(self._source_pos_mm["z"])-np.min(self._source_pos_mm["z"])+self.slice_pitch_mm
        self._angular_step_deg = (
            self.angular_range_deg / self.gantry_angles_nb
        )
        self.GenSinogramIndices()


    def Draw(self, use_jupyter=0,camera_pos_mm=(0,100,100)):
        """!@brief
             Render an experimental setup containing a CT detector
             @param use_jupyter: if 1 return an image representing the scene, otherwise a vtk interactive rendering is used
        """
        ren = vtk.vtkRenderer()
        camera =vtk.vtkCamera ()
        camera.SetPosition(camera_pos_mm)
        camera.SetFocalPoint(0, 0, 0)
        ren.SetActiveCamera(camera)
        # add detector
        ren.AddActor(CreateLabel(self._pixel_pos_mm[0], "Detector"))
        # add pixels
        for _pixel in self._pixel_pos_mm:
            if self.detector_shape==DetectorShape.PLANAR:
                ren.AddActor(
                    CreateCube(
                        _pixel,
                        default_pixel_size_CT,
                        0,
                        np.asarray([0, -1, 0]),
                        par_pixel_color,
                    )
                )
            elif self.detector_shape==DetectorShape.ARC:
                ren.AddActor(
                    CreateAutoOrientedeCube(
                        _pixel,
                        default_pixel_size_CT,
                        par_pixel_color,
                    )
                )
        # add srcs
        for _src in self._source_pos_mm:
            #ren.AddActor(CreateSphere(_src,self.slice_pitch_mm/2.0, par_src_color))            
            ren.AddActor(CreateSphere(_src,default_src_size_CT, par_src_color))            
        ren.AddActor(CreateLabel(self._source_pos_mm[0], "Sources"))
        #  add 3d axis
        ren.AddActor(CreateAxesActor(5))
        # add the TR
        ren.AddActor(CreateTR(self.image_matrix_size_mm, self.voxel_size_mm))
        # add a corner annotation
        ren.AddActor(CreateCornerAnnotation((10, 10), self.GetInfo()))
        ren.SetBackground(par_background_color)
        # if not jupyter use vtk to create a window
        if not use_jupyter:
            renWin = vtk.vtkRenderWindow()
            renWin.AddRenderer(ren)
            # create a render and a window interactor
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)
            iren.Start()
        else:
            data=RenderSceneJupyter(ren)
            return data