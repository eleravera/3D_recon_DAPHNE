import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize

# Initialize red_line as None
red_line = None

def Visualize3dImage(img, slice_axis=2, _cmap='gray', symmetric_colorbar=False, title=''):
    """!@brief Display a 3D ndarray with a slider to move along the third dimension.
            Extra keyword arguments are passed to imshow
            @param img: 3D image array
            @param slice_axis: axis along which to slice the image (default is 2)
            @param _cmap: colormap for the image
            @param symmetric_colorbar: flag to enable symmetric divergent colorbar centered around zero
    """
    # Check if img is a 3D array
    if not img.ndim == 3:
        raise ValueError("img should be an ndarray with ndim == 3")
    
    # Transpose the image if necessary to slice along the chosen axis
    transpose_rule = [0, 1, 2]
    transpose_rule[2], transpose_rule[slice_axis] = transpose_rule[slice_axis], transpose_rule[2]
    t_img = np.transpose(img, tuple(transpose_rule))
    
    # Find global min and max across the entire 3D array
    global_min = np.min(t_img)
    global_max = np.max(t_img)
    
    if symmetric_colorbar:
        # Create symmetric color normalization
        norm = Normalize(vmin=-max(abs(global_min), abs(global_max)), vmax=max(abs(global_min), abs(global_max)))
        _cmap = 'RdBu_r'
    else:
        # Use the global min and max for color normalization
        norm = Normalize(vmin=global_min, vmax=global_max)
    
    # Create figure and axis for the plot
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(title, fontsize=14) 
    ax1 = plt.Axes(fig, [0.1, 0.1, 0.6, .80])
    fig.add_axes(ax1)
    
    # Set initial slice to display
    current_slice = t_img[:, :, 0]
    
    # Display the image with the chosen colormap and color normalization
    l = ax1.imshow(current_slice, cmap=_cmap, norm=norm)
    
    # Add colorbar with fixed limits
    fig.colorbar(l, ax=ax1, orientation='vertical')
    
    # Add slider to navigate slices
    ax = fig.add_axes([0.1, 0, 0.6, 0.03])
    slice_max = img.shape[slice_axis] - 1
    slice_min = 0
    slider = Slider(ax, 'slice', slice_min, slice_max, valinit=0, valfmt='%i')
    
    def UpdateImage(event):
        slice_number = int(slider.val)
        current_slice = t_img[:, :, slice_number]
        
        # Update only the image, not the color normalization
        l.set_data(current_slice)
        fig.canvas.draw_idle()
    
    def Move(event):
        if event.key == 'right':
            val = min(slice_max, slider.val + 1)
            slider.set_val(val)
        if event.key == 'left':
            val = max(0, slider.val - 1)
            slider.set_val(val)
        UpdateImage(event)
    
    # Connect keyboard events for slice navigation
    fig.canvas.mpl_connect('key_press_event', Move)
    slider.on_changed(UpdateImage)
    
    
    
    
def Visualize3dImageWithProfile(img, slice_axis=2, profile_axis=0, _cmap='gray', symmetric_colorbar=False, title=''):
    """!@brief Display a 3D ndarray with a slider to move along the third dimension.
            Extra keyword arguments are passed to imshow
            @param img: 3D image array
            @param slice_axis: axis along which to slice the image (default is 2)
            @param _cmap: colormap for the image
            @param symmetric_colorbar: flag to enable symmetric divergent colorbar centered around zero
    """

    # Check if img is a 3D array
    if not img.ndim == 3:
        raise ValueError("img should be an ndarray with ndim == 3")
    
    # Transpose the image if necessary to slice along the chosen axis
    transpose_rule = [0, 1, 2]
    transpose_rule[2], transpose_rule[slice_axis] = transpose_rule[slice_axis], transpose_rule[2]
    t_img = np.transpose(img, tuple(transpose_rule))
    
    # Find global min and max across the entire 3D array
    global_min = np.nanmin(t_img)
    global_max = np.nanmax(t_img)
    
    if symmetric_colorbar:
        # Create symmetric color normalization
        vmin, vmax = -max(abs(global_min), abs(global_max)), max(abs(global_min), abs(global_max))
        norm = Normalize(vmin, vmax)
        _cmap = 'RdBu_r'
    else:
        vmin, vmax = global_min, global_max
        # Use the global min and max for color normalization
        norm = Normalize(vmin, vmax)


    # Create figure and axis for the plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax1, ax2 = axes
    fig.suptitle(title, fontsize=14)
    
    # Set initial slice to display
    current_slice = t_img[:, :, 0]
    profile_position = 0
    

    # Display the image with the chosen colormap and color normalization
    l = ax1.imshow(current_slice, cmap=_cmap, norm=norm)
    
    # Add colorbar with fixed limits
    fig.colorbar(l, ax=ax1, orientation='vertical')
    
    current_profile = current_slice[:, profile_position] if profile_axis == 0 else current_slice[profile_position, :]
    profile_line, = ax2.plot(current_profile)
    
    # Add slider to navigate slices
    ax_slider1 = plt.axes([0.1, 0.02, 0.35, 0.03])
    slice_max = img.shape[slice_axis] - 1
    slice_min = 0
    slider = Slider(ax_slider1, 'Slice', slice_min, slice_max, valinit=0, valfmt='%i')

    # Add slider to select profile position
    ax_slider2 = plt.axes([0.55, 0.02, 0.35, 0.03])
    profile_max = current_slice.shape[1 - profile_axis] - 1
    slider2 = Slider(ax_slider2, 'Profile Position', 0, profile_max, valinit=0, valfmt='%i')
    

    ax2.set_ylim(vmin, vmax)
    
    def UpdateImage(event):
        global red_line  # Aggiungi questa riga per dichiarare red_line come variabile globale
    
        slice_number = int(slider.val)
        current_slice = t_img[:, :, slice_number]

        # Update only the image, not the color normalization
        l.set_data(current_slice)
        
        profile_position = int(slider2.val)
        current_profile = current_slice[:, profile_position] if profile_axis == 0 else current_slice[profile_position, :]

        profile_line.set_ydata(current_profile)
        
        ax2.relim()
        ax2.autoscale_view()
        
        # Remove the previous red line, if it exists
        if red_line is not None:
            red_line.remove()
            
        # Add a new red line for the profile
        if profile_axis == 0:
            # For a horizontal slice, draw a horizontal line
            red_line = ax1.axvline(profile_position, color='red', linestyle='-', linewidth=2)
        else:
            # For a vertical slice, draw a vertical line
            red_line = ax1.axhline(profile_position, color='red', linestyle='-', linewidth=2)               
        fig.canvas.draw_idle()
        
    def Move1(event):
        if event.key == 'right':
            val = min(slice_max, slider.val + 1)
            slider.set_val(val)
          
        if event.key == 'left':
            val = max(0, slider.val - 1)
            slider.set_val(val)            
        UpdateImage(event)    
    
    def Move2(event):        
        if event.key == 'right':
            val = min(profile_max, slider2.val + 1)
            slider2.set_val(val)
                        
        if event.key == 'left':            
            val = max(0, slider2.val - 1)
            slider2.set_val(val)
            
        UpdateImage(event)
    
    # Connect keyboard events for slice navigation
    fig.canvas.mpl_connect('key_press_event', Move1)
    fig.canvas.mpl_connect('key_press_event', Move2)
    slider.on_changed(UpdateImage)
    slider2.on_changed(UpdateImage)
 


def Visualize3dImageWithProfileAndHistogram(img, slice_axis=2, profile_axis=0, _cmap='gray', symmetric_colorbar=False, title='', log_scale_hist = False):
    """!
    Display a 3D ndarray with sliders to move along the third dimension, 
    with a profile and histogram of the signal in the current slice.
    
    @param img: 3D image array
    @param slice_axis: axis along which to slice the image (default is 2)
    @param _cmap: colormap for the image
    @param symmetric_colorbar: flag to enable symmetric divergent colorbar centered around zero
    """
    # Check if img is a 3D array
    if not img.ndim == 3:
        raise ValueError("img should be an ndarray with ndim == 3")
    
    # Transpose the image if necessary to slice along the chosen axis
    transpose_rule = [0, 1, 2]
    transpose_rule[2], transpose_rule[slice_axis] = transpose_rule[slice_axis], transpose_rule[2]
    t_img = np.transpose(img, tuple(transpose_rule))
    
    # Find global min and max across the entire 3D array
    global_min = np.nanmin(t_img)
    global_max = np.nanmax(t_img)
    
    if symmetric_colorbar:
        # Create symmetric color normalization
        vmin, vmax = -max(abs(global_min), abs(global_max)), max(abs(global_min), abs(global_max))
        #cleaned_data = img[np.isfinite(img)]
        #vmin, vmax = np.percentile(cleaned_data, 5), np.percentile(cleaned_data, 95)
        norm = Normalize(vmin, vmax)
        _cmap = 'RdBu_r'
        #print("vmin, vmax: ", vmin, vmax)
    else:
        vmin, vmax = global_min, global_max
        # Use the global min and max for color normalization
        norm = Normalize(vmin, vmax)

    # Create figure and axis for the plot (3 subplots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax1, ax2, ax3 = axes
    fig.suptitle(title, fontsize=14)
    
    # Set initial slice to display
    current_slice = t_img[:, :, 0]
    profile_position = 0

    # Display the image with the chosen colormap and color normalization
    l = ax1.imshow(current_slice, cmap=_cmap, norm=norm)
    # Add colorbar with fixed limits
    fig.colorbar(l, ax=ax1, orientation='vertical')

    current_profile = current_slice[:, profile_position] if profile_axis == 0 else current_slice[profile_position, :]
    profile_line, = ax2.plot(current_profile)
    ax2.set_ylim(vmin, vmax)
    ax2.set_xlim(0, len(current_profile))
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Initialize histogram
    signal_values = current_slice.flatten()
    ax3.hist(signal_values[~np.isnan(signal_values)], bins=100, color='blue', alpha=0.7)
    ax3.set_title("Signal Histogram")
    ax3.set_ylabel("Frequency")
    ax3.set_xlim(vmin, vmax)    # Set fixed limits for histogram based on vmin and vmax
    
    if log_scale_hist == True:
        ax3.set_yscale('log')
    
    # Add sliders for image slice and profile position
    ax_slider1 = plt.axes([0.1, 0.01, 0.25, 0.03])  # Slider for image slice (below ax1)
    slice_max = img.shape[slice_axis] - 1
    slice_min = 0
    slider = Slider(ax_slider1, 'Slice', slice_min, slice_max, valinit=0, valfmt='%i')

    ax_slider2 = plt.axes([0.4, 0.01, 0.25, 0.03])  # Slider for profile position (below ax2)
    profile_max = current_slice.shape[1 - profile_axis] - 1
    slider2 = Slider(ax_slider2, 'Profile', 0, profile_max, valinit=0, valfmt='%i')
    
    # Red line for profile position in image
    red_line = None

    def UpdateImage(event):
        global red_line
        
        # Update the slice
        slice_number = int(slider.val)
        current_slice = t_img[:, :, slice_number]
        l.set_data(current_slice)
        
        # Update the profile
        profile_position = int(slider2.val)
        current_profile = current_slice[:, profile_position] if profile_axis == 0 else current_slice[profile_position, :]
        current_profile = np.nan_to_num(current_profile)  # Sostituisce i NaN con zeri
        profile_line.set_ydata(current_profile)
        
        # Update the histogram
        signal_values = current_slice.flatten()
        ax3.clear()
        ax3.hist(signal_values[~np.isnan(signal_values)], bins=50, color='blue', alpha=0.7)
        ax3.set_xlabel("Signal Intensity")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.set_xlim(vmin, vmax)# Set fixed x limits for histogram
        if log_scale_hist == True:
            ax3.set_yscale('log')
        
        # Remove previous red line
        if red_line is not None:
            red_line.remove()
            
        # Add a new red line for the profile
        if profile_axis == 0:
            red_line = ax1.axvline(profile_position, color='red', linestyle='-', linewidth=2)
        else:
            red_line = ax1.axhline(profile_position, color='red', linestyle='-', linewidth=2)
        
        fig.canvas.draw_idle()

    def Move1(event):
        if event.key == 'right':
            val = min(slice_max, slider.val + 1)
            slider.set_val(val)
        if event.key == 'left':
            val = max(0, slider.val - 1)
            slider.set_val(val)
        UpdateImage(event)

    def Move2(event):        
        if event.key == 'right':
            val = min(profile_max, slider2.val + 1)
            slider2.set_val(val)
        if event.key == 'left':            
            val = max(0, slider2.val - 1)
            slider2.set_val(val)
        UpdateImage(event)

    # Connect keyboard events for slice navigation
    fig.canvas.mpl_connect('key_press_event', Move1)
    fig.canvas.mpl_connect('key_press_event', Move2)
    slider.on_changed(UpdateImage)
    slider2.on_changed(UpdateImage)

    # Adjust spacing between subplots and sliders
    plt.subplots_adjust(hspace=0.5, bottom=0.1)
    



