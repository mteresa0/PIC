import glob
from paraview.simple import *

# --- CONFIGURATION ---
N = 3  # Number of species !!! for now set by hand !!!
base_path = 'Documents/PACS/outputs/' # !!! find a way to hide this !!!

view = GetActiveViewOrCreate('RenderView')

# --- POTENTIAL FIELD ---
potential_files = sorted(glob.glob(base_path + 'phi_*'))
if potential_files:
    potential_data = OpenDataFile(potential_files)
    pot_display = Show(potential_data, view)
    pot_display.Representation = 'Surface'
    
    array_name = 'solution' # name given in solver.hpp
    ColorBy(pot_display, ('POINTS', array_name))
    potential_data.UpdatePipeline()
    lut = GetColorTransferFunction(array_name)
    lut.ApplyPreset('Cool to Warm (Extended)', True)
    pot_display.SetScalarBarVisibility(view, True)
else:
    print("Warning: No potential files found.")

# --- MULTIPLE PARTICLE SPECIES ---
colors = [[1,1,1], [0,0.9,0.2], [0.95,0.25,0.]] # white, green, orange

for i in range(0, N):
    pattern = base_path + 'part-sp' + str(i) + '*'
    p_files = sorted(glob.glob(pattern))
    
    if not p_files:
        print(f"Warning: No files found for species {i} with pattern {pattern}")
        continue
        
    p_data = OpenDataFile(p_files)
    p_display = Show(p_data, view)
    
    p_display.Representation = 'Points'
    p_display.PointSize = 5.0
    p_display.AmbientColor = colors[i]

# --- FINALIZE ---
view.ResetCamera()
Render()

