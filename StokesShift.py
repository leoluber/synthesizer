import numpy as np
import warnings
import matplotlib.pyplot as plt
import os
warnings.filterwarnings('ignore')

# custom
from helpers import *



"""
    This script is used to evaluate the Stokes Shift from the PL and Absorption peaks of PEROVSKITE NPLs
    - the data is read from the "data/spectrum" folder
    - the Stokes Shift is calculated as the difference between the PL and Absorption peak positions
    - the data is visualized in a scatter plot

"""



def read_spectrum_abs(file_path):
    
    """
        Read the data from the file
    """

    wavelength, spectrum = [], []
    with open(file_path, "r") as filestream:
        for line in filestream:
            
            if line == "": break
            nm, intensity = line.split(",")

            wavelength.append(float(nm))
            #wavelength.append(nm_to_ev(float(nm)))
            spectrum.append(float(intensity)*5)

    # find the peak position -> to be improved
    peak_pos_abs = -1
    for i in range(30):
        new_max = max(spectrum[int(i*10):])
        new_max_index = spectrum.index(new_max)
        if new_max_index == peak_pos_abs:
            break
        peak_pos_abs = new_max_index
    
    return wavelength, spectrum, wavelength[peak_pos_abs]



def read_spectrum_pl(file_path):

    """
        Read the PL data from the file
    """

    wavelength, spectrum = [], []
    with open(file_path, "r") as filestream:
        for line in filestream:
            
            if line == "": break
            nm, intensity, norm = line.split(",")

            wavelength.append(float(nm))
            #wavelength.append(nm_to_ev(float(nm)))
            spectrum.append(float(norm))

    max_intensity = max(spectrum)
    max_intensity_index = spectrum.index(max_intensity)

    return wavelength, spectrum, wavelength[max_intensity_index]



def main():

    peak, shift = [], []
    for i in range(1, 36):

        if i in [6, 7]: continue   # bs data

        abs_file_path = f"data/spectrum/j{i}A.txt"
        pl_file_path = f"data/spectrum/j{i}.txt"

        # check if the files exist
        if not os.path.exists(abs_file_path) or not os.path.exists(pl_file_path):
            continue

        # read the data
        wavelength_abs, spectrum_abs, peak_pos_abs = read_spectrum_abs(abs_file_path)
        wavelength_pl,  spectrum_pl,  peak_pos_pl  = read_spectrum_pl(pl_file_path)

        # plot the data
        plt.plot(wavelength_abs, spectrum_abs, label="Absorption")
        plt.plot(wavelength_pl,  spectrum_pl,  label="PL")
        plt.show()

        # results
        stokes_shift = peak_pos_pl - peak_pos_abs

        print(i)
        print(f"Peak Position Pl:   {peak_pos_pl} nm")
        print(f"Peak Position Abs:  {peak_pos_abs} nm")
        print(f"Stokes Shift:       {stokes_shift} nm")

        peak.append(peak_pos_pl)
        shift.append(abs(stokes_shift))


    # plot the results
    plt.scatter(peak, shift)
    plt.xlabel("Peak Position [nm]")
    plt.ylabel("Stokes Shift [nm]")
    plt.title("Stokes Shift vs Peak Position")
    plt.show()



if __name__ == "__main__":
    main()