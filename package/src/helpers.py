""" 
    Project:     synthesizer
    File:        helpers.py
    Description: Helper functions for the synthesizer project
    Author:      << github.com/leoluber >> 
    License:     MIT
"""



from typing import Literal
import json
import numpy as np
import pubchempy as pcp




# dictionaries for antisolvent names and MLs
ml_dictionary       = json.load(open("data/raw/ml_dictionary.json", "r"))
# dictionaries
molecule_dictionary = {
    "Tol": "Toluene",
    "Ac": "Acetone",
    "MeOH": "Methanol",
    "EtOH": "Ethanol",
    "i-PrOH": "Isopropanol",
    "n-PrOH": "Propanol",
    "n-BuOH": "Butanol",
    "butanone": "Butanone",
    "pentanone": "Pentanone",
    "CyPen": "Cyclopentanone",
    "CyPol": "Cyclopentanol",
    "PenOH": "Pentanol",
    "HexOH": "Hexanol",
    "OctOH": "Octanol",
    "3-Penon": "3-Pentanone",
}


def nm_to_ev(nm) -> float:

    """ Convert nm to eV """

    return 1239.840/nm



def ev_to_nm(ev) -> float:
    
    """ Convert eV to nm """

    return 1239.840/ev



def get_ml_from_peak_pos(peak_pos) -> int:

    """ Get the ML from a peak position """

    if peak_pos < 10:
        peak_pos = ev_to_nm(peak_pos)

    for key in ml_dictionary:
        if peak_pos <= ml_dictionary[key][1] and peak_pos >= ml_dictionary[key][0]:
            return int(key)
    
    return None



def surface_proportion(peak_pos, mode: Literal['EV', 'NM'], l = 20) -> float:	

    """ Get the surface/bulk proportion for NPLs from a given wavelength (estimation) """

    if mode == 'EV':
        nm = ev_to_nm(peak_pos)
    else:	
        nm = peak_pos	

    prop = 0
    for key, value in ml_dictionary.items():
        if value[1] > nm:
            total = l**2 * int(key)
            surf  = total - (l-2)**2 * (int(key)-2)
            internal = (l-2)**2 * (int(key)-2)

            """either return bulk or surface proportion"""
            #prop = surf / total
            prop = internal / total
            return prop

    return prop




### ------------------------- GEOMETRY ------------------------- ###

# Get coordinate for atoms in molecule from pubchempy
def get_molecule_coordinates(molecule_name: str) -> np.ndarray:
    compound = pcp.get_compounds(molecule_name, "name", record_type="3d")[0]
    atoms = compound.to_dict(properties=["atoms"])["atoms"]
    coordinates = np.array([[atom["element"], atom["x"], atom["y"], atom["z"]] for atom in atoms])
    return coordinates


def calculate_angle(v1, v2) -> float:
    """Calculate the angle between two vectors"""

    # exception for zero vectors
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        raise ValueError("Zero vector")

    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)


def cone_angle(molecule_name) -> float:
    """get cone angle from molecule name

    -----
    USAGE
    >>> cone_angle("Ethanol")
    """
    if molecule_name in ["Acetylacetone", "EthylAcetate", "MethylAcetate"]:
        return np.nan

    # get the coordinates of all atoms in the molecule
    print(molecule_name)
    coordinates = get_molecule_coordinates(molecule_name)

    # get the coordinates of the functional group
    func_coords = coordinates[coordinates[:, 0] == "O"][:, 1:].astype(float)
    rest_coords = coordinates[coordinates[:, 0] == "C"][:, 1:].astype(float)

    if len(rest_coords) < 2:
        return 0

    # get all the connecting vectors (O - C)
    vectors = np.array([func_coords - carbon for carbon in rest_coords]).squeeze()

    # calculate the angle between the axis vector and all other vectors
    max_angle = 0
    for v1 in vectors:
        for v2 in vectors:
            if np.array_equal(v1, v2):
                continue

            max_angle = max(calculate_angle(v1, v2), max_angle)

    return np.around(max_angle, 2)


def calculate_length_of_molecule(molecule_name: str, axis: int = 0):
    """Calculate the length of a molecule along a given axis."""


    coordinates = get_molecule_coordinates(molecule_name)
    coordinates_1d = coordinates[:, axis + 1].astype(float)
    return np.around(max(coordinates_1d) - min(coordinates_1d), 2)




if __name__ == "__main__":

    dict = {}

    for molecule in molecule_dictionary.values():
        # define a dictionary for the molecule
        length = calculate_length_of_molecule(molecule, axis=0)
        cone_angle_ = cone_angle(molecule)
        strength = 0.5/(0.02 * length + (0.001 * cone_angle_))
        list = [float(length), float(cone_angle_), float(strength)]

        dict[molecule] = list

    
    # list to json
    json.dump(dict, open("data/raw/molecule_geometry.json", "w"), indent=4)
