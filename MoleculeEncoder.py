"""MoleculeEncoder.py"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem

#from molvs import standardize_smiles   TODO: install molvs, some error with molvs



class MoleculeEncoder:

        
    """ MoleculeEncoder:
    
    Encodes molecules using a variety of Fingerprints
    """

    def __init__(self, 
                 data,
                 encoding = "geometry", 
                 ):

        self.data = data
        self.encoding = encoding
        self.encoded_data = self.encode_molecules()


    def encode_molecules(self):

        """Encodes molecules using a variety of Fingerprints"""

        # TODO

        # geometry encoding

        # Fingerprint 1

        # Fingerprint 2

        # ...


        return self.data