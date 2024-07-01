import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.feature_selection import VarianceThreshold

df5 = pd.read_csv('../Data/df_2classification.csv')


# Want to develop a dataframe of fingerprints for the molecules and use them in a classification problem for
# activity.

def fingerprints(smiles):
    mols = []
    for elem in smiles:
        m = Chem.MolFromSmiles(elem)
        mols.append(m)

    # Now have an array of molecules, and we are going to produce a lot of new arrays that
    # we want in a dataframe. Initalise empty numpy array.
    fp_data = np.arange(1, 1)

    i = 0

    for mol in mols:
        # Initialise fingerprint generator
        rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=350)
        row = rdkgen.GetFingerprintAsNumPy(mol)

        if i == 0:
            fp_data = row
        else:
            fp_data = np.vstack([fp_data, row])
        i = i + 1

    fingerprint_descriptors = pd.DataFrame(fp_data)

    print(fp_data)

    return fingerprint_descriptors


df_fps = fingerprints(df5.canonical_smiles)

df_fps_classification = pd.concat([df_fps, df5.bioactivity_class], axis=1)

df_fps_classification['bioactivity_class'] = df_fps_classification['bioactivity_class'].map(
    {'active': 1, 'inactive': 0})

