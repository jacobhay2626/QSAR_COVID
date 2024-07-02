import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from MLP.nn import MLP

df5 = pd.read_csv('Data/df_2classification.csv')


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

    return fp_data


np_fps = fingerprints(df5.canonical_smiles)
list_fps = np_fps.tolist()

df5['bioactivity_class'] = df5['bioactivity_class'].map(
    {'active': 1, 'inactive': 0})


bioactivity_data = []
for i in df5['bioactivity_class']:
    bioactivity_data.append(i)

# Two hidden layers
# Number of hidden neurons = 2/3 input layer + output layer
n = MLP(350, [150, 84, 1])


###############
# Need to work out training splits
xs = []

ys = []  # desired targets
################

for k in range(50):

    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # back pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # gradient descent
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)



