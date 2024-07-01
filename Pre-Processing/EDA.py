import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import seed, randn
from scipy.stats import mannwhitneyu

df = pd.read_csv('../Data/bioactivity_preprocessed_data.csv')


# Now want to add in some descriptors for bioactivity
# Lipinski rule of 5
# MW
# logP
# HBDonors
# HBAcceptors
# Want to add the values for these descriptors of our molecules in the bioactivity to a dataset

# for our descriptors we need to give it a molecule in smiles.
# Going to pass the function the smiles column of the dataframe


def lipinski(smiles):
    moldata = []
    for elem in smiles:
        mol = Chem.MolFromSmiles(elem)
        moldata.append(mol)

    # Now have a list of molecules
    # Going to add all of these values into a dataset, so want an empty numpy array
    lip_data = np.arange(1, 1)
    # Now have to think about adding the data to it, the first row can be added quite simply by simply adding the array
    # of descriptors, but need to consider adding the subsequent rows. Have to use vstack.
    # initialise a condition so we can say if it is the first row or not.
    i = 0

    for mol in moldata:
        desc_MW = Descriptors.MolWt(mol)
        desc_logP = Crippen.MolLogP(mol)
        desc_HBA = Lipinski.NumHAcceptors(mol)
        desc_HBD = Lipinski.NumHDonors(mol)

        row = np.array([desc_MW,
                        desc_logP,
                        desc_HBA,
                        desc_HBD])

        if i == 0:
            lip_data = row
        else:
            lip_data = np.vstack([lip_data, row])
        i = i + 1

    columnNames = ['MW', 'logP', 'NumHAcceptors', 'NumHDonors']
    lipinski_descriptors = pd.DataFrame(lip_data, columns=['MW', 'logP', 'NumHAcceptors', 'NumHDonors'])

    return lipinski_descriptors


df_lipinski = lipinski(df.canonical_smiles)
df_combined = pd.concat([df, df_lipinski], axis=1)


# Convert IC50 to more normal distribution using pIC50
# with the larger values in the standard values, applying -log will make them negative so going to normalise

def norm_value(input):
    norm = []

    for i in input['standard_value']:
        if i > 100000000:
            i = 100000000
        norm.append(i)

    input['standard_value_norm'] = norm
    x = input.drop('standard_value', axis=1)

    return x


def pIC50(input):
    pIC50 = []

    for i in input['standard_value_norm']:
        molar = i * (10 ** 9)  # convert nM to M
        pIC50.append(-np.log10(molar))

    input['pIC50'] = pIC50
    x = input.drop('standard_value_norm', axis=1)

    return x


df_norm = norm_value(df_combined)

df_pIC50 = pIC50(df_norm)
df_pIC50.rename(columns={"0": "bioactivity_class"}, inplace=True)

# only want a binary classification problem so going to drop all the intermediate classes

df_binary = df_pIC50[df_pIC50.bioactivity_class != 'intermediate']

# Plotting the frequencies of the different classes.
# Shows a much higher frequency of inactive vs active.

plt.figure(figsize=(5.5, 5.5))

sns.countplot(x='bioactivity_class', data=df_binary, edgecolor='black')

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('Frequency', fontsize=14, fontweight='bold')

# Plotting scatter plot of MW versus LogP
plt.figure(figsize=(20, 15))

sns.scatterplot(x='MW', y='logP', data=df_binary, hue='bioactivity_class', size='pIC50', edgecolor='black', alpha=0.7)
plt.xlabel('MW', fontsize=14, fontweight='bold')
plt.ylabel('LogP', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# BOX PLOT pIC50

sns.boxplot(figsize=(20, 15))

sns.boxplot(x='bioactivity_class', y='pIC50', data=df_binary)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('pIC50', fontsize=14, fontweight='bold')


# Shows that there is a more variance in the inactive molecules, with a wider
# range betweem the minimum and maximum. IQR much greater for the inactive than active.

# Statistical tests:
# Performing a Mann and Whitney U test, possible outliers have no effect on the results

def mannwhitney(descriptor):
    # Seed the random number gen
    seed(1)

    # actives and inactives
    selection = [descriptor, 'bioactivity_class']
    df4 = df_binary[selection]
    active = df4[df4.bioactivity_class == 'active']
    active = active[descriptor]

    selection = [descriptor, 'bioactivity_class']
    df4 = df_binary[selection]
    inactive = df4[df4.bioactivity_class == 'inactive']
    inactive = inactive[descriptor]

    # compare samples
    stat, p = mannwhitneyu(active, inactive)

    # interpret
    alpha = 0.05
    if p > alpha:
        interpretation = 'Same distribution (fail to reject H0)'
    else:
        interpretation = 'Different distribution (reject H0)'

    results = pd.DataFrame({"Descriptor": descriptor,
                            'Statistics': stat,
                            "p": p,
                            "alpha": alpha,
                            "Intepretation": interpretation}, index=[0])

    return results

# pIC50: p = 4.4e-10 (reject H0)
# mannwhitney('pIC50')


# BOX PLOT MW

sns.boxplot(figsize=(20, 15))

sns.boxplot(x='bioactivity_class', y='MW', data=df_binary)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('MW', fontsize=14, fontweight='bold')


# MW: p = 0.003 (Reject H0)
# print(mannwhitney('MW'))


# BOX PLOT logP

sns.boxplot(figsize=(20, 15))

sns.boxplot(x='bioactivity_class', y='logP', data=df_binary)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('logP', fontsize=14, fontweight='bold')


# logP: p = 0.6303 (Accept H0)
# print(mannwhitney('logP')['p'])

# BOX PLOT HBA

sns.boxplot(figsize=(20, 15))

sns.boxplot(x='bioactivity_class', y='NumHAcceptors', data=df_binary)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHAcceptors', fontsize=14, fontweight='bold')


# HBA: p = 0.003402 (reject H0)
# print(mannwhitney('NumHAcceptors')['p'])


# BOX PLOT HBD

sns.boxplot(figsize=(20, 15))

sns.boxplot(x='bioactivity_class', y='NumHDonors', data=df_binary)

plt.xlabel('Bioactivity class', fontsize=14, fontweight='bold')
plt.ylabel('NumHDonors', fontsize=14, fontweight='bold')


# HBD: p = 0.000053 (reject H0)
# print(mannwhitney('NumHDonors')['p'])
