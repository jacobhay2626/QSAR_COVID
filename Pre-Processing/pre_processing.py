import pandas as pd
from chembl_webresource_client.new_client import new_client

# Searching for corona virus
target = new_client.target
target_query = target.search('coronavirus')
targets = pd.DataFrame.from_dict(target_query)
# Index(['cross_references', 'organism', 'pref_name', 'score',
#        'species_group_flag', 'target_chembl_id', 'target_components',
#        'target_type', 'tax_id']

# SARS coronavirus 3C-like proteinase
# CHMBL id = CHEMBL3927
selected_target = targets.target_chembl_id[6]
print(selected_target)

# Only want the activity of these compounds to be in IC50 values
activity = new_client.activity
res = activity.filter(target_chembl_id=selected_target).filter(standard_type="IC50")

df = pd.DataFrame.from_dict(res)

# print(df.standard_type.unique())
# Only IC50 in the column.
# drop any na values for standard value (there are none here)
df2 = df[df.standard_value.notna()]

# Pre processing steps
# Data I want from this dataframe:
# molecule_chembl_id, bioactivity (via standard values), standard values, and canonical_smiles

# We now want to use the IC50 values to determine if the molecule is active or inactive.

# Are they all the same scale?
# print(df2.standard_units)

# What do the values look like?
# print(df2.standard_value)


# print(type(df2.standard_value[2]))
# NOTE the data in this column are strings so to perform any mathematical operation need to either int or float
# Float for precision

bioactivity_class = []
for i in df2.standard_value:
    if float(i) >= 10000:
        bioactivity_class.append("inactive")
    elif float(i) <= 1000:
        bioactivity_class.append('active')
    else:
        bioactivity_class.append("intermediate")

selection = ['molecule_chembl_id', 'standard_value', 'canonical_smiles']
df3 = df2[selection]

df4 = pd.concat([df3, pd.DataFrame(bioactivity_class)], axis=1)
print(df4)

# Analysing the data:

