import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import re

data = pd.read_excel('Source Data File.xlsx')

data = data[data['category_response_1_or_non_response_0_or_unknown_2'].isin([0, 1])]

data['Group'] = data.iloc[:, 0].astype('category')  
data['DiseasesClass'] = data['DiseasesClass'].astype('category')
data['antibiotics'] = data['antibiotics'].astype('category')
data['Gender'] = data['Gender'].astype('category')

# DiseasesClass
# 1: Crohn's Disease; 2: Tourette_syndrome; 3: CDI; 4: Antibiotics Resistance;
# 5: Melanoma; 6: healthy volunteer; 7: ulcerative colitis; 8: IBS;
# 9: renal_carcinoma; 10: metabolic syndrome; 11: Diabetes; 12: Obesity; 13: PACS

def clean_column_name(col_name):
    if isinstance(col_name, str):
        col_name = col_name.replace(' ', '_')
        col_name = re.sub(r'[^a-zA-Z0-9_]', '', col_name)
    return col_name

column_mapping = {col: clean_column_name(col) for col in data.columns}
data = data.rename(columns=column_mapping)

# ANOVA-based Multiple Linear Regression

dependent_var = list(column_mapping.values())[4]  # the 4 here represengts the 5th column (increased similarity in bacteria profile)

formula = f'{dependent_var} ~ C(Group) + C(DiseasesClass) + C(Gender) + Age + C(antibiotics)'

model = smf.ols(formula=formula, data=data).fit()

print(model.summary())

group_coef_names = [name for name in model.params.index if 'Group' in name]
if group_coef_names:
    p_value = model.pvalues[group_coef_names[0]]
    print(f"\nGroup coefficient p-value: {p_value}")
    
    print("All Group coefficients p-values:")
    for name in group_coef_names:
        print(f"  {name}: {model.pvalues[name]}")
else:
    print("No Group coefficient found in the model")
