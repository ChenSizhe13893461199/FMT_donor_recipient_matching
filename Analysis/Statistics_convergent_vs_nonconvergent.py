# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:30:36 2026

@author: CHEN Sizhe
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices
import re

data = pd.read_excel('Source Data File.xlsx')

group_col_name = data.columns[42]  # here 42 represents the 43th column, represents convergent or non-convergent classification
data['Group'] = pd.Categorical(data[group_col_name]) 

data['DiseasesClass'] = pd.Categorical(data['DiseasesClass'])
data['Gender'] = pd.Categorical(data['Gender'])

# DiseasesClass：
# 1: Crohn's Disease; 2: Tourette_syndrome; 3: CDI; 4: Antibiotics Resistance;
# 5: Melanoma; 6: healthy volunteer; 7: ulcerative colitis; 8: IBS; 
# 9: renal_carcinoma; 10: metabolic syndrome; 11: Diabetes; 12: Obesity; 13: PACS

data['antibiotics'] = pd.Categorical(data['antibiotics'])

def clean_column_name(col_name):
    if isinstance(col_name, str):
        col_name = col_name.replace(' ', '_')
        col_name = re.sub(r'[^a-zA-Z0-9_]', '', col_name)
    return col_name

column_mapping = {col: clean_column_name(col) for col in data.columns}
data = data.rename(columns=column_mapping)

# ANOVA-based Multiple Linear Regression
dv_col_name = list(column_mapping.values())[28] # here the 28 represents the 29th column pre-FMT recipient Shannon,

formula = f'{dv_col_name} ~ Group + DiseasesClass + Gender + Age + antibiotics'

model = smf.ols(formula=formula, data=data).fit()

print(model.summary())

group_p_values = []
for name in model.params.index:
    if 'Group' in name:
        p_val = model.pvalues[name]
        group_p_values.append((name, p_val))
        print(f'{name}: p-value = {p_val:.6f}')
#It should be 0.000686 or 0.00069 in this example
# 
if group_p_values:
    print(f'\np value is: {group_p_values[0][1]:.6f}')


anova_table = sm.stats.anova_lm(model, typ=2)
print("\nTable:")
print(anova_table)
