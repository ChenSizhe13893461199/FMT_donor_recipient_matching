% ANOVA-based Multiple Linear Regression Codes with multi-confounders adjustment (This script can only be used for comparing convergent vs non-convergent pairs)

data = readtable('Source Data File.xlsx');
data.Group = categorical(data.(data.Properties.VariableNames{43}));  % Categorical Variable (convergent vs non-convergent pairs)

data.DiseasesClass = categorical(data.DiseasesClass); % Categorical Variable

%DiseasesClass (1: Crohn’s Disease; 2: Tourette_syndrome; 3: CDI; 4: Antibiotics Resistsance; 5: Melanoma; 6: healthy volunteer;
%7: ulcerative colitis; 8: IBS; 9: renal_carcinoma; 10: metabolic syndrome; 11: Diabetes; 12: Obesity; 13: PACS)

data.Gender = categorical(data.Gender); % Categorical Variable

% ANOVA-based Multiple Linear Regression
model = fitlm(data, ...
    sprintf('%s ~ Group + DiseasesClass+Gender+Age+antibiotics', data.Properties.VariableNames{29}), ...
    'CategoricalVars', {'Group', 'DiseasesClass','Gender','antibiotics'}); 
    
%Here 29 represents the 29th column (pre-FMT recipient Shannon); Please replace it with other numbers if further analysis for other indexes are needed
disp(model);
group_coef_name = model.CoefficientNames{contains(model.CoefficientNames, 'Group')};
p_value = model.Coefficients{group_coef_name, 'pValue'};
p_value
