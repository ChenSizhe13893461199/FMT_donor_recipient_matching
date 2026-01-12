% ANOVA-based Multiple Linear Regression Codes with multi-confounders adjustment

data = readtable('Source Data File.xlsx');
data = data(data.category_response_1_or_non_response_0_or_unknown_2 == 0 | data.category_response_1_or_non_response_0_or_unknown_2 == 1, :); % analyzing Responders & Non-Responders data
data.Group = categorical(data.(data.Properties.VariableNames{1}));  % Categorical Variable

data.DiseasesClass = categorical(data.DiseasesClass); % Categorical Variable

%DiseasesClass (1: Crohn’s Disease; 2: Tourette_syndrome; 3: CDI; 4: Antibiotics Resistsance; 5: Melanoma; 6: healthy volunteer;
%7: ulcerative colitis; 8: IBS; 9: renal_carcinoma; 10: metabolic syndrome; 11: Diabetes; 12: Obesity; 13: PACS)

data.Gender = categorical(data.Gender); % Categorical Variable

% ANOVA-based Multiple Linear Regression
model = fitlm(data, ...
    sprintf('%s ~ Group + DiseasesClass+Gender+Age+antibiotics', data.Properties.VariableNames{5}), ...
    'CategoricalVars', {'Group', 'DiseasesClass','Gender','antibiotics'}); 
    
%Here 5 represents the 5th column (bacteria profile), for fungi, virus, and archaea as well as overall microbiomne, please replace it with number 6, 7, 8, and 9, respectively
%This script can also be used for analyzing multiple indexes used in our study (see Source Data File.xlsx), and please replace the mentioned position with the
%corresponding number (e.g. the 14th column represents Overall_Microbiome_Dissimilarity, and it can be adjusted by the script to avoid influences from the confounding variables) to conduct analysis, respectively.
disp(model);
group_coef_name = model.CoefficientNames{contains(model.CoefficientNames, 'Group')};
p_value = model.Coefficients{group_coef_name, 'pValue'};
p_value
