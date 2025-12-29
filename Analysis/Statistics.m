% ANOVA-based Multiple Linear Regression Codes

data = readtable('Source Data File.xlsx');
data = data(data.category == 0 | data.category == 1, :); % Responders & Non-Responders data
data.Group = categorical(data.(data.Properties.VariableNames{1}));  % Categorical Variable
data.DiseasesClass = categorical(data.DiseasesClass); % Categorical Variable                 
data.Gender = categorical(data.Gender); % Categorical Variable

% ANOVA-based Multiple Linear Regression
model = fitlm(data, ...
    sprintf('%s ~ Group + DiseasesClass+Gender+Age+antibiotics', data.Properties.VariableNames{5}), ...
    'CategoricalVars', {'Group', 'DiseasesClass','Gender','antibiotics'}); 
    
%Here 5 represents the 5th column (bacteria profile), for fungi, virus, and archaea as well as overall microbiomne, please replace it with number 6, 7, 8, and 9, respectively

disp(model);

group_coef_name = model.CoefficientNames{contains(model.CoefficientNames, 'Group')};
p_value = model.Coefficients{group_coef_name, 'pValue'};
p_value
