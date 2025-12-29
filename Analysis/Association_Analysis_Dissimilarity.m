%Association Analysis (Profile Dissimilarity vs Increased Profile Similarity)
data = readtable('Source Data File.xlsx');
increased = data.Overall_Increased_Similarity; % Increased Similarity in Microbiome Profiles
%increased = data.Pathway_increased_similarity; % Increased Similarity in Pathway Profiles
%increased = data.KO_increased; % Increased Similarity in KO Profiles

dissimilarity = data.Overall_Microbiome_Dissimilarity; % Dissimilarity between donor pre-FMT recipient in microbiome profile
%dissimilarity = data.Pathway_dissimilarity; % Dissimilarity between donor pre-FMT recipient in Pathway profile
%dissimilarity = data.KO_dissimilarity; % Dissimilarity between donor pre-FMT recipient in KO profile

valid_idx = ~isnan(increased) & ~isnan(dissimilarity);
increased = increased(valid_idx);
dissimilarity = dissimilarity(valid_idx);

% coefficient calculation and analysis
[corr_coef, p_value] = corr(increased, dissimilarity);

positive_mask = increased > 0;
negative_mask = increased < 0;

% Visualization
figure;
hold on;
scatter_size = 40;

scatter(increased(positive_mask), dissimilarity(positive_mask),...
    scatter_size, 'b', 'filled');
scatter(increased(negative_mask), dissimilarity(negative_mask),...
    scatter_size, 'r', 'filled');

coefficients = polyfit(increased, dissimilarity, 1);
x_fit = linspace(min(increased), max(increased), 100);
y_fit = polyval(coefficients, x_fit);
plot(x_fit, y_fit, 'k--', 'LineWidth', 1.5)

text(0.05, 0.95,...
    sprintf('R = %.2f\np = %.3f', corr_coef, p_value),...
    'Units', 'normalized',...
    'FontSize', 10,...
    'VerticalAlignment', 'top');
