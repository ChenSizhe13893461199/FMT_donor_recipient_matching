%%
% Using the codes below can reproduce the PCA analysis in our response letter to reviewers
% Read Input Data
data = readtable('microbiome.xlsx'); 
%1-515 rows represent features of pre-FMT recipinet, while 516-1030 and 1031-1545 represent features of post-FMT recipient and donor, respectively

% Filter Features
data_matrix = data{:,:};
diseases_class = labels.Response;
%PCA Method
[Y_optimized, explained_opt, selected] = optimize_microbiome_pca(data_matrix, 0.1); % Preserve top 90% features
Y = Y_optimized(:, 1:2);

% If t-SNE is applied, please use the annotated codes below 
% please note that the t-SNE results will be slightly different from each time of coding due to its intrinsic randomization mechanisms;
% The overall t-SNE results will maintain the same trends
%Y = tsne(data_matrix);
%Y = score(:,1:2);

% For PCA figure of different categories (Response/Non-Response/Unknown)
point=Y(516:1030,:)-Y(1:515,:); % Post-FMT Recipient - Pre-FMYT Recipient
%point=Y(516:1030,:)-Y(1031:1545,:); % Post-FMT Recipient - Donor
new_class=diseases_class(1:515,:);

%Creating colors for different groups
cmap1 = lines(6);
cmap6 = lines(7)+0.03;
cmap6(:,2)=cmap6(:,2)/1.5;
cmap = [cmap1; cmap6];

% Figure codes and analysis
figure
hold on;
for i = 1:3
    idx = (new_class == i-1);
    scatter(point(idx,1), point(idx,2), 40, 'filled', ...
        'MarkerFaceColor', cmap(i,:), ...
        'MarkerFaceAlpha', 0.6, ...
        'MarkerEdgeColor', cmap(i,:), ...
        'MarkerEdgeAlpha', 0.6);
end

% calculating center point
centers = zeros(2,2);
for i = 1:2
    centers(i,:) = mean(point(new_class == i-1, :), 1);
end

% drawing the arrows
for i = 2:-1:1
    quiver(0, 0, centers(i,1), centers(i,2), ...
        'AutoScale', 'off', ...
        'Color', cmap(i,:), ...
        'LineWidth', 1.5, ...
        'MaxHeadSize', 6);
end



% For figure of different diseases categories
diseases_class = labels.Diseases_Class;
figure;
hold on;
for i = 1:13
    idx = (new_class == i);
    scatter(point(idx,1), point(idx,2), 40, 'filled', ...
        'MarkerFaceColor', cmap(i,:), ...
        'MarkerFaceAlpha', 0.3, ...
        'MarkerEdgeColor', cmap(i,:), ...
        'MarkerEdgeAlpha', 0.6);
end

% calculating center point
centers = zeros(13,2);
for i = 1:13
    centers(i,:) = mean(point(new_class == i, :), 1);
end

% For figure of different diseases categories
for i = 1:13
    quiver(0, 0, centers(i,1), centers(i,2), ...
        'AutoScale', 'off', ...
        'Color', cmap(i,:), ...
        'LineWidth', 1.5, ...
        'MaxHeadSize', 1);
end
