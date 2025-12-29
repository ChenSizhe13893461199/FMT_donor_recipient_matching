%Function of PCA
function [Y, explained, selected_features] = optimize_microbiome_pca(data_matrix, var_threshold)
    variances = var(data_matrix);
    threshold = quantile(variances, var_threshold);
    selected_features = variances > threshold;
    data_filtered = data_matrix(:, selected_features);
    
    pseudo_count = 0.5 * min(data_filtered(data_filtered>0));
    data_pseudo = data_filtered + pseudo_count;
    geo_mean = exp(mean(log(data_pseudo), 2));
    data_clr = log(data_pseudo ./ geo_mean);
    
    data_centered = data_clr - mean(data_clr);
    
    cov_matrix = cov(data_centered);
    reg_param = 0.01 * trace(cov_matrix) / size(cov_matrix,1);
    cov_reg = cov_matrix + reg_param * eye(size(cov_matrix,1));

    [V, D] = eig(cov_reg);
    [eigenvals, idx] = sort(diag(D), 'descend');
    V_sorted = V(:, idx);
    
    score = data_centered * V_sorted;
    
    Y = score(:, 1:2);
    
    total_variance = sum(eigenvals);
    explained = 100 * eigenvals(1:2) / total_variance;
    
    fprintf('PC1 explained rate：%.1f%%\n', explained(1));
    fprintf('PC2 explained rate：%.1f%%\n', explained(2));
end
