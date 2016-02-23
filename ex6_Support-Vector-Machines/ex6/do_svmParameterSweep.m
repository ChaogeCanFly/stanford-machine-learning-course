
load('ex6data3.mat');

C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

[cost, C_matrix, sigma_matrix] = svmParameterSweep(...
    dataset.X, dataset.y, C_list, sigma_list, ...
    dataset.Xval, dataset.yval);

%%
[cost_min, cost_min_idx] = min(cost(:));
[best_idx_C, best_idx_sigma] = ind2sub(size(cost), cost_min_idx);
best_C = C_list(best_idx_C);
best_sigma = sigma_list(best_idx_sigma);
fprintf('Best C=%.4f, sigma=%.4f, cost=%.4f\n', best_C, best_sigma, cost_min);

%%
figure;
imagesc(C_list, sigma_list, cost);
colormap(hot);
colorbar;