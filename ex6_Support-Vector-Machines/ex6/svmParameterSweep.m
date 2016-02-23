function [cost, C_matrix, sigma_matrix] = svmParameterSweep(...
    X, Y, C_list, sigma_list, ...
    Xval, Yval, kernelFunction)

if nargin<7
    kernelFunction = @gaussianKernel;
end

C_matrix = repmat(C_list(:), 1, numel(sigma_list));
sigma_matrix = repmat(sigma_list(:), 1, numel(C_list))';
cost = nan(numel(C_list), numel(sigma_list));

for iC = 1:numel(C_list)
    for iSigma = 1:numel(sigma_list)
        C = C_matrix(iC, iSigma);
        sigma = sigma_matrix(iC, iSigma);
        model = svmTrain(X, Y, C, @(x1, x2) kernelFunction(x1, x2, sigma));
        pred = svmPredict(model, Xval);
        cost(iC, iSigma) = mean(pred ~= Yval);
        fprintf('For C=%.4f, sigma=%.4f, the cost was %.4f\n', ...
            C, sigma, cost(iC, iSigma));
    end
end

end