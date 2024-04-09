function [Z, IDF] = IDF(X)
    % IDF: inverse document frequency.
    % X: tensor, 1*D, or 1*1*D or 1*1*1*D...
    % IDF: tensor, 1*D, or 1*1*D or 1*1*1*D...
    % Z: tensor, 1*D, or 1*1*D or 1*1*1*D...
    % Authors: F. Lu, 2021. 
    
    epsilon = 1e-8;
    n = ndims(X);
    DF = abs(X);
    IDF = log((sum(DF, n) + epsilon) ./ (DF + 1));
    Z = X .* IDF;
end