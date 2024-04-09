function features_pca = PCA_whitening(features, train, dim)
%% method 1
%     features = normalize(features, 2, 'norm');
%     train = normalize(train, 2, 'norm');
% 
%     x = train';
% 
%     avg = mean(x, 1);
% 
%     x = x - repmat(avg, size(x,1), 1);
% 
%     x(isnan(x)) = 0;
% 
%     sigma = (x*x')/size(x,2);
% 
%     [U,~,~] = svd(sigma);
%     
%     y = features';
% 
%     avg_y = mean(y, 1);
% 
%     y = y - repmat(avg_y, size(y,1), 1);
% 
%     y(isnan(y)) = 0;
% 
%     Xpca = U(:,1:dim)' * y; 
% 
%     sigma = (Xpca * Xpca')/size(Xpca,2);
% 
%     [u,s,~] = svd(sigma);
% 
%     xRot = u'* Xpca;
% 
%     epsilon = 1e-5;
% 
%     xPCAWhite = diag(1./(sqrt(diag(s)+epsilon)))*xRot;
% 
%      x2 = xPCAWhite';
% 
%     features_pca = normalize(x2, 2, 'norm');

%% method 2

    x_test  = normalize(features, 2, 'norm');
    x_train = normalize(train, 2, 'norm');
     
    [coeff, scoreTrain, ~, ~, ~, mu] = pca(x_train);   % PCA training
%     x_trainPCA = scoreTrain(:, 1:dim);   % train PCA, optional
%     x_train = scoreTrain*coeff'+mu;      % the relation of the four
    sigma = scoreTrain' * scoreTrain / size(scoreTrain, 1);
    sigma(isnan(sigma)) = 0;
    [~,s,~] = svd(sigma);   % [u,s,v] = svd(sigma), u is coeff
    
    type = 1;       % type 1 and 2 is equal.
    if type == 1
        x_testRot = (x_test - mu) * coeff;   % PCA apply type 1
    %     x_testPCA = x_testRot(:, 1:dim);   % test PCA, optional
        epsilon = 1e-5;
        p = 1/3;
        x_testPCAWhite = x_testRot * diag(1 ./ ((diag(s) + epsilon)).^p);    % whiten apply
        features_data = x_testPCAWhite(:,1:dim);
    end
    
    if type == 2
        x_testRot = (x_test - mu);   % PCA apply type 2
        epsilon = 1e-5;
        p = 1/3;
        params = coeff * diag(1 ./ ((diag(s) + epsilon)).^p);       % learn whiten parameters
        x_testPCAWhite = x_testRot * params;    % whiten apply
        features_data = x_testPCAWhite(:,1:dim);
    end
    
    features_pca = im_cross_normalize(features_data);

end