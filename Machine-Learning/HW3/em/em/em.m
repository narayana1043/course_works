%--------------------------------------------------------------------------
% Expectation-Maximization Algorithm, demo
% 2-D problem
%
% Predrag Radivojac
% Indiana University
% January, 2014 
%--------------------------------------------------------------------------

clear
clc

% seed the random number generator so that the experiment is repeatable
randn('state', 12345)

%% Parameters

M = 2; % mixture of M Gaussians, M is assumed to be known (must stay 2 for K-means to work)
K = 2; % K-dimensional data (K must stay 2, otherwise visualization breaks)

n = [250 750]; % number of data points (per component)

% below are four different parameter sets for data set generation (one must uncomment)
%m = [[1 2]' [6 3]'];     % true means
%S(:, :, 1) = [1 0; 0 1]; % true covariance matrices
%S(:, :, 2) = [1 0; 0 1]; 

m = [[4 4]' [6 3]'];           % true means
S(:, :, 1) = [1 0.75; 0.75 1]; % true covariance matrices
S(:, :, 2) = [1 0.75; 0.75 1]; 

%m = [[4 4]' [6 3]'];             % true means
%S(:, :, 1) = [1 -0.75; -0.75 1]; % true covariance matrices
%S(:, :, 2) = [1 0.75; 0.75 1]; 

%m = [[1 2]' [6 3]'];           % true means
%S(:, :, 1) = [1 0; 0 1];       % true covariance matrices
%S(:, :, 2) = [1 0.75; 0.75 1]; 

%% Data set
% X is observable data
% y_true is generated, but is unobserved below

X = [];
y_true = [];
for k = 1 : M
    X = [X; repmat(m(:, k)', n(k), 1) + randn(n(k), 2) * chol(S(:, :, k))];
    y_true = [y_true; (k - 1) * ones(n(k), 1)];
end

% visualize true data
plot_clusters(X, y_true, M, 'Truth')
pause


%% K-Means algorithm
algorithm = 'K-means algorithm';

display(algorithm);

eps = 0.000001; % tolerance level
max_step = 100; % maximum number of iterations

step = 0;
diff = 2 * eps; % just to ensure it is greater than eps
sse = 0;

% initial centroids
mu = [mean(X, 1)' mean(X, 1)' + [0.01 -0.01]']; 

while diff > eps && step < max_step
    sse_old = sse;
    mu_old = mu;
    
    % calculate proximity of each data point to each of the M = 2 centroids
    for i = 1 : size(X, 1)
        if (X(i, :)' - mu(:, 1))'* (X(i, :)' - mu(:, 1)) <= (X(i, :)' - mu(:, 2))'* (X(i, :)' - mu(:, 2))
            y(i, 1) = 1;
            py(i, :) = [1 0];
        else
            y(i, 1) = 2;
            py(i, :) = [0 1];
        end
    end

    mu(:, 1) = mean(X(y == 1, :));
    mu(:, 2) = mean(X(y == 2, :));
    
    % visualize this situation
    plot_clusters_prob(X, py, algorithm, step);
    pause
    
    sse = sum(sum((X(y == 1, :) - repmat(mu(:, 1)', size(X(y == 1, :), 1), 1)) .^ 2));
    sse = sse + sum(sum((X(y == 2, :) - repmat(mu(:, 2)', size(X(y == 2, :), 1), 1)) .^ 2));

    diff = abs(sse - sse_old);
    
    step = step + 1;
end

w = sum(py, 1) / size(X, 1)
mu

pause


%% Classification EM algorithm
algorithm = 'Classification EM algorithm';

display(algorithm);

eps = 0.000001; % tolerance level
max_step = 100; % maximum number of iterations

% initialize w, mu, and sg
w = [0.5 0.5];
mu = [mean(X, 1)' mean(X, 1)' + [0.01 -0.01]']; 
%mu = [mean(X)' mean(X)' + [0.01 -0.01]']; % use the overall mean (+ eps)
%mu = [[4 3]' [6 4]']; % pick some 'random' means
%[~, mu] = kmeans(X, M); mu = mu'; % use K-means to find initial clusters
sg(:, :, 1) = diag(ones(M, 1));
sg(:, :, 2) = diag(ones(M, 1));

% initialize "class labels" y
y = zeros(size(X, 1), 1);

step = 0;
diff = 2 * eps; % just to ensure it is greater than eps

while diff > eps && step < max_step
    %w
    %mu
    %sg
    
    y_old = y;
    
    % given w, mu, and sg, update "class labeles" y
    for i = 1 : size(X, 1)
        Z = 0;
        for k = 1 : M
            t = X(i, :)' - mu(:, k);
            Z = Z + w(k) / sqrt((4*pi) ^ K * det(sg(:, :, k))) * exp(-0.5 * (t') * inv(sg(:, :, k)) * t);
        end
        
        for k = 1 : M
            t = X(i, :)' - mu(:, k);
            py(i, k) = w(k) / sqrt((4*pi) ^ K * det(sg(:, :, k))) * exp(-0.5 * (t') * inv(sg(:, :, k)) * t) / Z;
        end
        [~, q] = max(py(i, :));
        y(i) = q - 1;
    end

    % given "class labeles" y, update w, mu, and sg
    sg = zeros(K, K, M);
    for k = 1 : M
        w(k) = length(find(y == k - 1)) / length(y);
        q = find(y == k - 1);
        mu(:, k) = (mean(X(q, :)))';
        
        for i = 1 : length(q)
            sg(:, :, k) = sg(:, :, k) + (X(q(i), :)' - mu(:, k)) * (X(q(i), :)' - mu(:, k))';
        end
        sg(:, :, k) = sg(:, :, k) / length(q);
    end

    % visualize current situation
    plot_clusters(X, y, M, algorithm, step);
    pause

    
    % calculate the difference between this and previous iteration, but
    % skip the calculation if we are in the first iteration
    if step == 0
        diff = 2 * eps;
    else
        diff = sum(abs(y - y_old)) / size(X, 1);
    end
    
    step = step + 1;
end

w
mu
sg

%% EM algorithm
algorithm = 'EM algorithm';

display(algorithm);

eps = 0.000001; % tolerance level
max_step = 100; % maximum number of iterations

% initialize w, mu, and sg
w = [0.5 0.5];
%mu = [[4 3]' [6 4]']; % pick some 'random' means
%mu = [mean(X)' mean(X)' + [0.01 -0.01]']; % use the overall mean (+ eps)
%[~, mu] = kmeans(X, M); mu = mu'; % use K-means to find initial clusters
mu = [mean(X, 1)' mean(X, 1)' + [0.01 -0.01]']; 
sg(:, :, 1) = diag(ones(M, 1));
sg(:, :, 2) = diag(ones(M, 1));

step = 0;
diff = 2 * eps; % just to ensure it is greater than eps

while diff > eps && step < max_step
    w_old = w;
    mu_old = mu;
    sg_old = sg;
    
    % given w, mu, and sg, update "class posterior probabilities" py
    for i = 1 : size(X, 1)
        Z = 0;
        for k = 1 : M
            t = X(i, :)' - mu(:, k);
            Z = Z + w(k) / sqrt((4*pi) ^ K * det(sg(:, :, k))) * exp(-0.5 * (t') * inv(sg(:, :, k)) * t);
        end
        for k = 1 : M
            t = X(i, :)' - mu(:, k);
            py(i, k) = w(k) / sqrt((4*pi) ^ K * det(sg(:, :, k))) * exp(-0.5 * (t') * inv(sg(:, :, k)) * t) / Z;
        end
    end

    % given "class posteriors" py, update w, mu, and sg
    sg = zeros(K, K, M);
    for k = 1 : M
        w(k) = mean(py(:, k), 1);

        mu(:, k) = zeros(K, 1);
        for i = 1 : size(X, 1)
            mu(:, k) = mu(:, k) + py(i, k) * X(i, :)';
        end
        mu(:, k) = mu(:, k) / sum(py(:, k), 1);
        
        for i = 1 : size(X, 1)
            sg(:, :, k) = sg(:, :, k) + py(i, k) * (X(i, :)' - mu(:, k)) * (X(i, :)' - mu(:, k))';
        end
        sg(:, :, k) = sg(:, :, k) / sum(py(:, k), 1);
    end

    % visualize this situation
    plot_clusters_prob(X, py, algorithm, step);
    pause
    
    diff = sum(abs(w - w_old)) + sum(sum(abs(mu - mu_old))) + sum(sum(sum(abs(sg - sg_old))));
    
    step = step + 1;
end

w
mu
sg



