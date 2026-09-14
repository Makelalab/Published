function [mu_est, sigma_est, p_est, counter, difference] = ...
    gmm_fixed_param(values, C, fixed_means, epsilon)
% Estimate the parameters of a 1D Gaussian Mixture model using EM.

%   Inputs:
%      values  = row vector of the observed values
%                   note this only works for 1D data
%      C       = number of classes (mixtures), which should be fairly small
%      fixed_means = define which means are fixed for fitting e.g. [NaN 0.3]         
%      epsilon = precision for convergence
%
%   Outputs:
%      mu_est    = vectors means of each class
%      sigma_est = vector of standard deviations of each class
%      p_est     = class membership probability estimates
%      counter   = number of iterations required
%      difference= total absolute difference in parameters at each iteration to get
%                  an idea of convergence rate
%
%   A Gaussian mixture model means that each data point is drawn (randomly) from one of C
%   classes of data, with probability p_i of being drawn from class i, and each class is
%   distributed as a Gaussian with mean standard deviation mu_i and sigma_i.
%
%   The algorithm used here for estimation is EM (Expectation Maximization). Simply put, if
%   we knew the class of each of the N input data points, we could separate them, and use
%   Maximum Likelihood to estimate the parameters of each class. This is the M step. The E
%   step makes (soft) choises of (unknown) classes for each of the data points based on the
%   previous round of parameter estimates for each class.
%

if (nargin < 4)
  epsilon = 1.0e-4;
end

if length(fixed_means) ~= C
    disp('fixed_means does not match C')
end

% initialize
counter = 0;
mu_est = [];
for ii = 1:C
    if isnan(fixed_means(ii))
        mu_est(ii) = mean(values) * sort(rand(1,1));
    else
        mu_est(ii) = fixed_means(ii);
    end
end

sigma_est = ones(C,1)*std(values);
p_est = ones(C,1)/C;
difference = epsilon;

% iterate until each iteration does not change significantly
while (difference >= epsilon & counter < 25000)
    % [mu_est, sigma_est, p_est]

    % E step: classification of the values into one of the mixtures 
    for jj = 1:C
        class(jj, :) = p_est(jj)*norm_density(values, mu_est(jj), sigma_est(jj));
    end

    % normalize
    class = class ./ repmat(sum(class), C, 1);

    % M step: ML estimate the parameters of each class (i.e., p, mu, sigma)
    mu_est_old = mu_est;
    sigma_est_old = sigma_est;
    p_est_old = p_est;

    % update the parameters of the distributions according to the new labeled dataset
    for jj = 1:C
        %  check if mu is fixed or not
        if isnan(fixed_means(jj))
            mu_est(jj) = sum( class(jj,:).*values ) / sum(class(jj,:));
        end
        sigma_est(jj) = sqrt( sum(class(jj,:).*(values - mu_est(jj)).^2) / sum(class(jj,:)) );
        p_est(jj) = mean(class(jj,:));
    end

    % estimate the speed of convergence
    difference(counter+1) = sum(abs(mu_est_old - mu_est)) + ...
	sum(abs(sigma_est_old - sigma_est)) + ...
	sum(abs(p_est_old - p_est));

    counter = counter + 1;
end
