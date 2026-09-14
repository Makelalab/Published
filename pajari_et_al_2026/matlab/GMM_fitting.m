%% Measure D for each molecule

% get *csv files
[filename,pathname] = uigetfile('*.csv', 'Select','Select','MultiSelect', 'on');
if ~iscell(filename)
    name = filename;
    filename = cell(1);
    filename{1} = name;
end

D = cell(1,10000);      % pre-allocate variable
no_cells = 0;
ind = 1;

D_single = [];
for hh = 1:length(filename)   
    A = readtable([pathname filename{hh}]);
    D_single = A.Dapp;
    data = A.cell_id_int;
    uniquE = unique(data);
    numUni = length(uniquE);
    no_cells = no_cells + numUni;

    D{ind} = D_single;
    ind = ind + 1;
end
% convert cell into a vector
D = cat(1, D{:});

disp(['Number of cells = ' num2str(no_cells)])
disp(['Mean D* value = ' num2str(mean(D))])
disp(['Number of tracks = ' num2str(length(D))])

% plot distribution on logarithmic axis
no_bins = 60;
edges = linspace(log10(min(D)), log10(max(D)), no_bins+1);
[N, X] = hist(D, 10.^edges); % Compute histogram data with custom bin edges

N = N./((sum(N)*(log10(X(2))-log10(X(1))))); 

figure;
bar(log10(X),N,'BarWidth',1,'FaceColor',"red", 'EdgeColor', 'none');
xlabel('log_{10}(D*)(\mum^2/s)')
ylabel('Probability density')
axis([-2.0 2 0 2.0])
hold on;

% ksdensity of the data
bandwidthValue = 0.075;
[f,xi] = ksdensity(log10(D),'Bandwidth', bandwidthValue);

% Plot the kernel density estimate
%plot(xi,f,'b','LineWidth', 2);

axis([-2.0 2.0 0 1.5])

%% EM fit of gaussian mixture model
% This fits N-Gaussian mixture model where for each gaussian location, std,
% and weight are parameters to be fitted
% Uses expectation maximization algorithm for fitting model to data
% Note! Initiation of search is random and can give different local maxima

% run previous section
positions = [NaN NaN NaN]; 	    % Number and locations of peaks. NaN is used for unknown location of peak. 
                            % E.g. [0.1 NaN NaN] is a 3 gaussian mixture with a fixed peak at 0.1 and two freely localized peaks
x = -2.0:0.01:2;            % range of values plotted for GMM fit
low_perc = 0.01;           % percentage of smallest values ignored in fitting

% convert D values into logarithmic space for fitting
values = log10(D');         

% remove smallest X% for better fitting
lower = quantile(values, low_perc);
values = values(values > lower);
x;
% fit GMM
[mu_est, sigma_est, w_est, counter, difference] = gmm_fixed_param(values, length(positions), positions, 1.0e-3);

% plot estimated distributions
% figure;
hold on;
p_est_all = [];
for ii = 1:length(mu_est)
    p_est_single = w_est(ii) * norm_density(x, mu_est(ii), sigma_est(ii));
    plot(x, p_est_single, 'k--', 'linewidth', 2);
    p_est_all = [p_est_all; p_est_single];
    disp(['Mean value ' num2str(mu_est(ii)) ' - weight ' num2str(w_est(ii))])
end
% sort GMM classes from slowest diffusion to largest (class 1 bound)
[mu_est,I] = sort(mu_est);
sigma_est = sigma_est(I);
w_est = w_est(I);

plot(x, sum(p_est_all), 'k-', 'linewidth', 2);
title(['Mean value ' num2str(mu_est) ', weight ' num2str(w_est')])
axis([-2.0 2.0 0 1.5])
yticks([0, 0.5, 1.0, 1.5, 2.0]);

disp(10.^([mu_est]));
mu_log = 10.^([mu_est])

save('GMM_model.mat','mu_est','sigma_est','w_est')
 

