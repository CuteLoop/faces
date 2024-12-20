%----------------------------------------------------------------
% File:     main_fitcecoc_ten_persons.m
%----------------------------------------------------------------
%
% Author:   [Your Name]
% Date:     [Current Date]
% Copying:  (C) [Your Name], [Year]. All rights reserved.
% 
%----------------------------------------------------------------
% Multi-class Classification
% Distinguish between ten specific persons with >5 and <60 pictures each.
%----------------------------------------------------------------

% Clear environment and close all figures
clear;
close all;
clc;

% Define parameters
targetSize = [128, 128];               % Image resizing dimensions
k = 60;                                 % Number of features to consider
location = fullfile('..','lfw');             % Path to the image dataset
svd_cache = fullfile('cache', 'svd.mat'); % Path to save SVD results (optional)

% Define the list of 10 specific persons
selectedPersons = {...
    'Abdullah_Gul', ...
    'Adrien_Brody', ...
    'Al_Gore', ...
    'Al_Sharpton', ...
    'Albert_Costa', ...
    'Alejandro_Toledo', ...
    'Ali_Naimi', ...
    'Alvaro_Uribe', ...
    'Amelia_Vega', ...
    'Amelie_Mauresmo'};

% Display progress
disp('Creating image datastore...');
% Create an image datastore with grayscale images resized to targetSize
imds0 = imageDatastore(location, ...
                       'IncludeSubfolders', true, ...
                       'LabelSource', 'foldernames', ...
                       'ReadFcn', @(filename) imresize(im2gray(imread(filename)), targetSize));

% Display progress
disp('Filtering selected persons with >5 and <60 images...');
% Count images per person
labelCounts = countEachLabel(imds0);

%
validPersons = selectedPersons
% Update the list if some persons are invalid

% Create a subset datastore containing only the selected persons
mask = ismember(imds0.Labels, validPersons);
idx = find(mask);
imds = subset(imds0, idx);

% Display a montage of the selected images and save it with 'ten_' prefix
disp('Creating and saving montage of selected individuals...');
figure;
montage(imds, 'Size', [ceil(length(imds.Files)/10), 10]);
title('Montage of Selected Individuals');
saveas(gcf, 'ten_montage.png');  % Save montage image
close;

% Display progress
disp('Reading all images...');
% Read all images from the subset datastore
A = readall(imds);

% Concatenate images into a 3D array and reshape into a 2D matrix
B = cat(3, A{:});
D = prod(targetSize);
B = reshape(B, D, []);

% Display progress
disp('Normalizing data...');
% Normalize pixel values to [0,1]
B = single(B) ./ 255;

% Perform further normalization: subtract mean and divide by std
% Equivalent to MATLAB's normalize function with default settings
[B, C, SD] = normalize(B);

% Optionally, save the normalization parameters
% save('ten_normalization.mat', 'C', 'SD');

% Perform Singular Value Decomposition (SVD) on the normalized data
disp('Performing Singular Value Decomposition (SVD)...');
tic;
[U, S, V] = svd(B, 'econ');
toc;

% Generate a montage of the top 16 eigenfaces
disp('Creating and saving montage of top 16 eigenfaces...');
Eigenfaces = arrayfun(@(j) reshape((U(:, j) - min(U(:, j))) ./ (max(U(:, j)) - min(U(:, j))), targetSize), ...
                      1:16, 'UniformOutput', false);

figure;
montage(Eigenfaces, 'Size', [4, 4]);
title('Top 16 Eigenfaces');
colormap(gray);
saveas(gcf, 'ten_eigenfaces.png');  % Save eigenfaces image
close;

% Extract the first k eigenfaces for feature extraction
X0 = U(:, 1:k)' * B;  % Project data onto eigenfaces (k x N)
X = X0';              % Transpose to get feature vectors in rows (N x k)

% Extract labels for the subset
Y = imds.Labels;

% Convert labels to categorical if not already
if ~iscategorical(Y)
    Y = categorical(Y);
end

% Ensure 'cats' is a cell array of strings
cats = cellstr(categories(Y));

% Define a colormap for plotting (assign distinct colors for 10 classes)
cm = lines(10);  % Generates 10 distinct colors

% Assign colors based on labels
c = cm(double(Y), :);  % double(Y) converts categories to 1 to 10

% Display progress
disp('Training Support Vector Machine (SVM) with multi-class classification...');
% Set SVM options (optional: enable parallel computing if available)
options = statset('UseParallel', true);

% Train a multi-class Support Vector Machine (SVM) classifier using one-vs-one
tic;
Mdl = fitcecoc(X, Y, ...
              'Learners', 'svm', ...
              'Coding', 'onevsone', ...
              'Verbose', 1, ...
              'Options', options);
toc;

% Generate and save scatter plots for feature pairs
disp('Generating and saving scatter plots...');
figure;
t = tiledlayout(2,1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Scatter Plot 1: Features 1 vs 2
nexttile(t);
scatter(X(:,1), X(:,2), 60, c, 'filled');
title('Scatter Plot of Features 1 vs 2');
xlabel('Feature 1 (x1)');
ylabel('Feature 2 (x2)');
legend(cats, 'Location', 'bestoutside');
grid on;

% Scatter Plot 2: Features 3 vs 4
nexttile(t);
scatter(X(:,3), X(:,4), 60, c, 'filled');
title('Scatter Plot of Features 3 vs 4');
xlabel('Feature 3 (x3)');
ylabel('Feature 4 (x4)');
legend(cats, 'Location', 'bestoutside');
grid on;

% Save scatter plots
saveas(gcf, 'ten_scatter_plots.png');
close;

% Predict on training data
disp('Predicting on training data...');
[YPred, Score, Cost] = predict(Mdl, X);

% Generate and save ROC curves for multi-class
disp('Generating and saving ROC curves for each class...');
figure;
hold on;
auc = zeros(10,1);  % To store AUC for each class
for i = 1:10
    currentClass = cats{i};
    binaryLabel = (Y == currentClass);
    [Xroc, Yroc, Troc, AUC] = perfcurve(binaryLabel, Score(:,i), true);
    plot(Xroc, Yroc, 'Color', cm(i,:), 'LineWidth', 2, 'DisplayName', sprintf('%s (AUC = %.2f)', currentClass, AUC));
    auc(i) = AUC;
end
hold off;
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves for Each Class (One-vs-Rest)');
legend('Location', 'southeastoutside');
grid on;
saveas(gcf, 'ten_roc_metrics.png');  % Save ROC curves
close;

% Generate and save the confusion matrix
disp('Generating and saving confusion matrix...');
figure;
confChart = confusionchart(Y, YPred);
confChart.Title = 'Confusion Matrix for 10-Class Classification';
confChart.RowSummary = 'row-normalized';
confChart.ColumnSummary = 'column-normalized';
saveas(gcf, 'ten_confusion_matrix.png');  % Save confusion matrix
close;

% Save the trained SVM model and relevant data with 'ten_' prefix
disp('Saving the trained SVM model and related data...');
save('ten_model.mat', 'Mdl', 'selectedPersons', 'U', 'targetSize', 'k');

disp('All outputs have been saved with the prefix "ten_".');
