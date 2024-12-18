%----------------------------------------------------------------
% File:     main_fitcecoc.m
%----------------------------------------------------------------
% Author:   Marek Rychlik
% Date:     Fri Nov 22 20:02:05 2024
%----------------------------------------------------------------

% Image preprocessing parameters
targetSize = [128,128]; % Resize images to 128x128
k = 40; % Number of features (eigenfaces) to consider
location = fullfile('lfw'); % Image dataset location

% Global Plot Settings
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultLineLineWidth', 1.5);
set(groot, 'DefaultFigurePosition', [100, 100, 1200, 800]);

%% Step 1: Load and preprocess images
disp('Creating image datastore...');
imds0 = imageDatastore(location, ...
                       'IncludeSubfolders', true, ...
                       'LabelSource', 'foldernames', ...
                       'ReadFcn', @(filename) imresize(im2gray(imread(filename)), targetSize));

% Subset images: persons with 10-40 images
tbl = countEachLabel(imds0);
mask = tbl{:,2} >= 10 & tbl{:,2} <= 40;
persons = unique(tbl{mask,1});

% Filter dataset
[lia, locb] = ismember(imds0.Labels, persons);
imds = subset(imds0, lia);

% Display and save selected images
figure('Position', [100, 100, 1200, 600]);
montage(imds);
title('Selected Dataset Images');
saveas(gcf, 'dataset_images.png'); % Save montage

%% Step 2: Data normalization and PCA
disp('Performing PCA...');
A = readall(imds);
B = cat(3, A{:}); 
B = reshape(B, prod(targetSize), []);
B = single(B) ./ 256;

[B, ~, ~] = normalize(B);
[U, S, V] = svd(B, 'econ');

% Display and save top 16 eigenfaces
Eigenfaces = arrayfun(@(j) mat2gray(reshape(U(:,j), targetSize)), 1:16, 'uni', false);
figure('Position', [100, 100, 1200, 600]);
montage(Eigenfaces, 'Size', [4, 4], 'BorderSize', [5, 5], 'BackgroundColor', 'white');
title('Top 16 Eigenfaces');
saveas(gcf, 'eigenfaces.png');

%% Step 3: Train multi-class SVM
disp('Training SVM classifier...');
W = S * V';
W = W(1:k, :);
U = U(:, 1:k);
X = W';
Y = categorical(imds.Labels, persons);

options = statset('UseParallel', true);
Mdl = fitcecoc(X, Y, 'Verbose', 2, 'Learners', 'svm', 'Options', options);

%% Step 4: Visualize and save results
% Scatter plot for top features
figure('Position', [100, 100, 1200, 600]);
scatter3(X(:,1), X(:,2), X(:,3), 50, uint8(Y), 'filled', 'MarkerFaceAlpha', 0.6);
title('Feature Space: Top 3 Features');
xlabel('x1'); ylabel('x2'); zlabel('x3');
grid on;
saveas(gcf, 'feature_space.png');

% ROC metrics with legend moved under the diagonal
[YPred, Score] = resubPredict(Mdl);
numClassesToShow = 10; % Limit classes
randomIdx = randperm(numel(persons), numClassesToShow);
selectedClasses = persons(randomIdx);
filteredScores = Score(:, randomIdx);
rm = rocmetrics(Y, filteredScores, selectedClasses);

figure('Position', [100, 100, 1200, 600]);
plot(rm, 'ShowLegend', true, 'LineWidth', 1.5);
legend('Location', 'southeast'); % Legend under the diagonal
title('ROC Curve (Subset of Classes)');
saveas(gcf, 'roc_metrics.png');

% Simplified confusion matrix
figure('Position', [100, 100, 1200, 600]);
subsetClasses = selectedClasses(1:5); % Only compare 5 classes
subsetMask = ismember(Y, subsetClasses);
confusionchart(Y(subsetMask), YPred(subsetMask), ...
    'Title', 'Confusion Matrix (Subset of 5 Classes)', ...
    'FontSize', 10, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
saveas(gcf, 'confusion_matrix.png');

%% Step 5: Save model
save('model.mat', 'Mdl', 'persons', 'U', 'targetSize');
disp('Model and outputs saved successfully.');
