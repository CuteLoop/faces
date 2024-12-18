%----------------------------------------------------------------
% File:     main_fitcecoc.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Fri Nov 22 20:02:05 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% Classification into several classes using PCA and SVM
% This script trains a facial recognition model, extracts features
% via PCA, and classifies images using a multi-class SVM.
%----------------------------------------------------------------

% Image preprocessing parameters
targetSize = [128,128]; % Resize images to 128x128
k = 40; % Number of features (eigenfaces) to consider
location = fullfile('lfw'); % Image dataset location

% Global Plot Settings
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultLineLineWidth', 1.5);
set(groot, 'DefaultFigurePosition', [100, 100, 800, 600]);

%% Step 1: Load and preprocess images
disp('Creating image datastore...');
imds0 = imageDatastore(location, ...
                       'IncludeSubfolders', true, ...
                       'LabelSource', 'foldernames', ...
                       'ReadFcn', @(filename) imresize(im2gray(imread(filename)), targetSize));

% Subset images: persons with 10-40 images
tbl = countEachLabel(imds0);
mask = tbl{:,2} >= 10 & tbl{:,2} <= 40;
disp(['Number of images: ', num2str(sum(tbl{mask,2}))]);
persons = unique(tbl{mask,1});

% Filter dataset
[lia, locb] = ismember(imds0.Labels, persons);
imds = subset(imds0, lia);

% Display and save selected images
figure;
montage(imds);
title('Selected Dataset Images');
saveas(gcf, 'dataset_images.png'); % Save montage of dataset images

%% Step 2: Data normalization and PCA
disp('Reading all images...');
A = readall(imds);
B = cat(3, A{:}); % Stack images
B = reshape(B, prod(targetSize), []); % Flatten images
B = single(B) ./ 256; % Normalize pixel values

disp('Performing PCA using SVD...');
tic;
[B, ~, ~] = normalize(B); % Normalize the dataset
[U, S, V] = svd(B, 'econ'); % Singular Value Decomposition
toc;

% Display and save top 16 eigenfaces
Eigenfaces = arrayfun(@(j) reshape(U(:,j), targetSize), 1:16, 'uni', false);
figure;



% Get an montage of eigenfaces
Eigenfaces = arrayfun(@(j)reshape((U(:,j)-min(U(:,j)))./(max(U(:,j))-min(U(:,j))),targetSize), ...
    1:size(U,2),'uni',false);
montage(Eigenfaces(1:16));
title('Top 16 Eigenfaces');
saveas(gcf, 'eigenfaces.png'); % Save top 16 eigenfaces

%% Step 3: Train multi-class SVM
disp('Training SVM classifier...');
W = S * V';
W = W(1:k, :); % Keep first k features
U = U(:, 1:k); % Top k eigenfaces
X = W'; % Features matrix
Y = categorical(imds.Labels, persons); % Labels

options = statset('UseParallel', true);
Mdl = fitcecoc(X, Y, 'Verbose', 2, 'Learners', 'svm', 'Options', options);

%% Step 4: Visualize and save results
% Scatter plot for top features
figure;
scatter3(X(:,1), X(:,2), X(:,3), 50, uint8(Y), 'filled', 'MarkerFaceAlpha', 0.6);
title('Feature Space: Top 3 Features');
xlabel('x1'); ylabel('x2'); zlabel('x3');
grid on;
saveas(gcf, 'feature_space.png'); % Save scatter plot of feature space

% ROC metrics with reduced legend entries
[YPred, Score] = resubPredict(Mdl);
numClassesToShow = 10; % Number of classes to display in the legend
randomIdx = randperm(numel(persons), numClassesToShow);
selectedClasses = persons(randomIdx);
filteredScores = Score(:, randomIdx);
rm = rocmetrics(Y, filteredScores, selectedClasses);

figure;
plot(rm);
title('ROC Curve (Subset of Classes)');
legend('show'); % Limited legend for clarity
saveas(gcf, 'roc_metrics.png'); % Save ROC metrics plot

% Improved confusion matrix visualization

% Improved confusion matrix visualization
disp('Displaying Confusion Matrix...');
figure('Position', [100, 100, 1000, 800]); % Larger figure size
cm = confusionchart(Y, YPred, ...
                    'Title', 'Confusion Matrix', ...
                    'FontSize', 8, ...
                    'RowSummary', 'row-normalized', ...
                    'ColumnSummary', 'column-normalized');

% Save the confusion matrix
saveas(gcf, 'confusion_matrix.png');
disp('Confusion Matrix saved successfully.');



%% Step 5: Save model
save('model.mat', 'Mdl', 'persons', 'U', 'targetSize');
disp('Model and outputs saved successfully.');
