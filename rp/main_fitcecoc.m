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
t = tiledlayout('flow');
nexttile(t);
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
montage(Eigenfaces);
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
scatter3(X(:,1), X(:,2), X(:,3), 50, uint8(Y), 'filled');
title('Feature Space: Top 3 Features');
xlabel('x1'); ylabel('x2'); zlabel('x3');
saveas(gcf, 'feature_space.png'); % Save scatter plot of feature space

% ROC metrics
[YPred, Score] = resubPredict(Mdl);
rm = rocmetrics(Y, Score, persons);

figure;
plot(rm);
title('ROC Metrics');
saveas(gcf, 'roc_metrics.png'); % Save ROC metrics plot

% Confusion matrix
figure;
confusionchart(Y, YPred);
title('Confusion Matrix');
saveas(gcf, 'confusion_matrix.png'); % Save confusion matrix plot

%% Step 5: Save model
save('model.mat', 'Mdl', 'persons', 'U', 'targetSize');
disp('Model and outputs saved successfully.');
