%----------------------------------------------------------------
% File:     yreport.m
% Purpose:  Load dataset, preprocess images, compute SVD (eigenfaces),
%           extract features, train a multi-class SVM model in parallel,
%           generate necessary figures for the report, and save the model.
%----------------------------------------------------------------
% Author:   Joel Maldonado (Original)
%           Marek Rychlik (Revised)
% Date:     2024-12-17
%----------------------------------------------------------------

% Image preprocessing parameters
targetSize = [128, 128]; % Resize images to 128x128
k = 60; % Number of eigenfaces (features) to use. Increased to capture more detail.
location = fullfile('..', 'lfw'); % Adjust path as needed

% Initialize a parallel pool if not already active for HPC usage
if isempty(gcp('nocreate'))
    parpool('local');
end

% Global Plot Settings
set(groot, 'DefaultAxesFontSize', 10);
set(groot, 'DefaultLineLineWidth', 1.5);
set(groot, 'DefaultFigurePosition', [100, 100, 1200, 800]);

%% Step 1: Load and Preprocess Images in Parallel
disp('Creating image datastore and preprocessing images...');

imds0 = imageDatastore(location, ...
                       'IncludeSubfolders', true, ...
                       'LabelSource', 'foldernames', ...
                       'ReadFcn', @(filename) imresize(im2gray(imread(filename)), targetSize));

% Filter persons with at least 10 images (to ensure adequate training samples)
tbl = countEachLabel(imds0);
mask = tbl{:, 2} >= 10;
persons = unique(tbl{mask, 1});

[lia, ~] = ismember(imds0.Labels, persons);
imds = subset(imds0, lia);

% Display and save selected images (for EDA)
figure('Position', [100, 100, 1200, 600]);
montage(imds);
title('Selected Dataset Images');
saveas(gcf, 'dataset_images.png'); % Save montage
close all;

%% Step 1a: Generate Histogram of Image Counts per Person
disp('Generating histogram of image counts per person...');
counts = countEachLabel(imds);

figure('Position', [100, 100, 800, 600]);
histogram(counts.Count, 'BinWidth', 1, 'FaceColor', [0.2 0.2 0.5]);
xlabel('Number of Images per Person');
ylabel('Number of People');
title('Histogram of Image Counts per Person');
xlim([0, 60]); 
grid on;
saveas(gcf, 'image_counts_histogram.png'); 
close all;

%% Step 1b: Generate Image Collage of Angelina Jolie
disp('Generating image collage for Angelina Jolie...');
targetPerson = 'Angelina_Jolie'; 
angelaIdx = find(imds0.Labels == targetPerson);
if isempty(angelaIdx)
    warning('No images found for %s.', targetPerson);
else
    angelaImds = subset(imds0, angelaIdx);
    figure('Position', [100, 100, 1200, 800]);
    montage(angelaImds, 'Size', [4, ceil(length(angelaIdx)/4)]);
    title(sprintf('Image Collage of %s', strrep(targetPerson, '_', ' ')));
    saveas(gcf, 'angelina_jolie_collage.png'); 
    close all;
end

%% Step 2: Parallelized Reading and Scaling
disp('Reading and vectorizing images in parallel...');
filePaths = imds.Files;
numImages = numel(filePaths);
D = prod(targetSize);

B = zeros(D, numImages, 'single');
parfor i = 1:numImages
    I = imread(filePaths{i});
    I = imresize(im2gray(I), targetSize);
    B(:, i) = single(I(:)) / 255; % Normalize pixel values to [0,1]
end

%% Step 3: Compute SVD for Dimensionality Reduction
disp('Performing SVD to extract eigenfaces...');
[U, S, V] = svd(B, 'econ');

% Display and save top eigenfaces
Eigenfaces = arrayfun(@(j) mat2gray(reshape(U(:, j), targetSize)), 1:16, 'UniformOutput', false);
figure('Position', [100, 100, 1200, 600]);
montage(Eigenfaces, 'Size', [4, 4], 'BorderSize', [5, 5], 'BackgroundColor', 'white');
title('Top 16 Eigenfaces');
saveas(gcf, 'eigenfaces.png');
close all;

%% Step 4: Feature Extraction
disp('Extracting features using top k eigenfaces...');
U_k = U(:, 1:k);
W = S(1:k, 1:k) * V(:, 1:k)';
X = W'; 
Y = categorical(imds.Labels, persons);

%% Step 5: Train Multi-Class SVM
disp('Training multi-class SVM (fitcecoc) using parallel processing...');
options = statset('UseParallel', true);
Mdl = fitcecoc(X, Y, ...
    'Verbose', 2, ...
    'Learners', 'svm', ...
    'Options', options);

%% Step 6a: Visualization of Feature Space
disp('Visualizing feature space...');
figure('Position', [100, 100, 1200, 600]);
scatter3(X(:, 1), X(:, 2), X(:, 3), 50, double(Y), 'filled', 'MarkerFaceAlpha', 0.6);
title('Feature Space: Top 3 Eigenface Features');
xlabel('Feature 1'); ylabel('Feature 2'); zlabel('Feature 3');
grid on;
saveas(gcf, 'feature_space.png');
close all;


disp('Computing predictions and ROC metrics...');
[YPred, Score] = resubPredict(Mdl);

% Limit number of classes for visualization (e.g., 10)
numClassesToShow = min(10, numel(persons));
randomIdx = randperm(numel(persons), numClassesToShow);
selectedClasses = persons(randomIdx);
filteredScores = Score(:, randomIdx);

rm = rocmetrics(Y, filteredScores, selectedClasses);

figure('Position', [100, 100, 1200, 600]);
plot(rm, 'LineWidth', 1.5); % Plot ROC curve
legend('Location', 'southeastoutside');
title('ROC Curve (Subset of Classes)');
saveas(gcf, 'roc_metrics.png');

% Confusion Matrix for a subset of classes
subsetClasses = selectedClasses(1:min(5, numClassesToShow));
subsetMask = ismember(Y, subsetClasses);
figure('Position', [100, 100, 1200, 600]);
confusionchart(Y(subsetMask), YPred(subsetMask), ...
    'Title', 'Confusion Matrix (Subset)', ...
    'FontSize', 10, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
saveas(gcf, 'confusion_matrix.png');



%% Step 7: Save the Model
disp('Saving model and parameters...');
fprintf('Number of people recognized: %d\n', numel(persons));
save('model.mat', 'Mdl', 'persons', 'U_k', 'targetSize'); 
disp('Model and outputs saved successfully.');
