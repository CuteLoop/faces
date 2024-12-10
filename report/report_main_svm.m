%----------------------------------------------------------------
% File:     main_svm.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Fri Nov 22 20:02:05 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
% This script uses Support Vector Machine (SVM) for binary classification.
% The goal is to distinguish between two persons (Angelina Jolie and Eduardo Duhalde)
% using their images. The script uses SVD for feature extraction and then trains
% an SVM model for classification.
%----------------------------------------------------------------

% Define the target size for the images
targetSize=[128,128];

% Define the number of features to consider
k=8;                                   

% Create a tiled layout for plots
t=tiledlayout('flow');

% Define the location of the image dataset and the SVD cache
location = fullfile('lfw');
svd_cache = fullfile('cache','svd.mat');

% Load the image dataset
disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

% Select a subset of images for two persons
disp('Creating subset of 2 persons...');
person1 = 'Angelina_Jolie';
person2 = 'Eduardo_Duhalde';

% Create masks for the two persons
mask0_1 = imds0.Labels==person1;
mask0_2 = imds0.Labels==person2;
mask0  = mask0_1|mask0_2;
idx0 = find(mask0);

% Create a subset of images for the two persons
imds = subset(imds0, idx0);

% Display a montage of the images
nexttile(t);
montage(imds);

% Read all images into a matrix
disp('Reading all images');
A = readall(imds);

% Reshape the matrix for further processing
B = cat(3,A{:});
D = prod(targetSize);
B = reshape(B,D,[]);

% Normalize the data
disp('Normalizing data...');
B = single(B)./256;
[B,C,SD] = normalize(B);

% Perform singular value decomposition (SVD) on the data
tic;
[U,S,V] = svd(B,'econ');
toc;

% Get a montage of eigenfaces
Eigenfaces = arrayfun(@(j)reshape((U(:,j)-min(U(:,j)))./(max(U(:,j))-min(U(:,j))),targetSize), ...
    1:size(U,2),'uni',false);

% Display the top 16 eigenfaces
nexttile(t);
montage(Eigenfaces(1:16));
title('Top 16 Eigenfaces');
colormap(gray);

% NOTE: Rows of V are observations, columns are features.
% Observations need to be in rows.
X0 = V(:,1:k);

% Create masks for the two persons
mask1 = imds.Labels==person1;
mask2 = imds.Labels==person2;
mask = mask1|mask2;
idx = find(mask);

% Select the features for the two persons
X = X0(idx,:);
Y = imds.Labels(idx);

% Limit the number of categories to just two
cats = {person1,person2};
Y=categorical(Y,cats);

% Create a colormap for the plots
cm=[1,0,0;
    0,0,1];

% Assign colors to target values
c=cm(uint8(Y),:);

% Train the SVM model
disp('Training Support Vector Machine...');
tic;
Mdl = fitcsvm(X, Y,'Verbose', 1);
toc;

% Generate a plot in feature space using top two features
nexttile(t);
scatter(X(:,1),X(:,2),60,c);
title('A top 2-predictor plot');
xlabel('x1');
ylabel('x2');

% Generate a plot in feature space using next two features
nexttile(t);
scatter(X(:,3),X(:,4),60,c);
title('A next 2-predictor plot');
xlabel('x3');
ylabel('x4');

% Predict the labels for the training data
[YPred,Score,Cost] = resubPredict(Mdl);

% Plot the receiver operating characteristic (ROC) metrics
disp('Plotting ROC metrics...');
rm = rocmetrics(imds.Labels, Score, {person1,person2});
nexttile(t);
plot(rm);

% Plot the confusion matrix
disp('Plotting confusion matrix...')
nexttile(t);
confusionchart(Y, YPred);
title(['Number of features: ' ,num2str(k)]);
