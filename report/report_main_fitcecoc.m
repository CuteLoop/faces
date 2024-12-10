%----------------------------------------------------------------
% File:     main_fitcecoc.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Fri Nov 22 20:02:05 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% This script performs image classification using the fitcecoc function
% in MATLAB, which fits a multiclass model for support vector machines (SVMs)
% or other error-correcting output codes (ECOC). The script uses an image 
% dataset and classifies the images into three classes.

%% Setting up the environment
% Set the target size for the images and the number of features to consider
targetSize=[128,128];
k=30;                                   

% Set the location of the image dataset
location = fullfile('lfw');

%% Creating image datastore
% The imageDatastore function creates a datastore for a collection of images
disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

%% Creating subset of several persons
% Select a subset of images for specific persons
disp('Creating subset of several persons...');
persons = {'Angelina_Jolie', 'Eduardo_Duhalde', 'Amelie_Mauresmo'}

idx = ismember(imds0.Labels, persons);
imds = subset(imds0, idx);

%% Displaying the images
% Create a tiled layout for displaying images and eigenfaces
t=tiledlayout('flow');
nexttile(t);
montage(imds);

%% Reading all images
disp('Reading all images');
A = readall(imds);

B = cat(3,A{:});
D = prod(targetSize);
B = reshape(B,D,[]);

%% Normalizing data
disp('Normalizing data...');
B = single(B)./256;
[B,C,SD] = normalize(B);
tic;
[U,S,V] = svd(B,'econ');
toc;

%% Displaying Eigenfaces
% Get an montage of eigenfaces
Eigenfaces = arrayfun(@(j)reshape((U(:,j)-min(U(:,j)))./(max(U(:,j))-min(U(:,j))),targetSize), ...
    1:size(U,2),'uni',false);

nexttile(t);
montage(Eigenfaces(1:16));
title('Top 16 Eigenfaces');
colormap(gray);

%% Preparing data for SVM
% NOTE: Rows of V are observations, columns are features.
% Observations need to be in rows.
k = min(size(V,2),k);
X0 = V(:,1:k);

[lia,locb] = ismember(imds.Labels, persons);

X = X0(lia,:);
Y = imds.Labels(lia);
cats = persons;
Y=categorical(Y,cats);

% Create colormap
cm=[1,0,0;
    0,0,1,
    0,1,0];
% Assign colors to target values
c=cm(uint8(Y),:);

%% Training Support Vector Machine
disp('Training Support Vector Machine...');
options = statset('UseParallel',true);
tic;
Mdl = fitcecoc(X, Y,'Verbose', 2,'Learners','svm','Options',options);
toc;

%% Plotting feature space
% Generate a plot in feature space using top two features
nexttile(t);
scatter3(X(:,1),X(:,2),X(:,3),50,c);
title('A top 3-predictor plot');
xlabel('x1');
ylabel('x2');
zlabel('x3');

nexttile(t);
scatter3(X(:,4),X(:,5),X(:,6),50,c);
title('A next 3-predictor plot');
xlabel('x4');
ylabel('x5');
zlabel('x6');

%% Predicting and evaluating the model
%[YPred,Score] = predict(Mdl,X);
[YPred,Score,Cost] = resubPredict(Mdl);

%% ROC metrics
% ROC = receiver operating characteristic
% See https://en.wikipedia.org/wiki/Receiver_operating_characteristic
disp('Plotting ROC metrics...');
rm = rocmetrics(imds.Labels, Score, persons);
nexttile(t);
plot(rm);

%% Confusion matrix
disp('Plotting confusion matrix...')
nexttile(t);
confusionchart(Y, YPred);
title(['Number of features: ' ,num2str(k)]);
