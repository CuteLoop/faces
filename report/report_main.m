%----------------------------------------------------------------
% File:     main.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Fri Nov 22 20:02:05 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------
% This script implements a basic face recognition system workflow.
% It uses the Singular Value Decomposition (SVD) and Support Vector Machine (SVM)
% for image classification.

%% Initialization
% Set the target size for images and specify the location of the image dataset.
% Also, specify the location of the cache file for SVD results.
targetSize=[128,128];
location = fullfile('lfw');
svd_cache = fullfile('cache','svd.mat');

%% Creating Image Datastore
% Create an image datastore object to manage a large collection of image files.
% The images are resized and converted to grayscale.
disp('Creating image datastore...');
imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));
montage(preview(imds));

%% Reading Images
% Read all images from the image datastore into a cell array.
disp('Reading all images...');
A = readall(imds);

%% Normalizing Data
% Normalize the image data to have values between 0 and 1.
disp('Normalizing data...');
B = cat(3,A{:});
D = prod(targetSize);
B = reshape(B,D,[]);
B = single(B)./256;
N=B;

%% Singular Value Decomposition (SVD)
% Perform SVD on the normalized image data.
% If the SVD results are cached, load them from the cache file.
% Otherwise, compute the SVD and save the results to the cache file.
if exist(svd_cache,'file') == 2
    disp('Loading SVD from cache...');
    load(svd_cache)
else
    disp('Finding SVD...');
    tic;
    [U,S,V] = svd(N,'econ');
    toc;
    disp('Writing SVD cache...')
    save(svd_cache,'U','S','V');
end

%% Training Support Vector Machine (SVM)
% Train a SVM classifier using the SVD results.
% The classifier is trained to recognize images of George W. Bush.
disp('Training Support Vector Machine...');
k=512;Z=U(:,1:k)*S(1:k,1:k)*V(:,1:k)';
X = V(:,1:k);
Y = imds.Labels=='George_W_Bush';
% Map to +1/-1
Y = 2.*Y-1;

tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',100,tTree);

options = statset('UseParallel',true);
Mdl = fitcecoc(X,Y,'Coding','onevsall','Learners',tEnsemble,...
               'Prior','uniform','NumBins',50,'Options',options,...
               'Verbose',2);

%% Testing the SVM Classifier
% Test the SVM classifier on an image of George W. Bush.
disp('Testing on "Dabya"...');
W = X(3949,:);
I = reshape(U(:,1:k)*S(1:k,1:k)*W',targetSize);
imagesc(I);
colormap gray;
drawnow;

%% Running Prediction
% Run the SVM classifier to predict the label of the test image.
disp('Running prediction...');
[label, score] = predict(Mdl, W)