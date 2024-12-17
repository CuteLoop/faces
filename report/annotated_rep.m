%----------------------------------------------------------------
% File:     main_gpu.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Mon Dec  9 07:43:04 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
%
%----------------------------------------------------------------
% Purpose:  This script demonstrates the use of GPU acceleration in MATLAB
%           to compute the Singular Value Decomposition (SVD) of an image 
%           dataset. The images are loaded, preprocessed (grayscale, resizing),
%           normalized, and then SVD is applied on the GPU for performance
%           optimization. 
%
% Key Steps:
%   1. Load and preprocess image data
%   2. Move data to GPU for parallel computation
%   3. Perform Singular Value Decomposition (SVD)
%   4. Visualize the singular values
%----------------------------------------------------------------

%% Initialization and Settings
targetSize = [128,128];                   % Target size for resizing images
location = fullfile('lfw');               % Path to the image dataset
svd_cache = fullfile('cache','svd.mat');  % Cache file for saving SVD results

% Displaying process status
disp('Creating image datastore...');

%% Step 1: Load and Preprocess the Images
% Create an imageDatastore to manage large image datasets.
% 'ReadFcn' resizes images to 'targetSize' and converts to grayscale.

imds = imageDatastore(location, 'IncludeSubfolders', true, ...
                      'LabelSource', 'foldernames', ...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)), targetSize));

% Display a preview of the image datastore as a montage
montage(preview(imds));   
disp('Reading all images');

%% Step 2: Convert Images to a Single Matrix
% Read all images into a cell array and concatenate them into a 3D array.
A = readall(imds);       % Read all images
B = cat(3, A{:});        % Concatenate all images along the 3rd dimension

% Display the first image
imshow(B(:,:,1));

% Flatten the 3D image matrix into a 2D matrix for SVD computation
% Each column represents an image vectorized into a column.
D = prod(targetSize);    % Total number of pixels per image
B = reshape(B, D, []);   % Reshape: Pixels x Number of Images

%% Step 3: Normalize Data and Move to GPU
disp('Normalizing data...');
B = single(B)./256;      % Normalize image pixel values to [0, 1]

% Move data to GPU for faster computation using gpuArray
N = gpuArray(B);

% Verify if the data successfully moved to GPU
if existsOnGPU(N)
    disp('Successfully moved image data array to GPU');
end

%% Step 4: Compute Singular Value Decomposition (SVD) on GPU
% SVD: Decomposes the matrix N into U, S, and V such that N = U*S*V'.
% This step benefits significantly from GPU acceleration for large datasets.

disp('Finding SVD...');
tic;                             % Start timer to measure computation time
[Ugpu, Sgpu, Vgpu] = svd(N);     % Perform SVD on GPU
toc;                             % Stop timer and display elapsed time

%% Step 5: Check the Status of Arrays on GPU
disp('Status of arrays:');
if existsOnGPU(Ugpu), disp('U is on GPU.'); end
if existsOnGPU(Vgpu), disp('V is on GPU.'); end
if existsOnGPU(Sgpu), disp('S is on GPU.'); end

%% Step 6: Gather Results Back to CPU
% Move the SVD results back to CPU memory for further analysis.
U = gather(Ugpu);
V = gather(Vgpu);
S = gather(Sgpu);

%% Step 7: Visualize Singular Values
% Plot the logarithm of the diagonal elements of S (singular values).
% Singular values provide information about the importance of each mode in SVD.

plot(log(diag(S)), '.');  % Logarithmic scale for better visualization
xlabel('Index of Singular Value');
ylabel('Log of Singular Value');
title('Singular Value Decay');
grid on;

%----------------------------------------------------------------
% End of Script
%----------------------------------------------------------------
