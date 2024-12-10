%% Initial Setup
% Set the target size for the images and the location of the image data.
targetSize=[128,128];
location = fullfile('lfw');
svd_cache = fullfile('cache','svd.mat');

%% Create Image Datastore
% Display a message indicating the creation of the image datastore.
disp('Creating image datastore...');

% Create an ImageDatastore object, which is a data source for reading images and image sequences.
% The 'IncludeSubfolders' option is set to true to include all subfolders in the specified location.
% The 'LabelSource' option is set to 'foldernames' to use the folder names as the source of labels.
% The 'ReadFcn' option is set to a function that reads an image file, converts it to grayscale, and resizes it to the target size.
imds = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
                      'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

% Display a montage of the first few images in the datastore.
montage(preview(imds));

%% Read All Images
% Display a message indicating the reading of all images.
disp('Reading all images');

% Read all images from the datastore into a cell array.
A = readall(imds);

% Concatenate all images along the third dimension and reshape the resulting 3-D array into a 2-D array.
B = cat(3,A{:});
imshow(B(:,:,1))
D = prod(targetSize);
B = reshape(B,D,[]);

%% Normalize Data
% Display a message indicating the normalization of the data.
disp('Normalizing data...');

% Normalize the data by dividing by 256 to bring the pixel values into the range [0,1].
B = single(B)./256;

% Move the normalized data to the GPU for faster computation.
N=gpuArray(B);

% Check if the data has been successfully moved to the GPU.
if existsOnGPU(N)
    disp('Successfully moved image data array to GPU')
end

%% Find Singular Value Decomposition (SVD)
% Display a message indicating the computation of the SVD.
disp('Finding SVD...');

% Compute the SVD of the data. This decomposes the data into three matrices: U, S, and V.
% The computation is timed using the tic and toc functions.
tic;
[Ugpu,Sgpu,Vgpu] = svd(N);
toc;

%% Check Status of Arrays
% Display a message indicating the status of the arrays.
disp('Status of arrays:')

% Check if the U, S, and V matrices are on the GPU.
if existsOnGPU(Ugpu)
    disp('U is on GPU.')
end
if existsOnGPU(Vgpu)
    disp('V is on GPU.')
end
if existsOnGPU(Sgpu)
    disp('S is on GPU.')
end

% Gather the U, S, and V matrices from the GPU to the CPU.
U = gather(Ugpu);
V = gather(Vgpu);
S = gather(Sgpu);

%% Plot Singular Values
% Plot the logarithm of the singular values. This can give an indication of the rank of the data.
plot(log(diag(S)),'.');