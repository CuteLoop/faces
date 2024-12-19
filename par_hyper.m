%---------------------------------------------------------------- 
% File:     main_fitcecoc_gpu_parallel.m
%----------------------------------------------------------------
%
% Author:   Original by Marek Rychlik (rychlik@arizona.edu)
%           Modified by [Your Name]
% Date:     [Current Date]
% 
% Description:
% This script trains a facial recognition model. The script has been
% modified to:
%    - Perform parallel computations where possible
%    - Utilize GPU acceleration for data preprocessing and SVD
%    - Use hyperparameter optimization in fitcecoc
%
%----------------------------------------------------------------

% Ensure a parallel pool is running
if isempty(gcp('nocreate'))
    % Use the default parallel pool; adjust the number of workers as needed
    parpool('threads');
end

% Select GPU device (if multiple GPUs are available, you can pick which one)
gpuDevice(1);

targetSize = [128,128];
k = 60; % Number of features

location = fullfile('lfw');

disp('Creating image datastore...');
imds0 = imageDatastore(location,'IncludeSubfolders',true,'LabelSource','foldernames',...
    'ReadFcn', @(filename)imresize(im2gray(imread(filename)),targetSize));

disp('Creating subset of several persons...');
% For demonstration, the following line is static; actual selection happens below
% persons = {'Angelina_Jolie', 'Eduardo_Duhalde', 'Amelie_Mauresmo'}

tbl = countEachLabel(imds0);
mask = tbl{:,2}>=5 & tbl{:,2}<=60;
disp(['Number of images: ', num2str(sum(tbl{mask,2}))]);
persons = unique(tbl{mask,1});
fprintf('Number of people recognized: %d\n', numel(persons));

[lia, locb] = ismember(imds0.Labels, persons);
imds = subset(imds0, lia);

t = tiledlayout('flow');
nexttile(t);
montage(imds);

disp('Reading all images...');
A = readall(imds);  % Cell array of images

% Concatenate into a single 3D array
B = cat(3,A{:});
D = prod(targetSize);
B = reshape(B,D,[]);

disp('Normalizing data...');
% Move data to GPU
B = gpuArray(single(B)./256);

% Normalize along dimension 1
[B,C,SD] = normalize(B);

disp('Computing SVD on GPU...');
tic;
[U,S,V] = svd(B,'econ');
toc;

% Convert back to CPU if needed (for final saving and training)
U = gather(U);
S = gather(S);
V = gather(V);

% Show top eigenfaces
Eigenfaces = arrayfun(@(j)reshape((U(:,j)-min(U(:,j)))./(max(U(:,j))-min(U(:,j))),targetSize), ...
    1:size(U,2),'uni',false);

nexttile(t);
montage(Eigenfaces(1:16));
title('Top 16 Eigenfaces');
colormap(gray);

k = min(size(V,2),k);

% Compute weight vectors (features)
W = S * V';         % W is k-by-numImages
W = W(1:k,:);
U = U(:,1:k);        % Keep only first k eigenfaces

X = W';              % Observations in rows
Y = categorical(imds.Labels, persons);

% Simple colormap
cm = [1,0,0; 0,0,1; 0,1,0];
c = cm(1+mod(uint8(Y),size(cm,1)),:);

disp('Training Support Vector Machine with Parallel and Hyperparameter Optimization...');
options = statset('UseParallel',true);

tic;
Mdl = fitcecoc(X, Y, ...
    'Verbose', 2, ...
    'Learners','svm', ...
    'OptimizeHyperparameters','all', ...
    'Options', options);
toc;

% Generate some plots
nexttile(t);
scatter3(X(:,1), X(:,2), X(:,3), 50, c);
title('A top 3-predictor plot');
xlabel('x1'); ylabel('x2'); zlabel('x3');

nexttile(t);
scatter3(X(:,4), X(:,5), X(:,6), 50, c);
title('A next 3-predictor plot');
xlabel('x4'); ylabel('x5'); zlabel('x6');

[YPred,Score,Cost] = resubPredict(Mdl);

disp('Plotting ROC metrics...');
rm = rocmetrics(imds.Labels, Score, persons);
nexttile(t);
plot(rm);

disp('Plotting confusion matrix...');
nexttile(t);
confusionchart(Y, YPred);
title(['Number of features: ' ,num2str(k)]);

% Save the model
% Includes Mdl (trained model), persons, U (eigenfaces), targetSize
save('model','Mdl','persons','U','targetSize');
disp('saved model.');
