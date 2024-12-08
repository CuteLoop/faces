%----------------------------------------------------------------
% File:     recognize_faces_test.m
%----------------------------------------------------------------
%
% Author:   Marek Rychlik (rychlik@arizona.edu)
% Date:     Sun Dec  8 15:55:17 2024
% Copying:  (C) Marek Rychlik, 2020. All rights reserved.
% 
%----------------------------------------------------------------

location = fullfile('lfw');

disp('Creating image datastore...');
% We read images in their original format
imds0 = imageDatastore(location,...
                       'IncludeSubfolders',true,...
                       'LabelSource','foldernames');

load('model',["persons"])

idx = ismember(imds0.Labels, persons);
imds = subset(imds0, idx);

RGB = readall(imds);

YPred = recognize_faces(RGB);

% Burn labels into the images
for j=1:numel(RGB)
    RGBannotated{j} = insertObjectAnnotation(RGB{j}, 'rectangle', [10,10,100,20], YPred(j));
end

montage(RGBannotated);
