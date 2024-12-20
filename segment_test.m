location = fullfile('lfw', 'Angelina_Jolie'); % Dataset folder
imds = imageDatastore(location); % Create an image datastore

while hasdata(imds)
    I = read(imds); % Read the next image
    [faceMask, maskedImage] = segment_faces(I); % Segment the face
    imshow(maskedImage, []);
    title('Face Segmentation with Landmarks');
    pause(5); % Pause to view results
end
