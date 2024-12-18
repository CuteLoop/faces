function label = recognize_faces(image, model)
    % Load the model variables
    load('model.mat', 'Mdl', 'persons', 'U', 'targetSize');
    
    % Preprocess the input image
    I = imresize(im2gray(image), targetSize);
    I = single(I(:)) ./ 255; % Normalize
    
    % Normalize using training data's mean and std (if saved)
    % If meanB and stdB are not saved, ensure consistent preprocessing
    % Example:
    % I = (I - meanB) ./ stdB;
    
    % Project onto eigenfaces
    feature = U' * I;
    
    % Predict using the trained SVM model
    label = predict(Mdl, feature');
end
