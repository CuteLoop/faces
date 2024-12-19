function [faceMask, maskedImage] = segment_faces(I)
    % SEGMENT_FACES Refines face segmentation using key points, scaling, skin detection,
    % and additional processing to handle common issues.
    %
    % Inputs:
    %   I - Input image (RGB or grayscale).
    %
    % Outputs:
    %   faceMask - Binary mask of the segmented face.
    %   maskedImage - Image with the segmented face highlighted.

    % Convert image to grayscale if necessary
    if size(I, 3) == 3
        grayImage = rgb2gray(I);
    else
        grayImage = I;
    end

    % Load a pretrained face detector
    faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');

    % Detect faces
    bbox = step(faceDetector, grayImage);

    % Display detected bounding boxes for debugging
    figure;
    imshow(I);
    hold on;
    for i = 1:size(bbox, 1)
        rectangle('Position', bbox(i, :), 'EdgeColor', 'r', 'LineWidth', 2);
    end
    hold off;
    title('Detected Faces');
    pause(1);
    close all;

    % Initialize the binary mask
    faceMask = false(size(grayImage));

    if isempty(bbox)
        warning('No faces detected in the image.');
        maskedImage = I; % Return the original image
        return;
    end

    for i = 1:size(bbox, 1)
        % Extract initial face region (bounding box)
        x = bbox(i, 1);
        y = bbox(i, 2);
        w = bbox(i, 3);
        h = bbox(i, 4);
        faceROI = I(y:y+h-1, x:x+w-1, :);

        % Apply skin detection in the bounding box
        skinMask = detect_skin(faceROI);

        % Display skin mask for debugging
        figure;
        subplot(1, 2, 1);
        imshow(faceROI);
        title('Face ROI');

        subplot(1, 2, 2);
        imshow(skinMask);
        title('Skin Mask');
        pause(1);
        close all;

        % Remove small islands
        refinedMask = remove_small_islands(skinMask);

        % Display refined mask after removing small islands
        figure;
        imshow(refinedMask);
        title('Refined Mask - After Removing Small Islands');
        pause(1);
        close all;

        % Refine the mask with boundary detection and shape analysis
        refinedMask = refine_skin_mask(refinedMask, faceROI);

        % Display refined mask after boundary refinement
        figure;
        imshow(refinedMask);
        title('Refined Mask - After Boundary Refinement');
        pause(1);
        close all;

        % Ensure the mask has no holes and is connected
        refinedMask = fill_holes(refinedMask);

        % Display final refined mask
        figure;
        imshow(refinedMask);
        title('Final Refined Mask');
        pause(1);
        close all;

        % Resize the refined mask back to the original image size
        fullMask = false(size(grayImage));
        fullMask(y:y+h-1, x:x+w-1) = refinedMask;

        % Combine the masks for all faces
        faceMask = faceMask | fullMask;
    end

    % Check if faceMask is empty
    if ~any(faceMask(:))
        warning('Face mask is empty after processing.');
        maskedImage = I; % Return the original image
        return;
    end

    % Highlight the segmented face region
    maskedImage = labeloverlay(I, faceMask, 'Transparency', 0.5);

    % Display the final overlay
    figure;
    subplot(1, 3, 1);
    imshow(I);
    title('Original Image');

    subplot(1, 3, 2);
    imshow(faceMask);
    title('Face Mask (Refined)');

    subplot(1, 3, 3);
    imshow(maskedImage);
    title('Refined Face Segmentation');
end

function skinMask = detect_skin(roi)
    % DETECT_SKIN Performs skin detection in the YCbCr color space.
    %
    % Input:
    %   roi - Region of interest (RGB image).
    %
    % Output:
    %   skinMask - Binary mask of the detected skin region.

    % Convert to YCbCr color space
    ycbcr = rgb2ycbcr(roi);
    cb = ycbcr(:, :, 2); % Cb channel
    cr = ycbcr(:, :, 3); % Cr channel

    % Define refined skin color range thresholds
    cbMin = 77; cbMax = 200;
    crMin = 134; crMax = 173;

    % Create a binary mask for skin regions
    skinMask = (cb >= cbMin & cb <= cbMax) & (cr >= crMin & cr <= crMax);

    % Optionally apply morphological operations to clean up the mask
    skinMask = imerode(skinMask, strel('disk', 1)); % Erode noise
end

function refinedMask = remove_small_islands(mask)
    % REMOVE_SMALL_ISLANDS Removes small islands from a binary mask.
    %
    % Inputs:
    %   mask - Binary mask with potentially multiple disconnected regions.
    %
    % Outputs:
    %   refinedMask - Binary mask retaining only the largest connected region.

    % Label connected components in the mask
    cc = bwconncomp(mask);

    if cc.NumObjects < 1
        refinedMask = mask;
        return;
    end

    % Measure the area of each connected component
    stats = regionprops(cc, 'Area');
    areas = [stats.Area];

    % Find the largest connected component
    [~, largestIdx] = max(areas);

    % Create a refined mask with only the largest component
    refinedMask = false(size(mask));
    refinedMask(cc.PixelIdxList{largestIdx}) = true;
end

function refinedMask = refine_skin_mask(mask, roi)
    % REFINE_SKIN_MASK Refines a skin mask using boundary detection and shape analysis.
    %
    % Inputs:
    %   mask - Binary mask of the skin-detected region.
    %   roi - Region of interest (RGB image corresponding to the mask).
    %
    % Outputs:
    %   refinedMask - Refined binary mask with fewer false positives.

    % Edge detection on the grayscale version of the region
    grayROI = rgb2gray(roi);
    edges = edge(grayROI, 'Canny');

    % Combine the edges with the initial mask
    boundaryMask = imdilate(edges, strel('disk', 2)); % Dilate the edges
    combinedMask = mask & ~boundaryMask; % Exclude boundary regions

    % Keep the largest connected component as the refined mask
    refinedMask = remove_small_islands(combinedMask);

    % Additional Shape Analysis
    stats = regionprops(refinedMask, 'MajorAxisLength', 'MinorAxisLength', 'Area');
    if ~isempty(stats)
        for i = 1:length(stats)
            % Calculate the aspect ratio
            aspectRatio = stats(i).MajorAxisLength / stats(i).MinorAxisLength;
            % Expect an aspect ratio close to 1 (oval shape)
            if aspectRatio < 0.75 || aspectRatio > 1.25
                % Remove regions that do not resemble an oval
                refinedMask(refinedMask == i) = false;
            end
        end
    end

    % Fill any remaining holes and ensure connectivity
    refinedMask = fill_holes(refinedMask);
end

function filledMask = fill_holes(mask)
    % FILL_HOLES Ensures the segmented mask has no holes and is connected.
    %
    % Inputs:
    %   mask - Binary mask of the segmented region.
    %
    % Outputs:
    %   filledMask - Binary mask with all holes filled and connectivity enforced.

    % Fill holes in the mask
    filledMask = imfill(mask, 'holes');

    % Ensure connectivity by retaining only the largest connected component
    filledMask = remove_small_islands(filledMask);
end
