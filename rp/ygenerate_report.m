%----------------------------------------------------------------
% File:     ygenerate_report.m
% Purpose:  Load dataset, preprocess images, compute SVD (eigenfaces),
%           extract features, train a multi-class SVM model in parallel,
%           generate necessary figures for the report, and save the model.
%----------------------------------------------------------------
% Author:   Joel Maldonado (Original)
%           [Your Name] (Revised)
% Date:     2024-12-19
%----------------------------------------------------------------

% Clear environment
clear all;
close all;
clc;

% Import necessary Report Generator classes
import mlreportgen.report.*
import mlreportgen.dom.*

% Create a new report
rpt = Report('FacialRecognitionReport2', 'pdf');

%% Add Title Page
tp = TitlePage;
tp.Title = 'Facial Recognition Using SVD, Eigenfaces, and SVM';
tp.Subtitle = 'With HPC Integration a MATLAB generated report';
tp.Author = 'Joel Maldonado';

% Optional: Add an image to the title page
% Ensure 'dataset_images.png' exists in the current directory or provide the correct path
if isfile('sam.png')
    tp.Image = Image('sam.png');
    tp.Image.Width = '2in';
    tp.Image.Height = '2in';
end
add(rpt, tp);

%% Add Table of Contents
add(rpt, TableOfContents);

%% Section 1: Data Exploration
sec1 = Section('Data Exploration');

% Paragraph introducing data exploration
para1 = Paragraph([ ...
    'The initial phase of our project involves exploring the dataset to understand its structure, ', ...
    'distribution, and inherent characteristics. This exploration is crucial for informed preprocessing ', ...
    'and effective feature extraction.', ...
    'The data set contains more than 13,000 images of faces collected from the web.
     Each face has been labeled with the name of the person pictured. 1680']);
add(sec1, para1);

% Paragraph and Image: Histogram of Image Counts per Person
para_hist = Paragraph([ ...
    'To assess the distribution of images per individual in the dataset, we generate a histogram. ', ...
    'This visualization reveals the number of images each person has, ranging from people with 10 to 60 images. ', ...
    'Such an analysis helps in identifying data imbalance and determining the sufficiency of training ', ...
    'samples for each class. For instance the person with the most images has 530 images. Failing to filter come of their images could lead to overfitting.']);
add(sec1, para_hist);

% Add the histogram image
if isfile('image_counts_histogram.png')
    img_hist = Image('image_counts_histogram.png');
    img_hist.Width = '6in';
    img_hist.Height = '4in';
    add(sec1, img_hist);
    
    % Add caption
    cap_hist = Paragraph('Figure 1: Histogram showing the distribution of image counts per person in the dataset.');
    add(sec1, cap_hist);
else
    warning('Image file "image_counts_histogram.png" not found.');
end

% Paragraph and Image: Image Collage of Angelina Jolie
para_angela = Paragraph([ ...
    'To observe the variation in images of a single individual, we create a collage of images of ', ...
    'Angelina Jolie. This collage showcases different expressions, angles, lighting conditions, and ', ...
    'occlusions, illustrating the diversity within the images of one person. Such variation is critical ', ...
    'for building a robust recognition system.']);
add(sec1, para_angela);

% Add Angelina Jolie collage image
if isfile('angelina_jolie_collage.png')
    img_angela = Image('angelina_jolie_collage.png');
    img_angela.Width = '6in';
    img_angela.Height = '4in';
    add(sec1, img_angela);
    
    % Add caption
    cap_angela = Paragraph('Figure 2: Image collage of Angelina Jolie showing variation across multiple images.');
    add(sec1, cap_angela);
else
    warning('Image file "angelina_jolie_collage.png" not found.');
end

% Paragraph and Image: Image Collage of Different People
para_collage = Paragraph([ ...
    'To understand the overall variation within the dataset, we create a collage of images of ', ...
    'different individuals. This collage highlights differences in facial features, skin tones, and ', ...
    'image conditions, providing a comprehensive view of the dataset diversity.']);
add(sec1, para_collage);

% Add different people collage image
if isfile('dataset_images.png')
    img_dataset = Image('dataset_images.png');
    img_dataset.Width = '6in';
    img_dataset.Height = '4in';
    add(sec1, img_dataset);
    
    % Add caption
    cap_dataset = Paragraph('Figure 3: Image collage of different individuals in the dataset showcasing diversity.');
    add(sec1, cap_dataset);
else
    warning('Image file "dataset_images.png" not found.');
end

% Add Section 1 to the report
add(rpt, sec1);

%% Section 2: Singular Value Decomposition and Eigenfaces
sec2 = Section('Singular Value Decomposition and Eigenfaces');

% Paragraph introducing SVD
para_svd = Paragraph([ ...
    'Singular Value Decomposition (SVD) is a linear algebra technique that ', ...
    'factorizes a data matrix into three components: ']);
add(sec2, para_svd);

% Add the SVD equation
eq1 = Equation('A = U \Sigma V^T');
add(sec2, eq1);

% Paragraph explaining SVD in facial recognition
para_svd_explanation = Paragraph([ ...
    'In the context of facial recognition, each image is first transformed into a vectorized ', ...
    'form and stacked into a large data matrix A, where each column represents one face image. ', ...
    'Applying SVD to A decomposes it into: U (left singular vectors) which represent the directions of maximum variance in the image space, ', ...
    '\Sigma (singular values) which quantify the importance of each direction, and ', ...
    'V^T (right singular vectors) which provide the coordinates of each image in the new feature space.']);
add(sec2, para_svd_explanation);

% Add another equation for reduced SVD
eq2 = Equation('A_k = U_k \Sigma_k V_k^T');
add(sec2, eq2);

% Paragraph explaining eigenfaces
para_eigenfaces = Paragraph([ ...
    'By retaining only the top k singular values and their corresponding vectors, we reduce the dimensionality ', ...
    'while preserving the most significant information. The columns of U corresponding to the top k singular values ', ...
    'are known as eigenfaces. These eigenfaces are essentially the principal components of the face space,¿. ', ...
    'Hopefully they capture features like facial contours, eyes, noses, and mouth positions that best explain the variance in the dataset.']);
add(sec2, para_eigenfaces);

% Add eigenfaces image
if isfile('eigenfaces.png')
    img_eigen = Image('eigenfaces.png');
    img_eigen.Width = '6in';
    img_eigen.Height = '4.5in';
    add(sec2, img_eigen);
    
    % Add caption
    cap_eigen = Paragraph('Figure 4: Top 16 extracted eigenfaces using SVD.');
    add(sec2, cap_eigen);
else
    warning('Image file "eigenfaces.png" not found.');
end

% Add Section 2 to the report
add(rpt, sec2);

%% Section 3: Feature Extraction and Classification with SVM
sec3 = Section('Feature Extraction and Classification with SVM');

% Paragraph introducing feature extraction and SVM
para_feature_svm = Paragraph([ ...
    'Once we have extracted the eigenfaces, each face image can be represented as a linear combination ', ...
    'of these eigenfaces. Projecting the original images onto the reduced eigenface space yields a feature vector ', ...
    'of significantly smaller dimension, capturing the most discriminative facial features.']);
add(sec3, para_feature_svm);

% Paragraph explaining SVM
para_svm = Paragraph([ ...
    'These low-dimensional feature vectors form the input to a supervised classifier. In this project, ', ...
    'we employ a Support Vector Machine (SVM) classifier. SVMs are chosen for their robustness and effectiveness ', ...
    'in high-dimensional spaces, and when combined with a one-vs-one or Error-Correcting Output Code (ECOC) scheme, ', ...
    'they can handle multiple classes.']);
add(sec3, para_svm);

% Paragraph summarizing the pipeline
para_pipeline = Paragraph([ ...
    'The result is a facial recognition pipeline: Raw images to Preprocessing  to SVD (for eigenfaces)  to ', ...
    'Feature Extraction (projecting onto eigenfaces)  to SVM Classification.']);
add(sec3, para_pipeline);

% Add feature space visualization image
if isfile('feature_space.png')
    img_feat = Image('feature_space.png');
    img_feat.Width = '6in';
    img_feat.Height = '4.5in';
    add(sec3, img_feat);
    
    % Add caption
    cap_feat = Paragraph('Figure 5: Visualization of the top 3 PCA/SVD-derived features of the dataset.');
    add(sec3, cap_feat);
else
    warning('Image file "feature_space.png" not found.');
end

% Add Section 3 to the report
add(rpt, sec3);

%% Section 4: Discussion of Other Approaches and State of the Art
sec4 = Section('Discussion of Other Approaches and State of the Art');

% Paragraph introducing the section
para_discussion = Paragraph([ ...
    'While Singular Value Decomposition (SVD) and Support Vector Machines (SVM) provide a framework ', ...
    'for facial recognition, numerous other methodologies exist in the literature. This section explores ', ...
    'alternative techniques and compares their advantages and limitations.']);
add(sec4, para_discussion);

% Paragraph on PCA
para_pca = Paragraph([ ...
    'Principal Component Analysis (PCA) is a widely used dimensionality reduction technique similar to SVD. ', ...
    'It transforms the data into a new coordinate system by identifying the principal components that maximize ', ...
    'variance. In facial recognition, PCA is employed to extract the most significant features, known as eigenfaces, ', ...
    'which capture the essential facial characteristics.']);
add(sec4, para_pca);

% Optional: Add PCA image
% Uncomment and ensure the image exists
% if isfile('pca_eigenfaces.png')
%     img_pca = Image('pca_eigenfaces.png');
%     img_pca.Width = '6in';
%     img_pca.Height = '4in';
%     add(sec4, img_pca);
%     
%     % Add caption
%     cap_pca = Paragraph('Figure 6: PCA-based Eigenfaces.');
%     add(sec4, cap_pca);
% else
%     warning('Image file "pca_eigenfaces.png" not found.');
% end

% Paragraph on LDA
para_lda = Paragraph([ ...
    'Linear Discriminant Analysis (LDA) focuses on finding a feature space that best separates different classes. ', ...
    'Unlike PCA, which is unsupervised, LDA is supervised and leverages class label information to maximize inter-class ', ...
    'variance while minimizing intra-class variance. This makes LDA particularly effective in classification tasks like ', ...
    'facial recognition.']);
add(sec4, para_lda);

% Optional: Add LDA image
% Uncomment and ensure the image exists
% if isfile('lda_projection.png')
%     img_lda = Image('lda_projection.png');
%     img_lda.Width = '6in';
%     img_lda.Height = '4in';
%     add(sec4, img_lda);
%     
%     % Add caption
%     cap_lda = Paragraph('Figure 7: LDA-based Feature Projection.');
%     add(sec4, cap_lda);
% else
%     warning('Image file "lda_projection.png" not found.');
% end

% Paragraph on Deep Learning Approaches
para_dl = Paragraph([ ...
    'In recent years, deep learning techniques, particularly Convolutional Neural Networks (CNNs), have revolutionized ', ...
    'facial recognition. Models like DeepFace, FaceNet, and VGG-Face learn hierarchical feature representations directly ', ...
    'from raw pixel data, capturing intricate patterns and invariant features that outperform traditional methods. ', ...
    'However, they require large datasets and substantial computational resources for training.']);
add(sec4, para_dl);

% Optional: Add Deep Learning image
% Uncomment and ensure the image exists
% if isfile('deep_learning_eigenfaces.png')
%     img_dl = Image('deep_learning_eigenfaces.png');
%     img_dl.Width = '6in';
%     img_dl.Height = '4in';
%     add(sec4, img_dl);
%     
%     % Add caption
%     cap_dl = Paragraph('Figure 8: Deep Learning-based Facial Feature Extraction.');
%     add(sec4, cap_dl);
% else
%     warning('Image file "deep_learning_eigenfaces.png" not found.');
% end

% Add Section 4 to the report
add(rpt, sec4);

%% Section 5: Results and Metrics of Face Recognition
sec5 = Section('Results and Metrics of Face Recognition');

% Paragraph introducing the results
para_results = Paragraph([ ...
    'To assess model performance, we compute a confusion matrix and ROC curves. The confusion matrix visualizes how well ', ...
    'the classifier distinguishes among different individuals, and the ROC curve shows the trade-off between True Positive ', ...
    'and False Positive rates. In practice, evaluating on subsets of classes helps gauge classifier effectiveness and reveals ', ...
    'areas needing improvement.']);
add(sec5, para_results);

% Add confusion matrix image
if isfile('confusion_matrix.png')
    img_conf = Image('confusion_matrix.png');
    img_conf.Width = '6in';
    img_conf.Height = '4.5in';
    add(sec5, img_conf);
    
    % Add caption
    cap_conf = Paragraph('Figure 6: Confusion Matrix for a subset of 5 classes.');
    add(sec5, cap_conf);
else
    warning('Image file "confusion_matrix.png" not found.');
end

% Add ROC metrics image
if isfile('roc_metrics.png')
    img_roc = Image('roc_metrics.png');
    img_roc.Width = '6in';
    img_roc.Height = '4.5in';
    add(sec5, img_roc);
    
    % Add caption
    cap_roc = Paragraph('Figure 7: ROC Curve for a subset of 10 classes, illustrating classifier sensitivity and specificity.');
    add(sec5, cap_roc);
else
    warning('Image file "roc_metrics.png" not found.');
end

% Add Section 5 to the report
add(rpt, sec5);

%% Section 6: Conclusion and Future Work
sec6 = Section('Conclusion and Future Work');

% Paragraphs summarizing the project and suggesting future directions
para_conclusion = Paragraph([ ...
    'This project demonstrates a scalable facial recognition pipeline that leverages SVD for eigenface extraction and SVM for classification. ', ...
    'By integrating MATLAB’s parallel computing capabilities, we ensure that the approach scales well, handling larger datasets efficiently.']);
add(sec6, para_conclusion);


% sam.png image
if isfile('sam.png')
    img_roc = Image('sam.png');
    img_roc.Width = '6in';
    img_roc.Height = '4.5in';
    add(sec5, img_roc);
    
    % Add caption
    cap_roc = Paragraph('Figure 8:Image segmentation with MATLAB’s 2024b imsegsam function.');
    add(sec5, cap_roc);
else
    warning('Image file "sam.png" not found.');
end


para_future = Paragraph([ ...
    'Future work could explore advanced feature extraction methods. One approach explored was to segment faces before calculating the SVD with matlab imsegsam function. ', ...
    'Unfortunately this function comes with MATLAB’s 2024b which is not available at the moment at the HPC. ', ...
    'Hence we tried to immplemnt our costume image segmentation function by bounding a face in a box. Then locate features like eyes, mouth, nose to approximate the scaling factor of the face. ', ...
    'This is followed by a resize of the box to fit the face and a skin color mask reported by literature to segment skin. ', ...
    'This approach is expected to improve recognition accuracy by focusing on facial features and reducing noise from the background.']);
add(sec6, para_future);

% bbox.png image
if isfile('bbox.png')
    img_roc = Image('bbox.png');
    img_roc.Width = '6in';
    img_roc.Height = '4.5in';
    add(sec5, img_roc);
    
    % Add caption
    cap_roc = Paragraph('Figure 8: Image of bounding box surrounding a face.');
    add(sec5, cap_roc);
else
    warning('Image file "bbox.png" not found.');
end

% seg_mask.png image
if isfile('seg_mask.png')
    img_roc = Image('seg_mask.png');
    img_roc.Width = '6in';
    img_roc.Height = '4.5in';
    add(sec5, img_roc);
    
    % Add caption
    cap_roc = Paragraph('Figure 9 : Image of applying the skincolor mask to segment the image inside the bounding boxs.');
    add(sec5, cap_roc);
else
    warning('Image file "seg_mask.png" not found.');
end




para_additional = Paragraph([ ...
    'Additionally, expanding the dataset to include more diverse subjects and varying image conditions can enhance the system''s robustness and generalizability. ', ...
    'Implementing real-time recognition capabilities and deploying the system in practical applications are also promising avenues for future development.']);
add(sec6, para_additional);

% Add Section 6 to the report
add(rpt, sec6);

%% Section 7: HPC Integration and Scalability
sec7 = Section('HPC Integration and Scalability');

% Paragraphs explaining HPC techniques used
para_hpc1 = Paragraph([ ...
    'High-Performance Computing (HPC) resources are pivotal in scaling facial recognition systems to handle ', ...
    'large datasets efficiently. MATLAB’s Parallel Computing Toolbox facilitates the distribution of computations ', ...
    'across multiple cores or clusters, significantly reducing processing times.']);
add(sec7, para_hpc1);

para_hpc2 = Paragraph([ ...
    'In this project, several HPC techniques are employed:', ...
    '1. **Parallelized Image Preprocessing:** Utilizing parallel loops (`parfor`) to expedite image reading and resizing.', ...
    '2. **Distributed SVD Computation:** Leveraging parallel algorithms to perform Singular Value Decomposition on large matrices.', ...
    '3. **Parallel SVM Training:** Training multi-class SVM models concurrently to enhance scalability and reduce training time.']);
add(sec7, para_hpc2);

% Optional: Add HPC architecture image
% Uncomment and ensure the image exists
% if isfile('parallel_computing.png')
%     img_parallel = Image('parallel_computing.png');
%     img_parallel.Width = '6in';
%     img_parallel.Height = '4in';
%     add(sec7, img_parallel);
%     
%     % Add caption
%     cap_parallel = Paragraph('Figure 8: Parallel Computing Architecture for Facial Recognition.');
%     add(sec7, cap_parallel);
% else
%     warning('Image file "parallel_computing.png" not found.');
% end

% Add Section 7 to the report
add(rpt, sec7);

%% Close and Generate Report
close(rpt);
disp('Report generation completed. Opening the report...');
rptview(rpt);
