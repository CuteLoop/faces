%----------------------------------------------------------------
% File:     ygenerate_report.m
% Purpose:  Load dataset, preprocess images, compute SVD (eigenfaces),
%           extract features, train a multi-class SVM model in parallel,
%           and generate a nicely organized PDF report.
%----------------------------------------------------------------
% Author:   Joel Maldonado (Original)
%           [Your Name] (Revised)
% Date:     2024-12-19
%----------------------------------------------------------------

clear all; close all; clc;

import mlreportgen.report.*
import mlreportgen.dom.*

% Create a new report
rpt = Report('FacialRecognitionReport3', 'pdf');

%% Add Title Page
tp = TitlePage;
tp.Title = 'Facial Recognition Using SVD, Eigenfaces, and SVM';
tp.Subtitle = 'With HPC Integration – A MATLAB Generated Report';
tp.Author = 'Joel Maldonado';

% Optional: Add an image to the title page if available
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

para1 = Paragraph([ ...
    'The initial phase of this project involves exploring the dataset to understand its structure and distribution. ', ...
    'This exploration guides preprocessing decisions and ensures that feature extraction methods capture the key ', ...
    'variations in the data. The dataset contains more than 13,000 images of faces collected from the web. ', ...
    'Each face is labeled with a person''s name, and some individuals have as many as 530 images. Such imbalances ', ...
    'can lead to overfitting if not addressed properly.']);
add(sec1, para1);

% Histogram of Image Counts
para_hist = Paragraph([ ...
    'We begin by examining the distribution of images per person. The histogram below shows the number of images ', ...
    'each person has. Most individuals have between 10 and 60 images.']);
add(sec1, para_hist);

if isfile('image_counts_histogram.png')
    img_hist = Image('image_counts_histogram.png');
    img_hist.Width = '4.5in';
    img_hist.Height = '3in';
    add(sec1, img_hist);
    cap_hist = Paragraph('Figure 1: Histogram showing the distribution of image counts per person.');
    add(sec1, cap_hist);
else
    warning('image_counts_histogram.png not found.');
end

% Angelina Jolie Collage
para_angela = Paragraph([ ...
    'To understand intra-class variability, consider the collage of Angelina Jolie. The images vary in pose, lighting, ', ...
    'and expression, which highlights the complexity of face recognition even for a single individual.']);
add(sec1, para_angela);

if isfile('angelina_jolie_collage.png')
    img_angela = Image('angelina_jolie_collage.png');
    img_angela.Width = '4.5in';
    img_angela.Height = '3in';
    add(sec1, img_angela);
    cap_angela = Paragraph('Figure 2: Image collage of Angelina Jolie.');
    add(sec1, cap_angela);
else
    warning('angelina_jolie_collage.png not found.');
end

% Dataset Collage
para_collage = Paragraph([ ...
    'Beyond one person, the dataset contains a diverse range of individuals from various backgrounds. ', ...
    'The montage below offers a glimpse into the rich variety of faces, which is crucial for building a robust model.']);
add(sec1, para_collage);

if isfile('dataset_images.png')
    img_dataset = Image('dataset_images.png');
    img_dataset.Width = '4.5in';
    img_dataset.Height = '3in';
    add(sec1, img_dataset);
    cap_dataset = Paragraph('Figure 3: Montage of different individuals, illustrating dataset diversity.');
    add(sec1, cap_dataset);
else
    warning('dataset_images.png not found.');
end

add(rpt, sec1);

%% Section 2: Singular Value Decomposition and Eigenfaces
sec2 = Section('Singular Value Decomposition and Eigenfaces');

para_svd = Paragraph([ ...
    'Singular Value Decomposition (SVD) decomposes the data matrix into three components: U, Σ, and V^T. ', ...
    'When applied to our collection of face images, SVD reveals directions of maximum variance, known as ', ...
    'eigenfaces. These eigenfaces serve as a compact basis for representing facial images.']);
add(sec2, para_svd);

eq1 = Equation('A = U \Sigma V^T');
add(sec2, eq1);

para_svd_explanation = Paragraph([ ...
    'Here, A is the data matrix where each column is a vectorized face. U contains eigenfaces, Σ holds singular values ', ...
    'indicating the importance of each eigenface, and V^T gives coordinates of faces in this reduced space. By choosing ', ...
    'only the top k eigenfaces, we capture the most critical variations and reduce dimensionality.']);
add(sec2, para_svd_explanation);

eq2 = Equation('A_k = U_k \Sigma_k V_k^T');
add(sec2, eq2);

para_eigenfaces = Paragraph([ ...
    'The resulting eigenfaces highlight essential facial features (eyes, nose, mouth), capturing primary variations. ', ...
    'Below are the top 16 eigenfaces extracted using SVD. Each eigenface is a "building block" of any face in the dataset.']);
add(sec2, para_eigenfaces);

if isfile('eigenfaces.png')
    img_eigen = Image('eigenfaces.png');
    img_eigen.Width = '5in';
    img_eigen.Height = '3.5in';
    add(sec2, img_eigen);
    cap_eigen = Paragraph('Figure 4: Top 16 extracted eigenfaces.');
    add(sec2, cap_eigen);
else
    warning('eigenfaces.png not found.');
end

add(rpt, sec2);

%% Section 3: Feature Extraction and Classification with SVM
sec3 = Section('Feature Extraction and Classification with SVM');

para_feature_svm = Paragraph([ ...
    'Projecting each face image onto these eigenfaces yields a feature vector. This lower-dimensional representation ', ...
    'is more tractable and retains the most discriminative information. Next, a Support Vector Machine (SVM) classifier ', ...
    'is trained on these feature vectors. SVMs are robust and perform well in complex classification tasks.']);
add(sec3, para_feature_svm);

para_svm = Paragraph([ ...
    'By using a multi-class strategy (e.g., one-vs-one or ECOC), we extend SVM for multiple person recognition. The pipeline ', ...
    'is thus: Raw Image → Preprocessing → SVD (Eigenfaces) → Feature Extraction → SVM Classification.']);
add(sec3, para_svm);

if isfile('feature_space.png')
    img_feat = Image('feature_space.png');
    img_feat.Width = '5in';
    img_feat.Height = '3.5in';
    add(sec3, img_feat);
    cap_feat = Paragraph('Figure 5: Visualization of the top 3 eigenface-based features.');
    add(sec3, cap_feat);
else
    warning('feature_space.png not found.');
end

add(rpt, sec3);

%% Section 4: Discussion of Other Approaches and State of the Art
sec4 = Section('Discussion of Other Approaches and State of the Art');

para_discussion = Paragraph([ ...
    'While SVD and SVM form a strong baseline, the field of face recognition has evolved. PCA, LDA, and deep learning ', ...
    'approaches provide alternative or complementary strategies.']);
add(sec4, para_discussion);

para_pca = Paragraph([ ...
    'Principal Component Analysis (PCA) is closely related to SVD and widely used for extracting eigenfaces. LDA ', ...
    'optimizes class separability, often improving classification performance.']);
add(sec4, para_pca);

para_dl = Paragraph([ ...
    'Deep learning methods (e.g., CNNs like FaceNet or VGG-Face) learn complex, hierarchical features directly from the data. ', ...
    'They typically outperform classical methods but require large datasets and high computational power.']);
add(sec4, para_dl);

add(rpt, sec4);

%% Section 5: Results and Metrics of Face Recognition
sec5 = Section('Results and Metrics of Face Recognition');

para_results = Paragraph([ ...
    'We evaluate the trained model using confusion matrices and ROC curves. The confusion matrix shows correct and misclassified faces. ', ...
    'ROC curves illustrate performance trade-offs in terms of true and false positive rates.']);
add(sec5, para_results);

if isfile('confusion_matrix.png')
    img_conf = Image('confusion_matrix.png');
    img_conf.Width = '5in';
    img_conf.Height = '3.5in';
    add(sec5, img_conf);
    cap_conf = Paragraph('Figure 6: Confusion Matrix for a subset of classes.');
    add(sec5, cap_conf);
else
    warning('confusion_matrix.png not found.');
end

para_roc = Paragraph([ ...
    'Below is an ROC curve for selected classes. Higher AUC values indicate better discrimination between individuals.']);
add(sec5, para_roc);

if isfile('roc_metrics.png')
    img_roc = Image('roc_metrics.png');
    img_roc.Width = '5in';
    img_roc.Height = '3.5in';
    add(sec5, img_roc);
    cap_roc = Paragraph('Figure 7: ROC Curve for selected classes.');
    add(sec5, cap_roc);
else
    warning('roc_metrics.png not found.');
end

add(rpt, sec5);

%% Section 6: Conclusion and Future Work
sec6 = Section('Conclusion and Future Work');

para_conclusion = Paragraph([ ...
    'This project demonstrates a scalable facial recognition pipeline using SVD (for eigenface extraction) and SVM for classification. ', ...
    'Parallelization techniques accelerate data handling and model training, making the approach suitable for larger datasets.']);
add(sec6, para_conclusion);

para_future = Paragraph([ ...
    'Future work may focus on advanced segmentation before applying SVD, such as using MATLAB’s 2024b "imsegsam" function, ', ...
    'or alternative methods like bounding boxes and skin color masks to isolate facial features. This preprocessing could improve recognition accuracy.']);
add(sec6, para_future);

% Add sam.png image (if available)
if isfile('sam.png')
    img_sam = Image('sam.png');
    img_sam.Width = '4.5in';
    img_sam.Height = '3in';
    add(sec6, img_sam);
    cap_sam = Paragraph('Figure 8: Attempted segmentation using "imsegsam" (MATLAB 2024b).');
    add(sec6, cap_sam);
else
    warning('sam.png not found.');
end

% Add bbox.png image (if available)
if isfile('bbox.png')
    img_bbox = Image('bbox.png');
    img_bbox.Width = '4.5in';
    img_bbox.Height = '3in';
    add(sec6, img_bbox);
    cap_bbox = Paragraph('Figure 9: Bounding box around a detected face.');
    add(sec6, cap_bbox);
else
    warning('bbox.png not found.');
end

% Add seg_mask.png image (if available)
if isfile('seg_mask.png')
    img_seg = Image('seg_mask.png');
    img_seg.Width = '4.5in';
    img_seg.Height = '3in';
    add(sec6, img_seg);
    cap_seg = Paragraph('Figure 10: Applying a skin-color mask to segment the face region.');
    add(sec6, cap_seg);
else
    warning('seg_mask.png not found.');
end

para_additional = Paragraph([ ...
    'Expanding the dataset, incorporating more robust segmentation, and exploring deep learning models are all potential avenues ', ...
    'for further research. Such enhancements promise to improve both the accuracy and scalability of future facial recognition systems.']);
add(sec6, para_additional);

add(rpt, sec6);

%% Section 7: HPC Integration and Scalability
sec7 = Section('HPC Integration and Scalability');

para_hpc1 = Paragraph([ ...
    'High-Performance Computing (HPC) resources streamline the pipeline. MATLAB’s Parallel Computing Toolbox can parallelize image ', ...
    'preprocessing, SVD computations, and SVM training. This reduces training time significantly and supports scaling to even larger datasets.']);
add(sec7, para_hpc1);

para_hpc2 = Paragraph([ ...
    'HPC integration ensures that methods like SVD and classification can run efficiently. Leveraging multiple cores or a cluster environment ', ...
    'can turn a time-consuming process into a more manageable and repeatable computation.']);
add(sec7, para_hpc2);

add(rpt, sec7);

%% Close and Generate Report
close(rpt);
disp('Report generation completed. Opening the report...');
rptview(rpt);
