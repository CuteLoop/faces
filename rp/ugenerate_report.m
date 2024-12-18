import mlreportgen.report.*
import mlreportgen.dom.*

% Create a new report
rpt = Report('FacialRecognitionReport', 'pdf');

%% Add Title Page
tp = TitlePage;
tp.Title = 'Facial Recognition Using SVD, Eigenfaces, and SVM';
tp.Subtitle = 'With HPC Integration in MATLAB';
tp.Author = 'Marek Rychlik';
tp.Image = which('face.png'); % Optional: Add a title image if available
add(rpt, tp);

%% Add Table of Contents
add(rpt, TableOfContents);

%% Section 1: Introduction and Exploratory Data Analysis (EDA)
sec1 = Section('Introduction and Exploratory Data Analysis (EDA)');

para1 = Paragraph([ ...
    'Facial recognition systems have become integral in various fields, including security, ', ...
    'authentication, and human-computer interaction. The primary challenge in facial ', ...
    'recognition lies in dealing with high-dimensional image data. Here, we harness the ', ...
    'power of Singular Value Decomposition (SVD) to reduce dimensionality and extract the ', ...
    'most discriminative features—often referred to as eigenfaces. By integrating MATLAB’s ', ...
    'capabilities with High-Performance Computing (HPC) resources, we aim to scale this ', ...
    'process to large datasets efficiently.']);

add(sec1, para1);

para2 = Paragraph([ ...
    'In the Exploratory Data Analysis (EDA) phase, we begin by examining the dataset and ', ...
    'its characteristics. We standardize image formats, inspect a sample of images, and ', ...
    'evaluate data quality. This helps ensure consistent preprocessing and informs the ', ...
    'choice of dimensionality reduction and classification techniques.']);
add(sec1, para2);

% Add a montage of images (saved as 'dataset_images.png')
img_dataset = Image('dataset_images.png');
img_dataset.Width = '6in';
img_dataset.Height = '3in';
cap_dataset = Paragraph('Figure 1: A sample of the processed dataset images.');
add(sec1, img_dataset);
add(sec1, cap_dataset);

add(rpt, sec1);

%% Section 2: Understanding SVD and Eigenfaces
sec2 = Section('Singular Value Decomposition and Eigenfaces');

para3 = Paragraph([ ...
    'Singular Value Decomposition (SVD) is a powerful linear algebra technique that ', ...
    'factorizes a data matrix A into three components: U, Σ (Sigma), and V^T. Mathematically:', ...
    ' A = U Σ V^T.']);
add(sec2, para3);

para4 = Paragraph([ ...
    'In the context of facial recognition, each image is first transformed into a vectorized ', ...
    'form and stacked into a large data matrix A, where each column represents one face image. ', ...
    'Applying SVD to A decomposes it into: ' ...
    'U (left singular vectors) which represent the directions of maximum variance in the image space, ', ...
    'Σ (singular values) which quantify the importance of each direction, and ', ...
    'V^T (right singular vectors) which provide the coordinates of each image in the new feature space.']);

add(sec2, para4);

para5 = Paragraph([ ...
    'By retaining only the top k singular values and their corresponding vectors, we reduce the dimensionality ', ...
    'while preserving the most significant information. The columns of U corresponding to the top k singular values ', ...
    'are known as “eigenfaces.” These eigenfaces are essentially the principal components of the face space, ', ...
    'capturing features like facial contours, eyes, noses, and mouth positions that best explain the variance in the dataset.']);
add(sec2, para5);

% Add eigenfaces image (saved as 'eigenfaces.png')
img_eigen = Image('eigenfaces.png');
img_eigen.Width = '6in';
img_eigen.Height = '4.5in';
cap_eigen = Paragraph('Figure 2: Top 16 extracted eigenfaces using SVD.');
add(sec2, img_eigen);
add(sec2, cap_eigen);

add(rpt, sec2);

%% Section 3: Feature Extraction and Classification with SVM
sec3 = Section('Feature Extraction and Classification with SVM');

para6 = Paragraph([ ...
    'Once we have extracted the eigenfaces, each face image can be represented as a linear combination ', ...
    'of these eigenfaces. Projecting the original images onto the reduced eigenface space yields a feature vector ', ...
    'of significantly smaller dimension, capturing the most discriminative facial features.']);

add(sec3, para6);

para7 = Paragraph([ ...
    'These low-dimensional feature vectors form the input to a supervised classifier. In this project, ', ...
    'we employ a Support Vector Machine (SVM) classifier. SVMs are chosen for their robustness and effectiveness ', ...
    'in high-dimensional spaces, and when combined with a one-vs-one or Error-Correcting Output Code (ECOC) scheme, ', ...
    'they can handle multiple classes.']);

add(sec3, para7);

para8 = Paragraph([ ...
    'The result is a facial recognition pipeline: Raw images → Preprocessing → SVD (for eigenfaces) → ', ...
    'Feature Extraction (projecting onto eigenfaces) → SVM Classification.']);

add(sec3, para8);

% Add a feature space visualization (saved as 'feature_space.png')
img_feat = Image('feature_space.png');
img_feat.Width = '6in';
img_feat.Height = '4.5in';
cap_feat = Paragraph('Figure 3: Visualization of the top 3 PCA/SVD-derived features of the dataset.');
add(sec3, img_feat);
add(sec3, cap_feat);

add(rpt, sec3);

%% Section 4: HPC Integration and Scalability
sec4 = Section('HPC Integration and Scalability');

para9 = Paragraph([ ...
    'High-Performance Computing (HPC) resources enable scaling the SVD and classification stages to very large datasets. ', ...
    'MATLAB’s Parallel Computing Toolbox allows distributing computations across multiple workers, ', ...
    'either on a local multicore machine or on a computing cluster. By utilizing parfor loops, distributed arrays, ', ...
    'and parallel-enabled functions (like fitcecoc with parallel options), we significantly reduce computation times.']);

add(sec4, para9);

para10 = Paragraph([ ...
    'In this project, the following HPC techniques are employed:', ...
    '1. Parallelized Reading of Images: Using parallel loops to speed up I/O and image preprocessing.', ...
    '2. Parallelized SVD Computation: Although SVD can be computationally expensive, especially on large datasets, ', ...
    '   MATLAB can offload these computations to multiple cores or to GPU resources if available.', ...
    '3. Parallelized Training of Classifiers: The classification (fitcecoc) can run in parallel, reducing training time significantly.']);
add(sec4, para10);

para11 = Paragraph([ ...
    'These HPC approaches ensure that the pipeline can handle increasingly large datasets without incurring prohibitive runtime costs.']);
add(sec4, para11);

add(rpt, sec4);

%% Section 5: Model Evaluation
sec5 = Section('Model Evaluation');

para12 = Paragraph([ ...
    'To assess model performance, we compute a confusion matrix and ROC curves. The confusion matrix visualizes how well ', ...
    'the classifier distinguishes among different individuals, and the ROC curve shows the trade-off between True Positive ', ...
    'and False Positive rates. In practice, evaluating on subsets of classes helps gauge classifier effectiveness and reveals ', ...
    'areas needing improvement.']);
add(sec5, para12);

% Add confusion matrix and ROC images
img_conf = Image('confusion_matrix.png');
img_conf.Width = '6in';
img_conf.Height = '4.5in';
cap_conf = Paragraph('Figure 4: Confusion Matrix for a subset of 5 classes.');
add(sec5, img_conf);
add(sec5, cap_conf);

img_roc = Image('roc_metrics.png');
img_roc.Width = '6in';
img_roc.Height = '4.5in';
cap_roc = Paragraph('Figure 5: ROC Curve for a subset of 10 classes, illustrating classifier sensitivity and specificity.');
add(sec5, img_roc);
add(sec5, cap_roc);

add(rpt, sec5);

%% Section 6: Conclusion and Future Work
sec6 = Section('Conclusion and Future Work');

para13 = Paragraph([ ...
    'This project demonstrates a scalable facial recognition pipeline that leverages SVD for eigenface extraction and SVM for classification. ', ...
    'By integrating MATLAB’s parallel computing capabilities, we ensure that the approach scales well, handling larger datasets efficiently.']);

para14 = Paragraph([ ...
    'Future work could explore advanced feature extraction methods, such as deep learning-based embeddings (e.g., using CNNs), and incorporate ', ...
    'GPU acceleration for even faster computations. These enhancements can further improve both the accuracy and efficiency of the facial recognition system.']);

add(sec6, para13);
add(sec6, para14);
add(rpt, sec6);

%% Close and Generate Report
close(rpt);
disp('Report generation completed. Opening the report...');
rptview(rpt);
