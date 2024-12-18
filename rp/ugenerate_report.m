import mlreportgen.report.*
import mlreportgen.dom.*

% Create a new report
rpt = Report('FacialRecognitionReport', 'pdf');

%% Add Title Page
tp = TitlePage;
tp.Title = 'Facial Recognition Model Training';
tp.Subtitle = 'PCA and SVM Implementation';
tp.Author = 'Marek Rychlik';
tp.Image = which('face.png'); % Optional: Add a title image
add(rpt, tp);

%% Add Table of Contents
add(rpt, TableOfContents);

%% Section 1: Project Overview
sec1 = Section('Project Overview');

para1 = Paragraph(['This project implements a facial recognition system using MATLAB. ', ...
                   'The main objectives are to preprocess image data, extract features using ', ...
                   'Principal Component Analysis (PCA), and classify faces using a Support Vector Machine (SVM).']);
add(sec1, para1);

para2 = Paragraph(['The key steps include:']);
list1 = UnorderedList({'Data Preparation: Resize, grayscale conversion, and flatten images.', ...
                       'Feature Extraction: Apply PCA to extract dominant facial features (eigenfaces).', ...
                       'Classification: Train a multi-class SVM using the extracted features.', ...
                       'Evaluation: Measure performance using Confusion Matrix and ROC metrics.'});
add(sec1, para2);
add(sec1, list1);
add(rpt, sec1);

%% Section 2: Data Preprocessing and PCA
sec2 = Section('Data Preprocessing and PCA');

para3 = Paragraph(['The image dataset consists of grayscale images resized to ', ...
                   '128x128 pixels. Each image is flattened into a vector, forming a large matrix of image data.']);
add(sec2, para3);

para4 = Paragraph(['PCA is applied to reduce the dimensionality of the data, extracting the most significant ', ...
                   'components known as eigenfaces. These eigenfaces capture the key facial features necessary ', ...
                   'for distinguishing between individuals.']);
add(sec2, para4);

% Add eigenfaces image
img1 = Image('eigenfaces.png');
img1.Width = '6in'; % Larger image to fit half a page
img1.Height = '4.5in';
caption1 = Paragraph('Figure 1: Top 16 Eigenfaces extracted using PCA.');
add(sec2, img1);
add(sec2, caption1);

para5 = Paragraph(['PCA uses Singular Value Decomposition (SVD) to decompose the data matrix into three components: ', ...
                   'U (eigenfaces), S (singular values), and V (coefficients). The top K eigenfaces are selected to form ', ...
                   'a reduced feature space for classification.']);
add(sec2, para5);

add(rpt, sec2);

%% Section 3: Model Training and Evaluation
sec3 = Section('Model Training and Results');

para6 = Paragraph(['The extracted features (eigenfaces) are used to train a Support Vector Machine (SVM) classifier. ', ...
                   'A multi-class SVM with Error-Correcting Output Codes (ECOC) is used to handle multiple classes.']);
add(sec3, para6);

% Add scatter plot of feature space
para7 = Paragraph('The scatter plot below visualizes the top 3 PCA features used for classification:');
add(sec3, para7);
img2 = Image('feature_space.png');
img2.Width = '6in'; % Larger image to fit half a page
img2.Height = '4.5in';
caption2 = Paragraph('Figure 2: Scatter plot of the top 3 PCA features.');
add(sec3, img2);
add(sec3, caption2);

% Add confusion matrix
para8 = Paragraph(['The confusion matrix provides an evaluation of the model''s performance, showing the relationship ', ...
                   'between predicted and true labels. To enhance clarity, only a subset of classes is displayed.']);
add(sec3, para8);
img3 = Image('confusion_matrix.png');
img3.Width = '6in'; % Larger image to fit half a page
img3.Height = '4.5in';
caption3 = Paragraph('Figure 3: Confusion Matrix for a subset of 5 classes.');
add(sec3, img3);
add(sec3, caption3);

% Add ROC curve explanation
para9 = Paragraph(['The ROC curve illustrates the trade-off between the True Positive Rate (TPR) and the False Positive Rate (FPR). ', ...
                   'A subset of 10 classes is shown, with the legend placed under the lower-right diagonal for better visualization.']);
add(sec3, para9);
img4 = Image('roc_metrics.png');
img4.Width = '6in'; % Larger image to fit half a page
img4.Height = '4.5in';
caption4 = Paragraph('Figure 4: ROC Curve for a subset of 10 classes.');
add(sec3, img4);
add(sec3, caption4);

% Summary paragraph
para10 = Paragraph(['The trained SVM model successfully classifies images using the PCA-extracted features. ', ...
                    'The confusion matrix and ROC metrics confirm the model''s accuracy and ability to generalize across classes.']);
add(sec3, para10);

add(rpt, sec3);

%% Section 4: Conclusion and Future Work
sec4 = Section('Conclusion and Future Work');

para11 = Paragraph(['In this project, a facial recognition pipeline was developed using PCA and SVM. ', ...
                    'PCA effectively reduced the dimensionality of the image data, while SVM provided robust classification.']);
add(sec4, para11);

para12 = Paragraph(['Future improvements could involve incorporating deep learning techniques, such as Convolutional Neural Networks (CNNs), ', ...
                    'to further enhance performance and scalability for larger datasets.']);
add(sec4, para12);

add(rpt, sec4);

%% Close and Generate Report
close(rpt);
disp('Report generation completed. Opening the report...');
rptview(rpt);
