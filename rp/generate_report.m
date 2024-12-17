import mlreportgen.report.*
import mlreportgen.dom.*

% Create a new report
rpt = Report('FacialRecognitionReport', 'pdf');
rpt.Title = 'Facial Recognition Model Report';

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

para1 = Paragraph(['This project demonstrates a facial recognition pipeline implemented in MATLAB. ', ...
                   'The main goal is to classify images of different individuals using ', ...
                   'Principal Component Analysis (PCA) for feature extraction and ', ...
                   'Support Vector Machines (SVM) for classification.']);
para1.Style = {Bold};
add(sec1, para1);

para2 = Paragraph(['PCA reduces the dimensionality of image data by finding the most significant features ', ...
                   '(eigenfaces) that represent the dataset. The reduced features are then used to train ', ...
                   'a multi-class SVM model to recognize faces.']);
add(sec1, para2);

list1 = UnorderedList({'Data Preparation: Resizing, Grayscale Conversion, and Flattening Images.', ...
                       'Feature Extraction: PCA using Singular Value Decomposition (SVD).', ...
                       'Classification: Multi-class SVM using ECOC.', ...
                       'Evaluation: Confusion Matrix and ROC Metrics.'});
add(sec1, list1);

add(rpt, sec1);

%% Section 2: Data Processing and PCA
sec2 = Section('Data Processing and PCA');

para3 = Paragraph(['The image dataset is first preprocessed by resizing all images to ', ...
                   'a fixed size of 128x128 pixels and converting them to grayscale. Each image is flattened ', ...
                   'into a vector, resulting in a large dataset matrix.']);
add(sec2, para3);

para4 = Paragraph(['Principal Component Analysis (PCA) is applied to reduce the dimensionality of this matrix. ', ...
                   'PCA identifies the most significant directions (eigenfaces) in the data, capturing key features ', ...
                   'for distinguishing between individuals.']);
add(sec2, para4);

% Add eigenfaces image
img1 = Image('eigenfaces.png');
img1.Width = '4in';
img1.Height = '3in';
caption1 = Paragraph('Figure 1: Top 16 Eigenfaces extracted using PCA.');
add(sec2, img1);
add(sec2, caption1);

% Add explanation of SVD
para5 = Paragraph(['Singular Value Decomposition (SVD) is the method used to compute PCA. The image data matrix ', ...
                   'is decomposed into three components: ', ...
                   'U (eigenfaces), S (singular values), and V (coefficients). The top eigenfaces are selected ', ...
                   'to form a reduced feature space.']);
add(sec2, para5);

add(rpt, sec2);

%% Section 3: Model Training and Results
sec3 = Section('Model Training and Results');

para6 = Paragraph(['Once the PCA-transformed features are extracted, a Support Vector Machine (SVM) classifier ', ...
                   'is trained using the top features. The SVM model utilizes a multi-class ECOC (Error-Correcting Output Codes) ', ...
                   'approach to handle multiple classes efficiently.']);
add(sec3, para6);

% Add training explanation
para7 = Paragraph(['Training is performed on the reduced feature space, and model performance is evaluated using ', ...
                   'two key metrics: the confusion matrix and ROC (Receiver Operating Characteristic) metrics.']);
add(sec3, para7);

% Add confusion matrix image
img2 = Image('confusion_matrix.png');
img2.Width = '4in';
img2.Height = '3in';
caption2 = Paragraph('Figure 2: Confusion Matrix showing model predictions.');
add(sec3, img2);
add(sec3, caption2);

% Add ROC curve explanation
para8 = Paragraph(['The ROC curve illustrates the trade-off between the true positive rate and false positive rate ', ...
                   'for the model predictions. It helps assess the classifier''s ability to distinguish between classes.']);
add(sec3, para8);

% Add ROC metrics image
img3 = Image('roc_metrics.png');
img3.Width = '4in';
img3.Height = '3in';
caption3 = Paragraph('Figure 3: ROC Curve for evaluating model performance.');
add(sec3, img3);
add(sec3, caption3);

% Summary of results
para9 = Paragraph(['Overall, the trained model achieves accurate classification by leveraging PCA for dimensionality ', ...
                   'reduction and SVM for classification. The confusion matrix and ROC metrics demonstrate the model''s performance.']);
add(sec3, para9);

add(rpt, sec3);

%% Section 4: Conclusion
sec4 = Section('Conclusion');
para10 = Paragraph(['This project successfully demonstrates a facial recognition system using PCA and SVM. ', ...
                    'The steps include image preprocessing, dimensionality reduction via PCA, and classification using SVM. ', ...
                    'The model provides a robust solution for recognizing faces from a given dataset.']);
add(sec4, para10);

para11 = Paragraph(['Future improvements could include using deep learning approaches such as Convolutional Neural Networks (CNNs) ', ...
                    'to further enhance recognition accuracy and generalizability.']);
add(sec4, para11);

add(rpt, sec4);

%% Close and generate report
close(rpt);
rptview(rpt);
