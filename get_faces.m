% Step 1: Define the URL and output file name
url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz';
outputFile = 'lfw.tgz';

% Step 2: Download the file
disp('Downloading dataset...');
websave(outputFile, url);
disp('Download complete.');

% Step 3: Extract the file
disp('Extracting dataset...');
untar(outputFile, 'lfw'); % Extracts to a folder named 'lfw'
disp('Extraction complete.');

% Step 4: Verify the contents
disp('Listing extracted contents...');
contents = dir('lfw');
disp({contents.name});
