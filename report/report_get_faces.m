%% Initial Setup
% This script downloads and extracts a dataset from a given URL. The dataset is then saved to a specified output file.

% Define the URL of the dataset
% The URL points to the location of the dataset on the web.
url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz';

% Define the output file name
% The output file is the name of the file where the downloaded dataset will be saved.
outputFile = 'lfw.tgz';

%% Downloading the Dataset
% This section downloads the dataset from the specified URL and saves it to the output file.

% Display a message indicating the download process has started
disp('Downloading dataset...');

% Download the dataset
% The websave function downloads the dataset from the URL and saves it to the output file.
websave(outputFile, url);

% Display a message indicating the download process has completed
disp('Download complete.');

%% Extracting the Dataset
% This section extracts the downloaded dataset.

% Display a message indicating the extraction process has started
disp('Extracting dataset...');

% Extract the dataset
% The untar function extracts the dataset from the output file and saves it to a folder named 'lfw'.
untar(outputFile, 'lfw');

% Display a message indicating the extraction process has completed
disp('Extraction complete.');

%% Verifying the Contents
% This section lists the contents of the extracted dataset to verify that the download and extraction processes were successful.

% Display a message indicating the verification process has started
disp('Listing extracted contents...');

% List the contents of the extracted dataset
% The dir function returns a list of files in the 'lfw' directory.
contents = dir('lfw');

% Display the names of the files in the extracted dataset
disp({contents.name});