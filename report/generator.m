%% Setup the Report
import mlreportgen.report.*
import mlreportgen.dom.*

% Define the output report name
pdfRpt = Report('SimpleDocumentationReport', 'pdf');

% Add Title Page
titlePage = TitlePage;
titlePage.Title = 'Simple MATLAB Code Report';
titlePage.Subtitle = 'Generated by MATLAB Report Generator';
titlePage.Author = 'Your Name';
add(pdfRpt, titlePage);

% Add Table of Contents
toc = TableOfContents;
add(pdfRpt, toc);

% Find MATLAB files (*.m) in the current directory
files = dir('*.m');

% Check if there are any files to process
if isempty(files)
    warning('No .m files found in the current directory.');
else
    % Iterate through each file
    for i = 1:length(files)
        filename = files(i).name;

        % Add a chapter for each file
        chapter = Chapter();
        chapter.Title = strrep(filename, '_', '\_'); % Escape underscores for LaTeX compatibility

        % Read the file content
        try
            codeText = fileread(filename);
        catch
            warning('Could not read file: %s', filename);
            continue;
        end

        % Add file content as a code block
        codeBlock = Code(codeText);
        add(chapter, codeBlock);

        % Add the chapter to the report
        add(pdfRpt, chapter);
    end
end

% Finalize and view the report
close(pdfRpt);
rptview(pdfRpt);
