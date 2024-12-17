import mlreportgen.report.*
import mlreportgen.dom.*

rpt = Report('GPU_SVD_Report', 'pdf');
add(rpt, TitlePage('Title', 'GPU-Accelerated SVD', 'Author', 'Marek Rychlik'));
add(rpt, Chapter('Title', 'Introduction', 'Content', 'This report explains GPU-accelerated SVD...'));
% Add more sections programmatically
close(rpt);

