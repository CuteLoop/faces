Here is the well-documented version of your MATLAB code:

```matlab
% XFORM Function
%
% The xform function is used to convert an input image to grayscale. The
% function takes an image as input and returns the grayscale version of the
% image. The function uses MATLAB's built-in im2gray function to perform
% the conversion.
%
% Syntax: [varargout] = xform(I, varagin)
%
% Input:
%   I - Input image. It can be a grayscale or color image.
%   varagin - Additional arguments (currently not used in this function).
%
% Output:
%   varargout - Output image in grayscale. The number of output arguments
%               depends on the number of input arguments.

function [varargout] = xform(I, varagin)
    % Convert the input image to grayscale using MATLAB's im2gray function.
    % The im2gray function converts the input image to grayscale by
    % eliminating the hue and saturation information while retaining the
    % luminance.
    G = im2gray(I)
end
```

Please note that the original code does not use the `varagin` input argument, and it does not assign the output `varargout`. Also, the function does not return the grayscale image `G`. You might want to revise the function to use these variables properly.