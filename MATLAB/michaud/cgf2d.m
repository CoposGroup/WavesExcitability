% cgf2d.m
% 
% generate a 2d correlated gaussian field
% 
% input:
%   s     - correlation length
%   sigma - desired standard deviation
%   mu    - desired mean
%   dim   - [nx, ny] size of the field
%
% output:
%   Z     - 2D correlated Gaussian random field
function Z = cgf2d(s, sigma, mu, dim)
    nx = dim(1);
    ny = dim(2);
    dx = 1.0;

    % Extended grid for covariance (circulant embedding)
    x = [0:nx-1, -nx+1:-1] * dx;
    y = [0:ny-1, -ny+1:-1] * dx;
    [X, Y] = meshgrid(x, y);
    R = sqrt(X.^2 + Y.^2);

    % Gaussian covariance
    C = exp(-R.^2 / (2 * s^2));
    S = real(fft2(ifftshift(C)));
    S(S < 0) = 0;

    % Generate white noise on extended grid
    W = randn(size(S));

    % Filter in frequency domain
    Z_ext = real(ifft2(fft2(W) .* sqrt(S)));

    % Crop back to original size
    Z = Z_ext(1:ny, 1:nx);

    % Normalize and scale
    Z = (Z - mean(Z(:))) / std(Z(:));
    Z = mu + sigma * Z;
end