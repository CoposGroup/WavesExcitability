% lap2d.m
%
% form the (scaled) matrix for the 2D Laplacian for Neumann boundary
% conditions on a rectangular node-centered nx by ny grid
%
% input: nx -- number of grid points in x-direction (no bdy pts)
% ny -- number of grid points in y-directio
%
% output: L2 -- (nx*ny) x (nx*ny) sparse matrix for discrete Laplacian

function L2 = lap2d( nx,ny )
    
    % make 1D Laplacians
    %
    Lx = lap1d(nx);
    Ly = lap1d(ny);
    
    % Neumann BC on y-direction
    Lx(1,1) = -1;
    Lx(1,2) = 1;
    Lx(nx,nx-1) = 1;
    Lx(nx,nx) = -1;
    
    % Neumann BC on x-direction
    Ly(1,1) = -1;
    Ly(1,2) = 1;
    Ly(ny,ny-1) = 1;
    Ly(ny,ny) = -1;
    
    % make 1D identities
    %
    Ix = speye(nx);
    Iy = speye(ny);
    
    % form 2D matrix from kron
    %
    L2 = kron(Iy,Lx) + kron(Ly,Ix);
    
end