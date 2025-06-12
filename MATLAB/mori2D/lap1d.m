% lap1d.m
%
% form the (scaled) 1D Laplacian for Dirichlet boundary conditions
% on a node-centered grid
%
% input: n -- number of grid points (no bdy pts)
%
% output: L -- n x n sparse matrix for discrete Laplacian
function L = lap1d(n)
    e=ones(n,1);
    L = spdiags([e -2*e e],[-1 0 1],n,n);
end