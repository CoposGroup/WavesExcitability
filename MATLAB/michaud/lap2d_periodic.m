function L2 = lap2d_periodic(nx, ny)
    % Create 1D periodic Laplacians
    e = ones(nx, 1);
    Lx = spdiags([e -2*e e], [-1 0 1], nx, nx);
    Lx(1, end) = 1;
    Lx(end, 1) = 1;

    e = ones(ny, 1);
    Ly = spdiags([e -2*e e], [-1 0 1], ny, ny);
    Ly(1, end) = 1;
    Ly(end, 1) = 1;

    % Identity matrices
    Ix = speye(nx);
    Iy = speye(ny);

    % 2D Laplacian with periodic BCs
    L2 = kron(Iy, Lx) + kron(Ly, Ix);
end