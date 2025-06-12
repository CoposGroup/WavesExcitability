clear; close all; clc

% set random seed
rng(42);

% set model parameters
Da = 0.1; % A diffusion rate
Db = 10; % B diffusion rate
delta = 1; % # basal rate of inactivation
gamma = 1.0; % GTPase autocatalytic activation rate
K = 1; % saturation parameter
k0 = 0.067; % basal GEF conversion rate

% set simulation parameters
dt = 0.01; % time step
dx = 1.0; % x step
Nt = 1000; % run time (s)
Nx = 100; % number of cells
tplot = 1; % frame plot interval (s)

% set initial concentrations
A = 2 * 0.2683312 * rand(Nx*Nx, 1);
B = 2 * ones(Nx*Nx, 1);

% create laplacian matrix
L = lap2d(Nx, Nx)/dx^2;

% set up figure
plt = pcolor(reshape(A, Nx, Nx));
shading flat;
colormap(parula);
clim([0 2]);
cbar = colorbar;
cbar.Label.String = '[A]';
axis square off;
title('Time = 0 s');

% create video writer
vidobj = VideoWriter('mori2D.mp4','mpeg-4');
vidobj.FrameRate = 15;
open(vidobj);

% Capture initial frame
frame = getframe(gcf);
writeVideo(vidobj, frame);

% reaction loop
for t = 1:Nt/dt

    % set reaction term
    R = (k0 + gamma*A.^2./(K^2 + A.^2)).*B - delta*A;

    % update concentrations
    A = A + dt * (Da * L * A + R);
    B = B + dt * (Db * L * B - R);

    % plot figure and save frame
    if mod(t,tplot/dt)==0
        set(plt, 'CData', reshape(A, Nx, Nx));
        title(sprintf('Time = %4.0f s', t*dt));
        drawnow;
        frame = getframe(gcf);
        writeVideo(vidobj, frame);
    end
end

close(vidobj);