clear; close all; clc

% set random seed
rng(42);

% set model parameters
Drt = 0.1;
Df = 0.1;

% set simulation parameters
dt = 0.01; % time step
dx = 1.0; % x step
Nt = 1000; % run time (s)
Nx = 100; % number of cells
tplot = 1; % frame plot interval (s)

% set initial concentrations
RT = 0.1+0.9*rand(Nx*Nx,1);
F = zeros(Nx*Nx,1);

% initialize stochastic noise term (dW)
sigma = 0.75;
s = 4;
mu = 1;
dWt = 10 / dt;
dW = reshape(cgf2d(s, sigma, mu, [Nx, Nx]),Nx*Nx,1);

% create laplacian matrix
L = lap2d(Nx, Nx)/dx^2;

% set up figure
plt = pcolor(reshape(RT, Nx, Nx));
shading flat;
colormap(parula);
clim([0 1]);
cbar = colorbar;
cbar.Label.String = '[RT]';
axis square off;
title('Time = 0 s');

% create video writer
vidobj = VideoWriter('michaud_simplified.mp4','mpeg-4');
vidobj.FrameRate = 15;
open(vidobj);

% Capture initial frame
frame = getframe(gcf);
writeVideo(vidobj, frame);

% reaction loop
for t = 1:Nt/dt

    % set new concentrations
    RTnew = RT + dt * (RT.^2.*(1-RT) - 0.4*F.*RT + Drt*L*RT);
    Fnew = F + dt * (0.05*RT - 0.025*dW.*F + Df*L*F);

    % update concentrations
    RT = RTnew;
    F = Fnew;

    % update stochastic noise term
    if mod(t,dWt)==0
        dW = reshape(cgf2d(s, sigma, mu, [Nx, Nx]),Nx*Nx,1);
    end

    % plot figure and save frame
    if mod(t,tplot/dt)==0
        set(plt, 'CData', reshape(RT, Nx, Nx));
        title(sprintf('Time = %4.0f s', t*dt));
        drawnow;
        frame = getframe(gcf);
        writeVideo(vidobj, frame);
    end
end

close(vidobj);