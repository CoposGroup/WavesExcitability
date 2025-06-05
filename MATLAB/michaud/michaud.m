clear; close all; clc

% set random seed
rng(42);

% set model parameters
k0 = 0.00625;
k1 = 0.3125;
k2 = 1;
k3 = 0.0625;
k4 = 0.05625;
k5 = 0.0625;
k6 = 0.02083;
k7 = 0.001875;
k8 = 0.140625;
k9 = 0.25;
k10 = 0.025;
Drt = 0.08;
Drd = 0.4;
Df = 0.001;
alpha = 1;
beta = 1;

% set simulation parameters
dt = 0.01; % time step
dx = 1.0; % x step
Nt = 1000; % run time (s)
Nx = 100; % number of cells
tplot = 1; % frame plot interval (s)

% set initial concentrations
RT = 0.1+0.9*rand(Nx*Nx,1);
RD = 0.1*ones(Nx*Nx,1);
F = zeros(Nx*Nx,1);

% initialize stochastic noise term (dW)
sigma = 0.75;
s = 4;
mu = 1;
dWt = 10 / dt;
dW = cgf2d(s, sigma, mu, [Nx, Nx]);

% create laplacian matrix
L = lap2d(Nx, Nx)/dx^2;

% set up figure
plt = pcolor(reshape(RT, Nx, Nx));
shading flat;
colormap(parula);
clim([0 5]);
cbar = colorbar;
cbar.Label.String = '[RT]';
axis square off;
title('Time = 0 s');

% create video writer
vidobj = VideoWriter('michaud.mp4','mpeg-4');
vidobj.FrameRate = 15;
open(vidobj);

% Capture initial frame
frame = getframe(gcf);
writeVideo(vidobj, frame);

% reaction loop
for t = 1:Nt/dt

    % set reaction term
    R = (k0 + alpha * k1 * RT.^3 ./ (1 + k2 * RT.^2)) .* RD - (k3 + k4 * (1 + beta) * F) .* RT;

    % set new concentrations
    RTnew = RT + dt * (R+Drt*L*RT);
    RDnew = RD + dt * (k5 - k6*RD - R + Drd*L*RD);
    Fnew = F + dt * (k7 + (k8*RT.^2) ./ (1 + k9*RT.^2) - k10*reshape(dW,Nx*Nx,1).*F + Df*L*F);

    % update concentrations
    RT = RTnew;
    RD = RDnew;
    F = Fnew;

    % update stochastic noise term
    if mod(t,dWt)==0
        dW = cgf2d(s, sigma, mu, [Nx, Nx]);
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