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
L = 10; % cell diameter (µm)

% set simulation parameters
Nx = 100; % number of cells
dt = 0.01; % time step
Nt = 1000; % run time (s)
tplot = 1; % animation frame interval (s)

% create sparse double derivative matrix
e = ones(Nx, 1);
D2 = spdiags([e, -2*e, e], [-1, 0, 1], Nx, Nx);

% apply Neumann boundary conditions
D2(1,2) = 2;
D2(end,end-1) = 2;

% set initial concentrations
A = 2 * 0.2683312 * rand(Nx, 1);
B = 2 * ones(Nx, 1);

% set up figure
x = 1:Nx;
plt = plot(x, A, 'LineWidth', 2);
ylim([0, 1.5]);
xlim([1, Nx]);
xlabel('Cell Diameter');
ylabel('[A]');
title('Time = 0 s');
grid on;

% create video writer
vidobj = VideoWriter('mori.mp4','mpeg-4');
vidobj.FrameRate = 15;
open(vidobj);

% capture inital frame
frame = getframe(gcf);
writeVideo(vidobj, frame);

% reaction loop
for t = 1:Nt/dt

    % set reaction term
    R = (k0 + gamma*A.^2./(K^2 + A.^2)).*B - delta*A;

    % update concentrations
    A = A + dt * (Da * D2 * A + R);
    B = B + dt * (Db * D2 * B - R);

    % plot figure and save frame
    if mod(t,tplot/dt)==0
        set(plt, 'YData', A);
        title(sprintf('Time = %4.0f s', t*dt));
        drawnow;
        frame = getframe(gcf);
        writeVideo(vidobj, frame);
    end
end

close(vidobj);