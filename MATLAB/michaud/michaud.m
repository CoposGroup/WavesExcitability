function michaud()

    clear; close all; clc
    
    dt = 0.01;
    dx = 1.0;
    Nt = 100000;
    Nx = 100;
    tplot = 1 / dt;
    
    % define parameters
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
    
    % set initial concentrations
    RT = 0.1+0.9*rand(Nx*Nx,1);
    RD = 0.1*ones(Nx*Nx,1);
    F = zeros(Nx*Nx,1);
    
    % initialize dW 
    sigma = 0.75;
    s = 4;
    mu = 1;
    dWt = 10 / dt;
    dW = cgf2d(s, sigma, mu, [Nx, Nx]);

    % set laplacian matrix
    L = lap2d(Nx, Nx)/dx^2;
    
    % 
    for t = 1:Nt

        % set reaction term
        R = (k0 + alpha * k1 * RT.^3 ./ (1 + k2 * RT.^2)) .* RD - (k3 + k4 * (1 + beta) * F) .* RT;
    
        % set new concentrations
        RTnew = RT + dt*(R+Drt*L*RT);
        RDnew = RD + dt*(k5 - k6*RD - R + Drd*L*RD);
        Fnew = F + dt*(k7 + (k8*RT.^2) ./ (1 + k9*RT.^2) - k10*reshape(dW,Nx*Nx,1).*F + Df*L*F);
    
        % update concentrations
        RT = RTnew;
        RD = RDnew;
        F = Fnew;
    
        if mod(t,dWt)==0
            dW = cgf2d(s, sigma, mu, [Nx, Nx]);
        end
    
        if mod(t,tplot)==0
            pcolor(reshape(RT,Nx,Nx));
            colormap('spring');
            shading flat;
            clim([0 1]);
            colorbar;
            set(gcf,'color','w');
            axis square;
            title(sprintf('Time = %4.2f s',t*dt),'fontweight','normal');
            axis off;
            pause(0.001);
        end
    end
end