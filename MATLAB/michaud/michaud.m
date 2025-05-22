function michaud()
    
    dt = 0.001;
    Nt = 100000;
    Nx = 100;
    tplot = 100;
    
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
    
    RT = 0.1+0.9*rand(Nx*Nx,1);
    RD = 0.1*ones(Nx*Nx,1);
    F = zeros(Nx*Nx,1);
    
    RTnew = zeros(Nx*Nx,1);
    RDnew = zeros(Nx*Nx,1);
    RFnew = zeros(Nx*Nx,1);
    
    sigma = 0.75;
    s = 4;
    mu = 1;
    dWt = 1000;
    dW = reshape(cgf2d(s, sigma, mu, [Nx, Nx]),Nx*Nx,1);

    L = lap2d_periodic(Nx, Nx);
    
    %start the FTCS
    for t = 1:Nt
        R = reaction(RT,RD,F, k0, k1, k2, k3, k4, alpha, beta);
    
        RTnew = RT + dt*(R+Drt*L*RT);
        RDnew = RD + dt*(k5-k6*RD-R+Drd*L*RD);
        Fnew = F + dt*(k7 + (k8*RT.^2)./(1+k9*RT.^2)-k10*dW.*F+Df*L*F);
    
        RT = RTnew;
        RD = RDnew;
        F = Fnew;
    
        if mod(t,dWt)==0
            dW = reshape(cgf2d(s, sigma, mu, [Nx, Nx]),Nx*Nx,1);
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