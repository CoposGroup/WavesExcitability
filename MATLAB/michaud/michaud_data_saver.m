clear; close all; clc
rng(42);                                % reproducibility

%% ----------------------------------------------------------------
% 1. CONSTANTS THAT NEVER CHANGE
%% ----------------------------------------------------------------
k0 = 0.00625;  k1 = 0.3125;  k2 = 1;
k3 = 0.0625;   k4 = 0.05625;
k5 = 0.0625;   k6 = 0.02083;
k7 = 0.001875; k8 = 0.140625*2;
k9 = 0.25;     k10 = 0.025;
Drt = 0.08;    Drd = 0.4;
Df0 = 0.001;                       % <-- base value you asked to vary
alpha = 1;    beta  = 1;

dt = 0.1;      dx = 1.0;
Nt = 1000;     Nx = 100;
tplot = 1;

nFrames  = floor(Nt/tplot) + 1;
sigma = 0.75; s = 4; mu = 1;         % noise parameters
dWt = 10/dt;                         % refresh period for noise

L = lap2d(Nx, Nx)/dx^2;              % Laplacian operator

%% ----------------------------------------------------------------
% 2. LIST OF FACTORS WE WANT TO TRY
%% ----------------------------------------------------------------
dfFactors = [1 2];           % <- put any multipliers you like
results = struct();                  % structure to hold in-memory copies

%% ----------------------------------------------------------------
% 3. MAIN LOOP OVER THOSE FACTORS
%% ----------------------------------------------------------------
for m = dfFactors
    % ---- names & parameters for THIS run ----
    Df  = Df0 * m;                              % scaled diffusion
    tag = sprintf('dfx%d', m);                  % ‘dfx20’, ‘dfx50’, …
    outfile = sprintf('michaud_simulation_%s.mat', tag);

    % ---- reset all state variables ----
    RT = 0.1 + 0.9*rand(Nx*Nx,1);
    RD = 0.1 * ones(Nx*Nx,1);
    F  = zeros(Nx*Nx,1);

    frames3D = zeros(Nx, Nx, nFrames);
    frameIdx = 1;
    frames3D(:,:,frameIdx) = reshape(RT, Nx, Nx);

    dW = cgf2d(s, sigma, mu, [Nx, Nx]);         % first noise field

    % ---- run the reaction–diffusion simulation ----
    for t = 1:Nt/dt
        R = (k0 + alpha*k1*RT.^3./(1 + k2*RT.^2)).*RD ...
            - (k3 + k4*(1 + beta)*F).*RT;

        RT = RT + dt*(R            + Drt*L*RT);
        RD = RD + dt*(k5 - k6*RD - R + Drd*L*RD);
        F  = F  + dt*(k7 + (k8*RT.^2)./(1 + k9*RT.^2) ...
              - k10*reshape(dW, Nx*Nx, 1).*F + Df*L*F);

        if mod(t, dWt) == 0,  dW = cgf2d(s, sigma, mu, [Nx, Nx]);  end
        if mod(t, round(tplot/dt)) == 0
            frameIdx = frameIdx + 1;
            frames3D(:,:,frameIdx) = reshape(RT, Nx, Nx);
        end
    end

    % ---- stack frames + stash results ----
    RTstack = permute(frames3D, [3 1 2]);
    results.(tag) = RTstack;                    % keeps it in memory
    save(outfile, 'RTstack', '-v7.3');          % writes to disk
    fprintf('Finished %s (Df = %.4g)\n', tag, Df);
end
