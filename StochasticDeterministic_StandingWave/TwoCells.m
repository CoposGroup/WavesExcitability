% Simulate competition for resources for actin filaments
% and fully coupled to stochastic biochemistry for polarity proteins
% on a circular cell membrane (periodic BC, 1.5D) with two cells
%
% Mechanics -- two actin networks: branched (a) and contractile (b)
% Polarity proteins : Rac (X) and Rho (Y)
%
% Last updated: 4/30/2024
% Katie Levandosky
% Calina Copos
addpath('./FigureAndMovieCode/freeze_colors')
addpath('../SingleCellCode_Published')

clear;
close all;
clc;

%coeff_vals=[1,10,100,1000];

%for c1_ind=4:4 %koffx,koffy,konx,kony
%    for c2_ind=2:2
%        for c1coeff_ind=3:3
%            for c2coeff_ind=2:2

%                if c1coeff_ind~=3 && c2coeff_ind~=3
%                    break
%                end

for c1_ind=1:1


save_matfile=0;
mat_location='';
move_cells=0;
writem=0;
%savelocation='./simulation_results/parameter_search_results/concentration_dependent_racrho/';
res_counters = [0,0,0,0,0,0,0]; %[coalign, collision, 1NP, 2NP, no, LF, dist. effort]

counter_ppp = 1;
ppp = 1;

% options=["koffx","koffy","konx","kony"];
%         if isfile(strcat(savelocation,...
%             int2str(coeff_vals(c1coeff_ind)),options(c1_ind),'C1_',...
%             int2str(coeff_vals(c2coeff_ind)),options(c2_ind),'C2','.xls'))
%             
%             res_counters=readmatrix(strcat(savelocation,...
%             int2str(coeff_vals(c1coeff_ind)),options(c1_ind),'C1_',...
%             int2str(coeff_vals(c2coeff_ind)),options(c2_ind),'C2','.xls'));
%             ppp=sum(res_counters(1:5))+1;
%             counter_ppp=sum(res_counters(1:5))+1;
%         end


        

while (ppp<=25)
    close all;
    savefigs=1;
    setnum=int2str(ppp);
    %savelocation='./simulation_results/results_signal/signal_switches2/resetRacRho/500stepsc2_9500stepsc1/forceindependent_branchedupbundleddownnosig_bundledupbrancheddownsig/7_5_-7_5';
    if c1_ind==1
        savelocation = '/Users/anamaria1/Downloads/Archives/';
        %savelocation='./simulation_results/results_nosignal/new_rho_bundled_rates/exp3';
    end
    if savefigs==1
        filenameCells=strcat(savelocation,'Cells_',setnum);
        filenameScatter=strcat(savelocation,'Scatter_',setnum);
    end

    vid = 0;
    if vid==1
        vidObjCells = VideoWriter(strcat(savelocation,'CellsVid_',setnum,'.mp4'),'MPEG-4');
        vidObjScatter = VideoWriter(strcat(savelocation,'ScatterVid_',setnum,'.mp4'),'MPEG-4');
    end

    counter_ppp = counter_ppp+1;

    clearvars -except counter_ppp vid vidObjCells vidObjScatter ppp ...
        savefigs filenameC1 filenameC2 savelocation ...
        filenameScatter filenameCells res_counters c1_vals c2_vals c1_ind ...
        c2_ind all_results_matrix polarize_time polarize_time_c1 ...
        polarize_time_c2 num_polarized num_pol_c1 num_pol_c2 countpol writem ...
        ka_ind kb_ind kc_ind kd_ind ka_vals kb_vals kc_vals kd_vals coeff_vals ...
        konx_ind koffx_ind kony_ind c1coeff_ind c2coeff_ind move_cells polarize_time_yes ...
        num_pol_lf polarize_time_lf num_pol_yes save_matfile mat_location ...
        kaa_vals kbb_vals kcc_vals kdd_vals kaa_ind kbb_ind kcc_ind kdd_ind ...
        add_actin_growth conc_dependent_racrho

    rng('shuffle');
    set(0,'DefaultFigureVisible','on')

    polarizedc1=0; %has cell1 polarized yet
    polc1_counter=0;
    polarizedc2=0;
    polc2_counter=0;
    pol_samedir=0; %have they polarized in the same direction yet
    samedir_counter=0; %how many timesteps in a row have they polarized in the same direction
    pol_lf=0;
    lf_counter=0;


    % Set actin filament parameters
    %
    Da      = 0.5;                  % diffusion coefficient for actin
    m0      = 2.0;                  % competition for actin monomers

    % Set polarity protein parameters
    %
    N       = 200;                  % total number of molecules in the cell (conserved)
    ron     = 0.001;                            % spontaneous association
    rfb     = 1.0;                  % enhanced association
    roff    = 0.09;                  % disaassociation
    D       = Da;                   % diffusion coefficient for membrane-bound particles

    % Set feedback (or coupling) strength
    %
    alpha = [2,2];
    beta = [2,2]; %first argument is away from overlap, second is on overlap

    % Set discretization
    %
    L      = 10.0;                  % cell length
    bper   = 0.25;                  % percent overlap
    dt     = 0.01;                  % temporal discretization
    Na     = 101;                   % number of space steps
    dxa    = 5.0/((Na-1)/2);        % spatial discretization
    Xa     = 0:dxa:L;
    Xb     = 0:dxa:L;
    pa     = dt*Da/(dxa^2);
    Tend   = 25.0;                  % total simulation time
    Nt     = Tend/dt;
    dx     = sqrt(2*D*dt);
    tplot  = 50;

    posx1 = zeros(N,Nt);              % array of positions of X(t) cell 1
    posy1 = zeros(N,Nt);              % array of positions of Y(t) cell 1

    posx2 = zeros(N,Nt);              % array of positions of X(t) cell 2
    posy2 = zeros(N,Nt);              % array of positions of Y(t) cell 2


    % Competition for limited resource (actin monomers) term
    %
    %F = @(U,V) -U.*U - m0*U.*V;
    F = @(U,V) -m0*U.*V;


    % Boundary between cells
    blen = floor(bper * L); % length of overlap between cell membranes
    boundC1 = (floor((Na-1)*3/4 - floor((Na-1)*bper/2)))+1:(floor((Na-1)*3/4 + floor((Na-1)*bper/2)))+1;
    boundC2 = (floor((Na-1)*1/4 - floor((Na-1)*bper/2)))+1:(floor((Na-1)*1/4 + floor((Na-1)*bper/2)))+1;

    % Signal
    signal=0;
    sigswitch_time=500;
    sigper=0.40;
    sigBound1 = (floor((Na-1)*1/8 - floor((Na-1)*sigper/2)))+1:(floor((Na-1)*1/8 + floor((Na-1)*sigper/2)))+1;
    sigBound1(sigBound1<=0)=sigBound1(sigBound1<=0)+Na;
    sigBound1(sigBound1>Na)=sigBound1(sigBound1>Na)-Na;
    sigBound2 = (floor((Na-1)*5/8 - floor((Na-1)*sigper/2)))+1:(floor((Na-1)*5/8 + floor((Na-1)*sigper/2)))+1;
    sigBound2(sigBound2<=0)=sigBound2(sigBound2<=0)+Na;
    sigBound2(sigBound2>Na)=sigBound2(sigBound2>Na)-Na;


    % options: 'uncoupled', 'racupc1-rhoupc2', 'rhoupc1-rhodownc2',
    % 'racupc1-rhoupc2-concdependent', 'rhoupc1-rhodownc2-forcedependent',
    % 'bundledupc1-branchedupc2', 'branched-bundled-crosspromotion',
    % 'ractorho-antagonism'
    pathway='uncoupled';



    % set antagonism (numRhoToRemove=0 and numRacToRemove=0 --> uncoupled)
    epsilon=0.5; % distance to detect other molecules (finding nearby rac/rho to remove)
    numRhoToRemove=0;
    numRacToRemove=0;
    counter1=0;
    counter2=0;

    % Set branched bundled coupling
    %
    % constants multiplied: da/dt=Ka1(a1(1+nrac))-a1^2, etc
    % branchedConst,bundledConst multiplied in contact region, otherwise 1
    branchedConst1 = 1.0;
    bundledConst1 = 1.0;
    branchedConst2 = 1.0;
    bundledConst2 = 1.0;
    Ka1=ones(Na,1);
    Kb1=ones(Na,1);
    Ka2=ones(Na,1);
    Kb2=ones(Na,1);
    Ka1(boundC1) = branchedConst1*Ka1(boundC1);
    Kb1(boundC1) = bundledConst1*Kb1(boundC1);
    Ka2(boundC2) = branchedConst2*Ka2(boundC2);
    Kb2(boundC2) = bundledConst2*Kb2(boundC2);

    % branched/bundled promotion/inhibition between cells
    % ka: how does branched affect branched
    % kb: how does bundled affect branched
    % kc: how does branched affect bundled
    % kd: how does bundled affect bundled
    ka_vals=0.9*[0,1,-1];
    kb_vals=0.9*[0,1,-1];
    kc_vals=0.9*[0,1,-1];
    kd_vals=0.9*[0,1,-1];
    ka_ind=1; %index of ka_vals
    kb_ind=1;
    kc_ind=1;
    kd_ind=1;


    % branched/bundled add/subtract in growth term
    kaa_vals=[0,7.5,-7.5];
    kbb_vals=[0,7.5,-7.5];
    kcc_vals=[0,7.5,-7.5];
    kdd_vals=[0,7.5,-7.5];
    kaa_ind=1;
    kbb_ind=1;
    kcc_ind=1;
    kdd_ind=1;


    % Set initial conditions for actin distribution
    %
    ictype = 2;
    %    1 = step in the middle
    %    2 = random
    %    3 = sigmoidal curve
    %    4 = odd condition #1: branched peak in the middle, contractile peaks at
    %    the front and rear cell
    %    5 = odd condition #2: both peaks in the middle (w/ noise)

    a1       = zeros(length(Xa),1);
    anew1    = zeros(length(Xa),1);
    b1       = zeros(length(Xb),1);
    bnew1    = zeros(length(Xb),1);

    a2       = zeros(length(Xa),1);
    anew2    = zeros(length(Xa),1);
    b2       = zeros(length(Xb),1);
    bnew2    = zeros(length(Xb),1);


    % (1) pulse in middle
    if (ictype==1)
        a1   = ones(N,1);
        anew1= ones(N,1);

        b1   = 0.5*ones(N,1);
        bnew1= 0.5*ones(N,1);

        a2   = ones(N,1);
        anew2= ones(N,1);

        b2   = 0.5*ones(N,1);
        bnew2= 0.5*ones(N,1);

        % (2) random
    elseif (ictype==2)
        a1 = 0.1 + 0.9.*rand(length(Xa),1);
        b1 = 0.1 + 0.9.*rand(length(Xb),1);

        a2 = 0.1 + 0.9.*rand(length(Xa),1);
        b2 = 0.1 + 0.9.*rand(length(Xb),1);

        % (3) arctangent
    elseif (ictype==3)
        steepness = 20;
        a1 = (tanh(steepness*(X1-0.375)) - tanh(steepness*(X1-1.125)) + 0.2)/2.2;
        b1 = (2 - tanh(steepness*(X1-0.375)) + tanh(steepness*(X1-1.125)) + 0.2)/2.2;

        a2 = (tanh(steepness*(X2-0.375)) - tanh(steepness*(X2-1.125)) + 0.2)/2.2;
        b2 = (2 - tanh(steepness*(X2-0.375)) + tanh(steepness*(X2-1.125)) + 0.2)/2.2;

        %a = (tanh(steepness*(X-0.5)) - tanh(steepness*(X-1.5)) + 0.2)/2.2;
        %b = (2 - tanh(steepness*(X-0.5)) + tanh(steepness*(X-1.5)) +0.2)/2.2;
    elseif (ictype==4)
        % (4) odd condition #1 (multiple peaks)
        steepness = 10;
        a1 = (1-cos(3*Xa*pi/5))/2; a1=a1';
        b1 = (tanh(steepness*(Xb-7.5))+1)/2 + (1-tanh(steepness*(Xb-2.5)))/2; b1=b1';

        a2 = (1-cos(3*Xa*pi/5))/2; a2=a2';
        b2 = (tanh(steepness*(Xb-7.5))+1)/2 + (1-tanh(steepness*(Xb-2.5)))/2; b2=b2';

    elseif (ictype==5)
        % (4) odd condition #2
        steepness = 10;
        b1 = (1-cos(Xb*pi/5))/2; b1=b1';
        a1 = (tanh(steepness*(Xa-7.5))+1)/2 + (1-tanh(steepness*(Xa-2.5)))/2; a1=a1';

        b2 = (1-cos(Xb*pi/5))/2; b2=b2';
        a2 = (tanh(steepness*(Xa-7.5))+1)/2 + (1-tanh(steepness*(Xa-2.5)))/2; a2=a2';

    elseif (ictype==6)
        % (5) odd condition #3
        mu = 1.8; sigma = 0.1;
        a1 = awgn(exp(-0.5*((Xa-mu)/sigma).^2)./(sigma*sqrt(32*pi)),20); a1=a1';
        a2 = awgn(exp(-0.5*((Xa-mu)/sigma).^2)./(sigma*sqrt(32*pi)),20); a2=a2';
        mu = 1.9; sigma = 0.1;
        b1 = awgn(exp(-0.5*((Xb-mu)/sigma).^2)./(sigma*sqrt(32*pi)),20); b1=b1';
        b2 = awgn(exp(-0.5*((Xb-mu)/sigma).^2)./(sigma*sqrt(32*pi)),20); b2=b2';
    end

    a1all=zeros(length(Xa),Nt);
    a2all=zeros(length(Xb),Nt);
    b1all=zeros(length(Xa),Nt);
    b2all=zeros(length(Xb),Nt);
    a1all(:,1)=a1;
    a2all(:,1)=a2;
    b1all(:,1)=b1;
    b2all(:,1)=b2;

    % Laplacian difference operator with no flux boundary conditions
    % Crank-Nicolson operators
    II1 = speye(Na,Na);
    Lapdiff1 = spdiags(ones(Na,1)*[0 1 -2 1 0], [-Na+1, -1, 0 , 1, Na-1], Na, Na);
    Lapdiff1(1,1) = -2; Lapdiff1(1,2) = 1; Lapdiff1(1,Na) = 1;
    Lapdiff1(Na,1) = 1; Lapdiff1(Na,Na-1) = 1; Lapdiff1(Na,Na) = -2;
    Hm1 = II1+(pa/2)*Lapdiff1;
    Hs1 = II1-(pa/2)*Lapdiff1;

    II2 = speye(Na,Na);
    Lapdiff2 = spdiags(ones(Na,1)*[0 1 -2 1 0], [-Na+1, -1, 0 , 1, Na-1], Na, Na);
    Lapdiff2(1,1) = -2; Lapdiff2(1,2) = 1; Lapdiff2(1,Na) = 1;
    Lapdiff2(Na,1) = 1; Lapdiff2(Na,Na-1) = 1; Lapdiff2(Na,Na) = -2;
    Hm2 = II2+(pa/2)*Lapdiff2;
    Hs2 = II2-(pa/2)*Lapdiff2;

    % Setup polarity concentrations
    %
    MAX_OUTPUT_LENGTH = 10000;
    nx1   = zeros(N,Nt);                      % state of the particle (0 inactive, 1 active)
    ny1   = zeros(N,Nt);                      % state of the particle (0 inactive, 1 active)
    Tx1   = zeros(MAX_OUTPUT_LENGTH,1);       % times of chemical reactions for X(t)
    Ty1   = zeros(MAX_OUTPUT_LENGTH,1);       % times of chemical reactions for Y(t)
    X1    = zeros(MAX_OUTPUT_LENGTH,1);       % number of X(t) molecules on the membrane
    Y1    = zeros(MAX_OUTPUT_LENGTH,1);       % number of Y(t) molecules on the membrane
    NNx1  = zeros(Nt,1);
    NNy1  = zeros(Nt,1);

    nx2   = zeros(N,Nt);                      % state of the particle (0 inactive, 1 active)
    ny2   = zeros(N,Nt);                      % state of the particle (0 inactive, 1 active)
    Tx2   = zeros(MAX_OUTPUT_LENGTH,1);       % times of chemical reactions for X(t)
    Ty2   = zeros(MAX_OUTPUT_LENGTH,1);       % times of chemical reactions for Y(t)
    X2    = zeros(MAX_OUTPUT_LENGTH,1);       % number of X(t) molecules on the membrane
    Y2    = zeros(MAX_OUTPUT_LENGTH,1);       % number of Y(t) molecules on the membrane
    NNx2  = zeros(Nt,1);
    NNy2  = zeros(Nt,1);

    % Set initial conditions for polarity molecules distribution
    %
    rxn_count_x1       = 1;
    rxn_count_y1       = 1;

    rxn_count_x2       = 1;
    rxn_count_y2       = 1;

    X1(1)              = 0.1*N;                 % # of particles on membrane
    Y1(1)              = 0.1*N;                 % # of particles on membrane
    NNx1(1)            = X1(1);
    NNy1(1)            = Y1(1);
    Tx1(1)             = 0.0;
    Ty1(1)             = 0.0;
    nx1(1:X1(1),1)      = 1;                     % activate mem-bound particles
    ny1(1:X1(1),1)      = 1;
    r1 = randperm(ceil(L/(0.0102)),X1(1)+Y1(1))*0.0102;
    posx1(1:X1(1),1)=r1(1:X1(1));
    posy1(1:Y1(1),1)=r1(X1(1)+1:end);

    X2(1)              = 0.1*N;                 % # of particles on membrane
    Y2(1)              = 0.1*N;                 % # of particles on membrane
    NNx2(1)            = X2(1);
    NNy2(1)            = Y2(1);
    Tx2(1)             = 0.0;
    Ty2(1)             = 0.0;
    nx2(1:X2(1),1)      = 1;                     % activate mem-bound particles
    ny2(1:X2(1),1)      = 1;
    r2 = randperm(ceil(L/(0.0102)),X2(1)+Y2(1))*0.0102;
    posx2(1:X2(1),1)=r2(1:X2(1));
    posy2(1:Y2(1),1)=r2(X2(1)+1:end);

    % Sample concentration at actin filament spatial scale
    %
    [s1,xC1,yC1] = resamplePolarityMolecules(posx1(1:NNx1(1),1),posy1(1:NNy1(1),1),NNx1(1),NNy1(1),L,Na);
    [s2,xC2,yC2] = resamplePolarityMolecules(posx2(1:NNx2(1),1),posy2(1:NNy2(1),1),NNx2(1),NNy2(1),L,Na);

    xC1all=zeros(Na,Nt);
    yC1all=zeros(Na,Nt);
    xC2all=zeros(Na,Nt);
    yC2all=zeros(Na,Nt);

    xC1all(:,1)=xC1;
    yC1all(:,1)=yC1;
    xC2all(:,1)=xC2;
    yC2all(:,1)=yC2;

    posx1saved=posx1;
    posy1saved=posy1;
    posx2saved=posx2;
    posy2saved=posy2;

    aic1 = a1;
    bic1 = b1;

    aic2 = a2;
    bic2 = b2;


    % Setup convergence checks for actin quantities
    %
    conv1_1 = zeros(Nt,2);
    conv2_1 = zeros(Nt,2);
    convsup_1 = zeros(Nt,2);

    conv1_2 = zeros(Nt,2);
    conv2_2 = zeros(Nt,2);
    convsup_2 = zeros(Nt,2);

    %amount cells have moved
    xshift1=zeros(1,Nt);
    yshift1=zeros(1,Nt);
    xshift2=zeros(1,Nt);
    yshift2=zeros(1,Nt);

    %positions of center of each cell
    posn1=[0,0];
    posn2=[0,-2];

    if vid==1
        vidObjScatter.FrameRate=5;
        vidObjScatter.Quality=100;
        vidObjCells.FrameRate=5;
        vidObjCells.Quality=100;
        open(vidObjScatter);
        open(vidObjCells);
    end

    %Initial plot
    if vid==1
        % Define circles
        gapsize=0.01;
        [th,rad] = meshgrid((0:3.6:360)*pi/180,0.85:0.01:1);
        [Xcol,Ycol] = pol2cart(th,rad);
        Ycol1=Ycol;
        Ycol2=Ycol;
        Ycol1(:,boundC1)=Ycol1(:,boundC1(1)*ones(1,length(boundC1)));
        Ycol2(:,boundC2)=Ycol2(:,boundC2(1)*ones(1,length(boundC2)));
        Ycol2 = Ycol2 - 2*abs(max(max(Ycol2)))-gapsize;
        ZBranch1 = [a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1]';
        ZBund1 = [b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1]';
        ZBranch2 = [a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2]';
        ZBund2 = [b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2]';
        [th,rad] = meshgrid((0:3.6:360)*pi/180,0.8);
        [Xsm,Ysm] = pol2cart(th,rad);
        Ysm1=Ysm;
        Ysm2=Ysm;
        Ysm1(:,boundC1)=Ysm1(:,boundC1(1)*ones(1,length(boundC1)));
        Ysm2(:,boundC2)=Ysm2(:,boundC2(1)*ones(1,length(boundC2)));
        [th,rad] = meshgrid((0:3.6:360)*pi/180,0.86:0.01:0.93);
        [Xmid,Ymid] = pol2cart(th,rad);


        %Define colors
        colorLength = 50;
        white = [1,1,1];
        darkyellow = [227/256,180/256,76/256];
        yellow2 = [254/256,254/256,98/256];
        pink = [211/256,95/256,183/256];
        darkpink = [141/256,45/256,113/256];
        whiteyellow2 = [linspace(white(1),yellow2(1),colorLength)',linspace(white(2),yellow2(2),colorLength)',linspace(white(3),yellow2(3),colorLength)'];
        yellow2darkyellow = [linspace(yellow2(1),darkyellow(1),colorLength)',linspace(yellow2(2),darkyellow(2),colorLength)',linspace(yellow2(3),darkyellow(3),colorLength)'];
        whitedarkyellow2 = [whiteyellow2;yellow2darkyellow];
        whitepink = [linspace(white(1),pink(1),colorLength)',linspace(white(2),pink(2),colorLength)',linspace(white(3),pink(3),colorLength)'];
        pinkdarkpink = [linspace(pink(1),darkpink(1),colorLength)',linspace(pink(2),darkpink(2),colorLength)',linspace(pink(3),darkpink(3),colorLength)'];
        whitedarkpink = [whitepink;pinkdarkpink];


        branchedColor = whitedarkpink;
        bundledColor = whitedarkyellow2;
        branchedColName = 'Pink';
        bundledColName = 'Yellow';

        % Make scatterplots
        scatplot=figure(ppp);
        clf
        subplot(1,2,1); %Cell 1
        plot(Xa,a1,'-o','color',branchedColor(end,:),'linewidth',3); hold on;
        plot(Xa,b1,'-ok','color',bundledColor(end,:),'linewidth',3);
        plot(s1,xC1,'-.','color',branchedColor(end,:),'linewidth',1);
        plot(s1,yC1,'-.k','color',bundledColor(end,:),'linewidth',1);
        set(gca,'fontname','times','fontsize',20); box on;
        lgd = legend('Branched network','Bundled network','Rac','Rho','Location','northeast');
        lgd.NumColumns = 2;
        set(gcf,'color','w');
        title('Cell 1')
        hold off;

        subplot(1,2,2); %Cell 2
        plot(Xa,a2,'-o','color',branchedColor(end,:),'linewidth',3); hold on;
        plot(Xa,b2,'-ok','color',bundledColor(end,:),'linewidth',3);
        plot(s2,xC2,'-.','color',branchedColor(end,:),'linewidth',1);
        plot(s2,yC2,'-.k','color',bundledColor(end,:),'linewidth',1);
        set(gca,'fontname','times','fontsize',20); box on;
        lgd = legend('Branched network','Bundled network','Rac','Rho','Location','northeast');
        lgd.NumColumns = 2;
        set(gcf,'color','w');
        title('Cell 2')
        hold off;

        if vid==1
            scatframe = getframe(scatplot);
            writeVideo(vidObjScatter,scatframe);
        end


        % Plot cells
        t=1;
        allmax=12;
        showtime=1;
        cellsplot=figure(ppp+1);
        clf
        range=3;
        hold on
        alphaData=ZBranch1+max(0,max(max(ZBranch2))-max(max(ZBranch1)));
        surf(Xcol+xshift1(t),Ycol+yshift1(t),ZBranch1,'AlphaData',alphaData,'FaceAlpha','interp','FaceColor','interp');
        colormap(branchedColor)
        clim([0,12])
        freezeColors;
        shading interp
        alphaData=ZBund1+max(0,max(max(ZBund2))-max(max(ZBund1)));
        surf(Xcol+xshift1(t),Ycol+yshift1(t),ZBund1,'AlphaData',alphaData,'FaceAlpha','interp','FaceColor','interp');
        colormap(bundledColor)
        clim([0,12])
        freezeColors;
        shading interp
        view(2)
        if max(xC1)>=(range+2)
            racxvals1=(range-1)*xC1/max(xC1)+1;
            racyvals1=(range-1)*xC1/max(xC1)+1;
        else
            racxvals1=(range-2)*xC1/max(xC1)+1;
            racyvals1=(range-1)*xC1/max(xC1)+1;
        end
        racxvals1=(racxvals1)'.*cos(2*pi*Xa/L);
        racyvals1=(racyvals1)'.*sin(2*pi*Xa/L);
        plot3(racxvals1+xshift1(t),racyvals1+yshift1(t),(allmax+1)*ones(1,length(racxvals1)),'color',...
            [branchedColor(end,:),1],'LineWidth',3)
        plot3([racxvals1(end)+xshift1(t),racxvals1(1)+xshift1(t)],[racyvals1(end)+yshift1(t),racyvals1(1)+yshift1(t)],...
            [allmax+1,allmax+1],'color',[branchedColor(end,:),1],'LineWidth',3)
        if max(yC1)>=(range+2)
            rhoxvals1=(range-1)*yC1/max(yC1)+1;
            rhoyvals1=(range-1)*yC1/max(yC1)+1;
        else
            rhoxvals1=(range-2)*yC1/max(yC1)+1;
            rhoyvals1=(range-2)*yC1/max(yC1)+1;
        end
        rhoxvals1=(rhoxvals1)'.*cos(2*pi*Xa/L);
        rhoyvals1=(rhoyvals1)'.*sin(2*pi*Xa/L);
        plot3(rhoxvals1+xshift1(t),rhoyvals1+yshift1(t),(allmax+1)*ones(1,length(rhoxvals1)),'color',...
            [bundledColor(end,:),1],'LineWidth',3)
        plot3([rhoxvals1(end)+xshift1(t),rhoxvals1(1)+xshift1(t)],[rhoyvals1(end)+yshift1(t),rhoyvals1(1)+yshift1(t)],...
            [allmax+1,allmax+1],'color',[bundledColor(end,:),1],'LineWidth',3)
        plot3(cos(2*pi*Xa/L)+xshift1(t),sin(2*pi*Xa/L)+yshift1(t),(allmax+2)*ones(1,Na),'color','black','LineWidth',1)
        hold off

        %cell 2
        hold on
        alphaData=ZBranch2+max(0,max(max(ZBranch1))-max(max(ZBranch2)));
        surf(Xcol+xshift2(t),Ycol+yshift2(t)-2-(range-1),ZBranch2,'AlphaData',alphaData,'FaceAlpha','interp','FaceColor','interp');
        colormap(branchedColor)
        clim([0,12])
        cb=colorbar('Location','eastoutside');
        freezeColors;
        freezeColors(cb);
        cbpos=cb.Position;
        % set(cb,'Position',[cbpos(1)+2*cbpos(3),cbpos(2),cbpos(3),cbpos(4)/2])
        set(cb,'Position',[0.9062    0.1097    0.0235    0.4077])
        set(cb,'TickLabels',{});
        cbpos=cb.Position;
        shading interp
        alphaData=ZBund2+max(0,max(max(ZBund1))-max(max(ZBund2)));
        surf(Xcol+xshift2(t),Ycol+yshift2(t)-2-(range-1),ZBund2,'AlphaData',alphaData,'FaceAlpha','interp','FaceColor','interp');
        colormap(bundledColor)
        clim([0,12])
        freezeColors;
        jcb=jicolorbar;
        freezeColors(jcb);
        jcbpos=jcb.Position;
        set(jcb,'Position',[cbpos(1)+cbpos(3),cbpos(2),cbpos(3),cbpos(4)])
        shading interp
        view(2)
        if max(xC2)>=(range+2)
            racxvals2=(range-1)*xC2/max(xC2)+1;
            racyvals2=(range-1)*xC2/max(xC2)+1;
        else
            racxvals2=(range-2)*xC2/max(xC2)+1;
            racyvals2=(range-2)*xC2/max(xC2)+1;
        end
        racxvals2=(racxvals2)'.*cos(2*pi*Xa/L);
        racyvals2=(racyvals2)'.*sin(2*pi*Xa/L);
        plot3(racxvals2+xshift2(t),racyvals2+yshift2(t)-2-(range-1),(allmax+1)*ones(1,length(racxvals2)),'color',...
            [branchedColor(end,:),0.5],'LineWidth',3,'LineStyle','-.')
        plot3([racxvals2(end)+xshift2(t),racxvals2(1)+xshift2(t)],[racyvals2(end)+yshift2(t),racyvals2(1)+yshift2(t)]-2-(range-1),...
            [allmax+1,allmax+1],'color',[branchedColor(end,:),0.5],'LineWidth',3,'LineStyle','-.')
        if max(yC2)>=(range+2)
            rhoxvals2=(range-1)*yC2/max(yC2)+1;
            rhoyvals2=(range-1)*yC2/max(yC2)+1;
        else
            rhoxvals2=(range-2)*yC2/max(yC2)+1;
            rhoyvals2=(range-2)*yC2/max(yC2)+1;
        end
        rhoxvals2=(rhoxvals2)'.*cos(2*pi*Xa/L);
        rhoyvals2=(rhoyvals2)'.*sin(2*pi*Xa/L);
        plot3(rhoxvals2+xshift2(t),rhoyvals2+yshift2(t)-2-(range-1),(allmax+1)*ones(1,length(rhoxvals2)),'color',...
            [bundledColor(end,:),0.5],'LineWidth',3,'LineStyle','-.')
        plot3([rhoxvals2(end)+xshift2(t),rhoxvals2(1)+xshift2(t)],[rhoyvals2(end)+yshift2(t),rhoyvals2(1)+yshift2(t)]-2-(range-1),...
            [allmax+1,allmax+1],'color',[bundledColor(end,:),0.5],'LineWidth',3,'LineStyle','-.')
        plot(cos(2*pi*Xa/L)+xshift2(t),sin(2*pi*Xa/L)+yshift2(t)-2-(range-1),'color','black','LineWidth',1)

        xlim([-3,3])
        ylim([-8,4])
        set(gca,'plotBoxAspectRatio',[6 12 1]);
        hold off

        cbpos=cb.Position;
        if showtime==1
            timebox=annotation('textbox', [0.75, cbpos(2), 0.1, 0.05], 'String', "t = " + 0,'FitBoxToText','on','EdgeColor','none','FontSize',20);
        end

        if signal==1
            [th,rad] = meshgrid((0:3.6:360)*pi/180,1.1);
            [Xsig,Ysig] = pol2cart(th,rad);
            if t<=sigswitch_time
                hold on;
                scatter3(Xsig(sigBound2)+xshift2(t),Ysig(sigBound2)+yshift2(t)-2-(range-1),(allmax+3)*ones(1,length(sigBound2)),50,'black','.')
                hold off;
            else
                hold on;
                scatter3(Xsig(sigBound1)+xshift1(t),Ysig(sigBound1)+yshift1(t),(allmax+3)*ones(1,length(sigBound1)),50,'black','.')
                hold off;
            end
        end

        grid off
        set(gca,'XTick',[],'YTick',[])
        set(gca,'XColor','w')
        set(gca,'YColor','w')
        set(gcf,'color','w');
        set(gcf,'Position',[209   561   682   474])
        ohf = findobj(gcf);
        figaxes = findobj(ohf(1), 'Type', 'axes');
        set(figaxes(1),'Fontsize',15)
        set(figaxes(2),'Fontsize',14)
        camroll(90)


        if vid==1
            cellsframe = getframe(cellsplot);
            writeVideo(vidObjCells,cellsframe);
        end
    end

    %% Run simulation
    %
    tic
    quit_cond = 0;
    cond = 0;
    for t=1:(Nt-1)

        %% Run biochemistry
        [Konx1,Kony1,Kfbx1,Kfby1,Koffx1,Koffy1] = spatialrates(ron,rfb,roff,a1,b1,s1,beta,cond,boundC1); % set rates
        [Konx2,Kony2,Kfbx2,Kfby2,Koffx2,Koffy2] = spatialrates(ron,rfb,roff,a2,b2,s2,beta,cond,boundC2);


        
        
        % Add external signal for cell 2
        % this works
        if signal==1
            steepness = 20;
            if t<=sigswitch_time
                Konx2 = (ron*(tanh(steepness*(s2-s2(sigBound2(1)))) ...
                    - tanh(steepness*(s2-s2(sigBound2(end)))) + 0.2)/2.2)';
                Kony2 = (ron*(2 - tanh(steepness*(s2-s2(sigBound2(1)))) ...
                    + tanh(steepness*(s2-s2(sigBound2(end)))) + 0.2)/2.2)';
                Kfbx2 = (rfb*(tanh(steepness*(s2-s2(sigBound2(1)))) ...
                    - tanh(steepness*(s2-s2(sigBound2(end)))) + 0.2)/2.2)';
                Kfby2 = (rfb*(2 - tanh(steepness*(s2-s2(sigBound2(1)))) ...
                    + tanh(steepness*(s2-s2(sigBound2(end)))) + 0.2)/2.2)';
                Koffx2 = (roff*(2 - tanh(steepness*(s2-s2(sigBound2(1)))) ...
                    + tanh(steepness*(s2-s2(sigBound2(end)))) + 0.2)/2.2)';
                Koffy2 = (roff*(tanh(steepness*(s2-s2(sigBound2(1)))) ...
                    - tanh(steepness*(s2-s2(sigBound2(end)))) + 0.2)/2.2)';
            else
                Konx1 = (ron*(tanh(steepness*(s1-s1(sigBound1(1)))) ...
                    - tanh(steepness*(s1-s1(101))) + tanh(steepness*(s1-s1(1))) ...
                    - tanh(steepness*(s1-s1(sigBound1(end)))) + 0.2)/2.2)';
                Kony1 = (ron*(2 - tanh(steepness*(s1-s1(sigBound1(1)))) ...
                    + tanh(steepness*(s1-s1(101))) - tanh(steepness*(s1-s1(1))) ...
                    + tanh(steepness*(s1-s1(sigBound1(end)))) + 0.2)/2.2)';
                Kfbx1 = (rfb*(tanh(steepness*(s1-s1(sigBound1(1))))  ...
                    - tanh(steepness*(s1-s1(101))) + tanh(steepness*(s1-s1(1))) ...
                    -  tanh(steepness*(s1-s1(sigBound1(end)))) + 0.2)/2.2)';
                Kfby1 = (rfb*(2 - tanh(steepness*(s1-s1(sigBound1(1)))) ...
                    + tanh(steepness*(s1-s1(101))) - tanh(steepness*(s1-s1(1))) ...
                    + tanh(steepness*(s1-s1(sigBound1(end)))) + 0.2)/2.2)';
                Koffx1 = (roff*(2 - tanh(steepness*(s1-s1(sigBound1(1)))) ...
                    + tanh(steepness*(s1-s1(101))) - tanh(steepness*(s1-s1(1))) ...
                    + tanh(steepness*(s1-s1(sigBound1(end)))) + 0.2)/2.2)';
                Koffy1 = (roff*(tanh(steepness*(s1-s1(sigBound1(1))))  ...
                    - tanh(steepness*(s1-s1(101))) + tanh(steepness*(s1-s1(1))) ...
                    -  tanh(steepness*(s1-s1(sigBound1(end)))) + 0.2)/2.2)';
            end
        end

%         if t<=sigswitch_time
%             Koffy1(boundC1)=Koffy1(boundC1)*1000;
%             Kony2(boundC2)=Kony2(boundC2)*1000;
%         else
%             Kony1(boundC1)=Kony1(boundC1)*1000;
%             Koffy2(boundC2)=Koffy2(boundC2)*1000;
%         end


%        if c1_ind==1
%            Koffx1(boundC1)=Koffx1(boundC1)*coeff_vals(c1coeff_ind);
%        elseif c1_ind==2
%            Koffy1(boundC1)=Koffy1(boundC1)*coeff_vals(c1coeff_ind);
%        elseif c1_ind==3
%            Konx1(boundC1)=Konx1(boundC1)*coeff_vals(c1coeff_ind);
%        elseif c1_ind==4
%            Kony1(boundC1)=Kony1(boundC1)*coeff_vals(c1coeff_ind);
%        end
%
%        if c2_ind==1
%            Koffx2(boundC2)=Koffx2(boundC2)*coeff_vals(c2coeff_ind);
%        elseif c2_ind==2
%            Koffy2(boundC2)=Koffy2(boundC2)*coeff_vals(c2coeff_ind);
%        elseif c2_ind==3
%            Konx2(boundC2)=Konx2(boundC2)*coeff_vals(c2coeff_ind);
%        elseif c2_ind==4
%            Kony2(boundC2)=Kony2(boundC2)*coeff_vals(c2coeff_ind);
%        end


%  epsilon1 = 0.1;
%  flipc2=flip(boundC2);
%  scaledC1 = (L*boundC1/Na);
%  scaledC2 = L*flipc2/Na;
%  for i=1:length(boundC1)
%      sumx1 = sum(abs(posx1(:,t)-scaledC1(i))<=epsilon1);
%      sumx2 = sum(abs(posx2(:,t)-scaledC2(i))<=epsilon1);
%      sumy1 = sum(abs(posy1(:,t)-scaledC1(i))<=epsilon1);
%      sumy2 = sum(abs(posy2(:,t)-scaledC2(i))<=epsilon1);
%  
%      if c1_ind==1
%          if (c2_ind==1 || c2_ind==3) && sumx2>0
%              Koffx1(boundC1(i))=Koffx1(boundC1(i))*coeff_vals(c1coeff_ind)*sumx2;
%          elseif (c2_ind==2 || c2_ind==4) && sumy2>0
%              Koffx1(boundC1(i))=Koffx1(boundC1(i))*coeff_vals(c1coeff_ind)*sumy2;
%          end
%      elseif c1_ind==2
%          if (c2_ind==1 || c2_ind==3) && sumx2>0
%              Koffy1(boundC1(i))=Koffy1(boundC1(i))*coeff_vals(c1coeff_ind)*sumx2;
%          elseif (c2_ind==2 || c2_ind==4) && sumy2>0
%              Koffy1(boundC1(i))=Koffy1(boundC1(i))*coeff_vals(c1coeff_ind)*sumy2;
%          end
%      elseif c1_ind==3
%          if (c2_ind==1 || c2_ind==3) && sumx2>0
%              Konx1(boundC1(i))=Konx1(boundC1(i))*coeff_vals(c1coeff_ind)*sumx2;
%          elseif (c2_ind==2 || c2_ind==4) && sumy2>0
%              Konx1(boundC1(i))=Konx1(boundC1(i))*coeff_vals(c1coeff_ind)*sumy2;
%          end
%      elseif c1_ind==4
%          if (c2_ind==1 || c2_ind==3) && sumx2>0
%              Kony1(boundC1(i))=Kony1(boundC1(i))*coeff_vals(c1coeff_ind)*sumx2;
%          elseif (c2_ind==2 || c2_ind==4) && sumy2>0
%              Kony1(boundC1(i))=Kony1(boundC1(i))*coeff_vals(c1coeff_ind)*sumy2;
%          end
%      end
%  
%      if c2_ind==1
%          if (c1_ind==1 || c1_ind==3) && sumx1>0
%              Koffx2(flipc2(i))=Koffx2(flipc2(i))*coeff_vals(c2coeff_ind)*sumx1;
%          elseif (c1_ind==2 || c1_ind==4) && sumy1>0
%              Koffx2(flipc2(i))=Koffx2(flipc2(i))*coeff_vals(c2coeff_ind)*sumy1;
%          end
%      elseif c2_ind==2
%          if (c1_ind==1 || c1_ind==3) && sumx1>0
%              Koffy2(flipc2(i))=Koffy2(flipc2(i))*coeff_vals(c2coeff_ind)*sumx1;
%          elseif (c1_ind==2 || c1_ind==4) && sumy1>0
%              Koffy2(flipc2(i))=Koffy2(flipc2(i))*coeff_vals(c2coeff_ind)*sumy1;
%          end
%      elseif c2_ind==3
%          if (c1_ind==1 || c1_ind==3) && sumx1>0
%              Konx2(flipc2(i))=Konx2(flipc2(i))*coeff_vals(c2coeff_ind)*sumx1;
%          elseif (c1_ind==2 || c1_ind==4) && sumy1>0
%              Konx2(flipc2(i))=Konx2(flipc2(i))*coeff_vals(c2coeff_ind)*sumy1;
%          end
%      elseif c2_ind==4
%          if (c1_ind==1 || c1_ind==3) && sumx1>0
%              Kony2(flipc2(i))=Kony2(flipc2(i))*coeff_vals(c2coeff_ind)*sumx1;
%          elseif (c1_ind==2 || c1_ind==4) && sumy1>0
%              Kony2(flipc2(i))=Kony2(flipc2(i))*coeff_vals(c2coeff_ind)*sumy1;
%          end
%      end
%  end

   




switch pathway
    case 'uncoupled'

    case 'racupc1-rhoupc2'
        Konx1(boundC1)=Konx1(boundC1)*1000;
        Kony2(boundC2)=Kony2(boundC2)*1000;
    case 'rhoupc1-rhodownc2'
        Kony1(boundC1)=Kony1(boundC1)*1000;
        Koffy2(boundC2)=Koffy2(boundC2)*1000;
    case 'racupc1-rhoupc2-concdependent'
        epsilon1 = 0.1;
        flipc2=flip(boundC2);
        scaledC1 = (L*boundC1/Na);
        scaledC2 = L*flipc2/Na;
        for i=1:length(boundC1)
            sumx1 = sum(abs(posx1(:,t)-scaledC1(i))<=epsilon1);
            sumx2 = sum(abs(posx2(:,t)-scaledC2(i))<=epsilon1);
            sumy1 = sum(abs(posy1(:,t)-scaledC1(i))<=epsilon1);
            sumy2 = sum(abs(posy2(:,t)-scaledC2(i))<=epsilon1);
            if sumx1>0
                Kony2(flipc2(i)) = Kony2(flipc2(i))*sumx1*1000;
            end
            if sumy2>0
                Konx1(flipc2(i)) = Konx1(flipc2(i))*sumy2*1000;
            end
        end
    case 'rhoupc1-rhodownc2-forcedependent'
        Kony1(boundC1) = Kony1(boundC1).*flip(a2(boundC2))*1000;
        Koffy2(boundC2) = Koffy2(boundC2).*flip(b1(boundC1))*1000;
    case 'bundledupc1-branchedupc2'
        kbb_ind=2;
        kcc_ind=2;
    case 'branched-bundled-crosspromotion'
        kb_ind=2;
        kc_ind=2;
    case 'ractorho-antagonism'
        epsilon=0.5;
        numRhoToRemove=10;
end


        %Cell 1
        resetc1x=0;
        if((t-1)*dt<Tx1(rxn_count_x1)) % have we reached the time of next reaction?
            NNx1(t+1) = X1(rxn_count_x1-1); % if not, nothing happens
        else
            rng('shuffle');
            nnx1 = X1(rxn_count_x1); % number of rac on membrane at time of reaction
            taux1 = zeros(nnx1,1);
            dnx1 = zeros(nnx1,1);
            rx1 = rand(nnx1,1);

%             if(nnx1==0) % no rac on membrane
%                 sprintf('here 1rac')
%                 counter_ppp = ppp;
%                 quit_cond = 1;
%                 break
%             end

            if(nnx1==0) % no rac on membrane
                sprintf('here 1rac')

                X1(rxn_count_x1+1) = 0.1*N; % put 20 rac on membrane
                NNx1(t+1) = X1(rxn_count_x1+1);
                Tx1(rxn_count_x1+1) = 0.0;
                nx1(1:X1(rxn_count_x1+1),1) = 1;
                r1 = randperm(ceil(L/(0.0102)),X1(rxn_count_x1+1))*0.0102;
                posx1(1:X1(rxn_count_x1+1),1)=r1(1:X1(rxn_count_x1+1));
                % rxn_count_x1 = rxn_count_x1 + 1;
                resetc1x=1;


            else

                for j=1:nnx1          % all agents
                    konx1 = interp1(s1,Konx1,posx1(j,t));
                    koffx1 = interp1(s1,Koffx1,posx1(j,t));
                    kfbx1 = interp1(s1,Kfbx1,posx1(j,t));
                    % Sample earliest time-to-fire (tau)
                    a0_x1 = koffx1 + (konx1+kfbx1*nnx1/N)*(N/nnx1-1);
                    taux1(j) = -log(rx1(j))/a0_x1;
                    rr_x1 = rand(1,1);
                    dnx1(j) = (rr_x1<((konx1+kfbx1*nnx1/N)*(N/nnx1-1)/a0_x1))*1.0 ...
                        + (rr_x1>=((konx1+kfbx1*nnx1/N)*(N/nnx1-1)/a0_x1))*(-1.0); %does particle bind (+1) or unbind (-1)
                end

                [mintaux1,minidx1] = min(taux1(1:j));       % find first chemical rxn
                Tx1(rxn_count_x1+1) = Tx1(rxn_count_x1) + mintaux1;
                X1(rxn_count_x1+1) = nnx1 + dnx1(minidx1);
                rxn_count_x1 = rxn_count_x1 + 1;
                NNx1(t+1) = X1(rxn_count_x1-1);
            end
        end

        %Cell 2
        resetc2x=0;
        if((t-1)*dt<Tx2(rxn_count_x2))
            NNx2(t+1) = X2(rxn_count_x2-1);
        else
            rng('shuffle');
            nnx2 = X2(rxn_count_x2);
            taux2 = zeros(nnx2,1);
            dnx2 = zeros(nnx2,1);
            rx2 = rand(nnx2,1);

%             if(nnx2==0)
%                 sprintf('here 2rac')
%                 counter_ppp = ppp;
%                 quit_cond = 1;
%                 break
%             end

            if(nnx2==0) % no rac on membrane
                sprintf('here 2rac')

                X2(rxn_count_x2+1) = 0.1*N; % put 20 rac on membrane
                NNx2(t+1) = X2(rxn_count_x2+1);
                Tx2(rxn_count_x2+1) = 0.0;
                nx2(1:X2(rxn_count_x2+1),1) = 1;
                r2 = randperm(ceil(L/(0.0102)),X2(rxn_count_x2+1))*0.0102;
                posx2(1:X2(rxn_count_x2+1),1)=r2(1:X2(rxn_count_x2+1));
                % rxn_count_x2 = rxn_count_x2 + 1;
                resetc2x=1;

            else

                for j=1:nnx2          % all agents
                    konx2 = interp1(s2,Konx2,posx2(j,t));
                    koffx2 = interp1(s2,Koffx2,posx2(j,t));
                    kfbx2 = interp1(s2,Kfbx2,posx2(j,t));
                    % Sample earliest time-to-fire (tau)
                    a0_x2 = koffx2 + (konx2+kfbx2*nnx2/N)*(N/nnx2-1);
                    taux2(j) = -log(rx2(j))/a0_x2;
                    rrx2 = rand(1,1);
                    dnx2(j) = (rrx2<((konx2+kfbx1*nnx2/N)*(N/nnx2-1)/a0_x2))*1.0 ...
                        + (rrx2>=((konx2+kfbx2*nnx2/N)*(N/nnx2-1)/a0_x2))*(-1.0);
                end

                [mintaux2,minidx2] = min(taux2(1:j));       % find first chemical rxn
                Tx2(rxn_count_x2+1) = Tx2(rxn_count_x2) + mintaux2;
                X2(rxn_count_x2+1) = nnx2 + dnx2(minidx2);
                rxn_count_x2 = rxn_count_x2 + 1;
                NNx2(t+1) = X2(rxn_count_x2-1);
            end
        end

        %Cell 1
        resetc1y=0;
        if((t-1)*dt<Ty1(rxn_count_y1))
            NNy1(t+1) = Y1(rxn_count_y1-1);
        else
            rng('shuffle');
            nny1 = Y1(rxn_count_y1);
            tauy1 = zeros(nny1,1);
            dny1 = zeros(nny1,1);
            ry1 = rand(nny1,1);

%             if(nny1==0)
%                 sprintf('here 1rho')
%                 counter_ppp = ppp;
%                 quit_cond = 1;
%                 break
%             end

            if(nny1==0) % no rac on membrane
                sprintf('here 1rho')

                Y1(rxn_count_y1+1) = 0.1*N; % put 20 rac on membrane
                NNy1(t+1) = Y1(rxn_count_y1+1);
                Ty1(rxn_count_y1+1) = 0.0;
                ny1(1:Y1(rxn_count_y1+1),1) = 1;
                r1 = randperm(ceil(L/(0.0102)),Y1(rxn_count_y1+1))*0.0102;
                posy1(1:Y1(rxn_count_y1+1),1)=r1(1:Y1(rxn_count_y1+1));
                % rxn_count_y1 = rxn_count_y1 + 1;
                resetc1y=1;

            else

                for j=1:nny1          % all agents
                    kony1 = interp1(s1,Kony1,posy1(j,t));
                    koffy1 = interp1(s1,Koffy1,posy1(j,t));
                    kfby1 = interp1(s1,Kfby1,posy1(j,t));
                    % Sample earliest time-to-fire (tau)
                    a0_y1 = koffy1 + (kony1+kfby1*nny1/N)*(N/nny1-1);
                    tauy1(j) = -log(ry1(j))/a0_y1;
                    rry1 = rand(1,1);
                    dny1(j) = (rry1<((kony1+kfby1*nny1/N)*(N/nny1-1)/a0_y1))*1.0 ...
                        + (rry1>=((kony1+kfby1*nny1/N)*(N/nny1-1)/a0_y1))*(-1.0);
                end

                [mintauy1,minidy1] = min(tauy1(1:j));       % find first chemical rxn
                Ty1(rxn_count_y1+1) = Ty1(rxn_count_y1) + mintauy1;
                Y1(rxn_count_y1+1) = nny1 + dny1(minidy1);
                rxn_count_y1 = rxn_count_y1 + 1;
                NNy1(t+1) = Y1(rxn_count_y1-1);
            end
        end

        %Cell 2
        resetc2y=0;
        if((t-1)*dt<Ty2(rxn_count_y2))
            NNy2(t+1) = Y2(rxn_count_y2-1);
        else
            rng('shuffle');
            nny2 = Y2(rxn_count_y2);
            tauy2 = zeros(nny2,1);
            dny2 = zeros(nny2,1);
            ry2 = rand(nny2,1);

%             if(nny2==0)
%                 sprintf('here 2rho')
%                 counter_ppp = ppp;
%                 quit_cond = 1;
%                 break
%             end

            if(nny2==0) % no rac on membrane
                sprintf('here 2rho')

                Y2(rxn_count_y2+1) = 0.1*N; % put 20 rac on membrane
                NNy2(t+1) = Y2(rxn_count_y2+1);
                Ty2(rxn_count_y2+1) = 0.0;
                ny2(1:Y2(rxn_count_y2+1),1) = 1;
                r2 = randperm(ceil(L/(0.0102)),Y2(rxn_count_y2+1))*0.0102;
                posy2(1:Y2(rxn_count_y2+1),1)=r2(1:Y2(rxn_count_y2+1));
                % rxn_count_y2 = rxn_count_y2 + 1;
                resetc2y=1;

            else

                for j=1:nny2          % all agents
                    kony2 = interp1(s2,Kony2,posy2(j,t));
                    koffy2 = interp1(s2,Koffy2,posy2(j,t));
                    kfby2 = interp1(s2,Kfby2,posy2(j,t));
                    % Sample earliest time-to-fire (tau)
                    a0_y2 = koffy2 + (kony2+kfby2*nny2/N)*(N/nny2-1);
                    tauy2(j) = -log(ry2(j))/a0_y2;
                    rry2 = rand(1,1);
                    dny2(j) = (rry2<((kony2+kfby2*nny2/N)*(N/nny2-1)/a0_y2))*1.0 ...
                        + (rry2>=((kony2+kfby2*nny2/N)*(N/nny2-1)/a0_y2))*(-1.0);
                end

                [mintauy2,minidy2] = min(tauy2(1:j));       % find first chemical rxn
                Ty2(rxn_count_y2+1) = Ty2(rxn_count_y2) + mintauy2;
                Y2(rxn_count_y2+1) = nny2 + dny2(minidy2);
                rxn_count_y2 = rxn_count_y2 + 1;
                NNy2(t+1) = Y2(rxn_count_y2-1);
            end
        end

        if (quit_cond==1)
            sprintf("It's happening at kk = %d, ppp = %d\n",kk,ppp)
            ppp = counter_ppp-1;
            break
        end

        %% Run diffusion of membrane-bound polarity proteins
        p  = 0.5;                  % probability of hoping left or right

        % Fetch the number of particles at this time
        Kx1 = NNx1(t+1);
        Ky1 = NNy1(t+1);

        Kx2 = NNx2(t+1);
        Ky2 = NNy2(t+1);

        % Between reactions, perform Brownian motion with periodic BC
        r1_1 = rand(Kx1,1);    % coin flip
        nx1(1:Kx1,t+1) = 1;
        posx1(1:Kx1,t+1) = posx1(1:Kx1,t) + dx*((r1_1<p)*1.0 + (r1_1>(1-p))*(-1.0));

        r2_1 = rand(Kx2,1);    % coin flip
        nx2(1:Kx2,t+1) = 1;
        posx2(1:Kx2,t+1) = posx2(1:Kx2,t) + dx*((r2_1<p)*1.0 + (r2_1>(1-p))*(-1.0));

        r1_2 = rand(Ky1,1);    % coin flip
        ny1(1:Ky1,t+1) = 1;
        posy1(1:Ky1,t+1) = posy1(1:Ky1,t) + dx*((r1_2<p)*1.0 + (r1_2>(1-p))*(-1.0));

        r2_2 = rand(Ky2,1);    % coin flip
        ny2(1:Ky2,t+1) = 1;
        posy2(1:Ky2,t+1) = posy2(1:Ky2,t) + dx*((r2_2<p)*1.0 + (r2_2>(1-p))*(-1.0));

        % Check for collision(s) and resolve any collisions
        % Resolution strategy: No one advances
        %
        % Cell 1
        firstcoll1 = sum(ismembertol(posx1(1:Kx1,t+1),posy1(1:Ky1,t+1),0.005,'DataScale',1));
        if firstcoll1~=0
            % Get indices of collisions
            aa1 = ismembertol(posx1(1:Kx1,t+1),posy1(1:Ky1,t+1),0.005,'DataScale',1);
            list_idx1 = find(aa1~=0);
            bb1 = ismembertol(posy1(1:Ky1,t+1),posx1(1:Kx1,t+1),0.005,'DataScale',1);
            list_idy1 = find(bb1~=0);

            posx1(list_idx1,t+1) = posx1(list_idx1,t);
            posy1(list_idy1,t+1) = posy1(list_idy1,t);
        end

        % Cell 2
        firstcoll2 = sum(ismembertol(posx2(1:Kx2,t+1),posy2(1:Ky2,t+1),0.005,'DataScale',1));
        if firstcoll2~=0
            % Get indices of collisions
            aa2 = ismembertol(posx2(1:Kx2,t+1),posy2(1:Ky2,t+1),0.005,'DataScale',1);
            list_idx2 = find(aa2~=0);
            bb2 = ismembertol(posy2(1:Ky2,t+1),posx2(1:Kx2,t+1),0.005,'DataScale',1);
            list_idy2 = find(bb2~=0);

            posx2(list_idx2,t+1) = posx2(list_idx2,t);
            posy2(list_idy2,t+1) = posy2(list_idy2,t);
        end

        % Enforce periodic boundary conditions
        posx1(1:Kx1,t+1) = posx1(1:Kx1,t+1) + (-L).*(posx1(1:Kx1,t+1)>L) ...
            + (L).*(posx1(1:Kx1,t+1)<0.0);
        posy1(1:Ky1,t+1) = posy1(1:Ky1,t+1) + (-L).*(posy1(1:Ky1,t+1)>L) ...
            + (L).*(posy1(1:Ky1,t+1)<0.0);

        posx2(1:Kx2,t+1) = posx2(1:Kx2,t+1) + (-L).*(posx2(1:Kx2,t+1)>L) ...
            + (L).*(posx2(1:Kx2,t+1)<0.0);
        posy2(1:Ky2,t+1) = posy2(1:Ky2,t+1) + (-L).*(posy2(1:Ky2,t+1)>L) ...
            + (L).*(posy2(1:Ky2,t+1)<0.0);

        % Enforce no-flux boundary conditions
        % posx(1:K1,t+1) = posx(1:K1,t+1) + (posx(1:K1,t)-posx(1:K1,t+1)).*(posx(1:K1,t+1)>L) ...
        %     + (posx(1:K1,t)-posx(1:K1,t+1)).*(posx(1:K1,t+1)<0.0);
        % posy(1:K2,t+1) = posy(1:K2,t+1) + (posy(1:K2,t)-posy(1:K2,t+1)).*(posy(1:K2,t+1)>L) ...
        %     + (posy(1:K2,t)-posy(1:K2,t+1)).*(posy(1:K2,t+1)<0.0);

        %% Determine if a biochemical rxn has occured - update positions

        % Find spontaneous association location cell 1
        if resetc1x==0
            % if Kx1>=1
            ss1 = sort(posx1(1:Kx1,t));
            [ijk1] = find(ss1==posx1(minidx1,t),1);
            prevind1 = (ijk1-1)*(ijk1>1) + (Kx1)*(ijk1==1);
            nextind1 = (ijk1+1)*(ijk1<Kx1) + 1*(ijk1==Kx1);
            x2_1 = posx1(minidx1,t)+(ss1(prevind1)-posx1(minidx1,t))/2;
            x1_1 = posx1(minidx1,t)+(ss1(nextind1)-posx1(minidx1,t))/2;
            locx1 = (x2_1-x1_1).*rand(1,1) + x1_1; % random location halfway between the closest left/right particles

            ponx1 = ron/(ron+rfb*(N-Kx1));
        end
        % else
        %     locx1 = rand(1,1)*L;
        % end
        if resetc1y==0
            % if Ky1>=1
            ss1 = sort(posy1(1:Ky1,t));
            [ijk1] = find(ss1==posy1(minidy1,t),1);
            prevind1 = (ijk1-1)*(ijk1>1) + (Ky1)*(ijk1==1);
            nextind1 = (ijk1+1)*(ijk1<Ky1) + 1*(ijk1==Ky1);
            y2_1 = posy1(minidy1,t)+(ss1(prevind1)-posy1(minidy1,t))/2;
            y1_1 = posy1(minidy1,t)+(ss1(nextind1)-posy1(minidy1,t))/2;
            locy1 = (y2_1-y1_1).*rand(1,1) + y1_1; % random location halfway between the closest left/right particles
            % else
            %     locy1=rand(1,1)*L;
            % end

            pony1 = ron/(ron+rfb*(N-Ky1));
        end

        % Find spontaneous association location cell 2
        if resetc2x==0
            % if Kx2>=1
            ss2 = sort(posx2(1:Kx2,t));
            [ijk2] = find(ss2==posx2(minidx2,t),1);
            prevind2 = (ijk2-1)*(ijk2>1) + (Kx2)*(ijk2==1);
            nextind2 = (ijk2+1)*(ijk2<Kx2) + 1*(ijk2==Kx2);
            x2_2 = posx2(minidx2,t)+(ss2(prevind2)-posx2(minidx2,t))/2;
            x1_2 = posx2(minidx2,t)+(ss2(nextind2)-posx2(minidx2,t))/2;
            locx2 = (x2_2-x1_2).*rand(1,1) + x1_2; % random location halfway between the closest left/right particles

            ponx2 = ron/(ron+rfb*(N-Kx2));
        end
        % else
        %     locx2=rand(1,1)*L;
        % end
        % if Ky2>=1
        if resetc2y==0
            ss2 = sort(posy2(1:Ky2,t));
            [ijk2] = find(ss2==posy2(minidy2,t),1);
            prevind2 = (ijk2-1)*(ijk2>1) + (Ky2)*(ijk2==1);
            nextind2 = (ijk2+1)*(ijk2<Ky2) + 1*(ijk2==Ky2);
            y2_2 = posy2(minidy2,t)+(ss2(prevind2)-posy2(minidy2,t))/2;
            y1_2 = posy2(minidy2,t)+(ss2(nextind2)-posy2(minidy2,t))/2;
            locy2 = (y2_2-y1_2).*rand(1,1) + y1_2; % random location halfway between the closest left/right particles
            % else
            %     locy2=rand(1,1)*L;
            % end


            pony2 = ron/(ron+rfb*(N-Ky2));
        end

        %Cell 1 rac
        if resetc1x==0
            if(NNx1(t+1) < NNx1(t))                % diassociation event (particle off)
                oldcolx1 = posx1(minidx1,1:end); % Find the particle to be removed
                othercolsx1 = posx1([1:minidx1-1,minidx1+1:Kx1],1:end); % Gather other "on" particles
                otherothercolsx1 = posx1(Kx1+1:end,1:end); % Gather "off" particles
                newposx1 = [othercolsx1;oldcolx1;otherothercolsx1]; % Put removed particle at the end of "on" particles
                posx1 = newposx1;
                nx1(Kx1,t+1) = 0; % Set the removed particle to inactive
            elseif(NNx1(t+1) > NNx1(t))             % association event (on or recruitment)
                rrx1 = rand(1,1);
                posx1(Kx1,t+1) = posx1(Kx1,t)+(rrx1<ponx1)*locx1; % on event
                posx1(Kx1,t+1) = posx1(Kx1,t)+(rrx1>=ponx1)*posx1(minidx1,t);   % recruitment event
                nx1(Kx1,t+1) = 1;
                % Look for nearby rho (posy1), take them off
                % posx1(K1_1,t+1)=location of rac binding
                if numRhoToRemove>0
                    boundC1Scaled=(L*boundC1/Na);
                    locRemovey1 = find(abs(posy1(:,t+1)-posx1(Kx1,t+1))<epsilon,numRhoToRemove);
                    numFoundy1 = length(locRemovey1);
                    if ~isempty(locRemovey1) && boundC1Scaled(1)<=posx1(Kx1,t+1) ...&& boundC1Scaled(end)>=posx1(Kx1,t+1)
                            % posy1(locRemovey1,t+1)=0;
                        oldcoly1 = posy1(locRemovey1,1:end); % Find the particle(s) to be removed
                        othercolsy1 = posy1(setdiff(1:Ky1,locRemovey1),1:end); % Gather other "on" particles
                        otherothercolsy1 = posy1(Ky1+1:end,1:end); % Gather "off" particles
                        newposy1 = [othercolsy1;oldcoly1;otherothercolsy1]; % Put removed particle at the end of "on" particles
                        posy1 = newposy1;
                        ny1(Ky1-numFoundy1+1:Ky1,t+1) = 0;
                    end
                end
            end
        end

        %Cell 2 rac
        if resetc2x==0
            if(NNx2(t+1) < NNx2(t))                % diassociation event (particle off)
                oldcolx2 = posx2(minidx2,1:end);
                othercolsx2 = posx2([1:minidx2-1,minidx2+1:Kx2],1:end);
                otherothercolsx2 = posx2(Kx2+1:end,1:end);
                newposx2 = [othercolsx2;oldcolx2;otherothercolsx2];
                posx2 = newposx2;
                nx2(Kx2,t+1) = 0;
            elseif(NNx2(t+1) > NNx2(t))             % association event (on or recruitment)
                rrx2 = rand(1,1);
                posx2(Kx2,t+1) = posx2(Kx2,t)+(rrx2<ponx2)*locx2;              % on event
                posx2(Kx2,t+1) = posx2(Kx2,t)+(rrx2>=ponx2)*posx2(minidx2,t);   % recruitment event
                nx2(Kx2,t+1) = 1;
                % Look for nearby rho (posy2), take them off
                % locx2=location of rac binding
                if numRhoToRemove>0
                    boundC2Scaled=(L*boundC2/Na);
                    locRemovey2 = find(abs(posy2(:,t+1)-posx2(Kx2,t+1))<epsilon,numRhoToRemove);
                    numFoundy2 = length(locRemovey2);
                    if ~isempty(locRemovey2) && boundC2Scaled(1)<=posx2(Kx2,t+1) && boundC2Scaled(end)>=posx2(Kx2,t+1)
                        % posy2(locRemovey2,t+1)=0;
                        oldcoly2 = posy2(locRemovey2,1:end); % Find the particle to be removed
                        othercolsy2 = posy2(setdiff(1:Ky2,locRemovey2),1:end); % Gather other "on" particles
                        otherothercolsy2 = posy2(Ky2+1:end,1:end); % Gather "off" particles
                        newposy2 = [othercolsy2;oldcoly2;otherothercolsy2]; % Put removed particle at the end of "on" particles
                        posy2 = newposy2;
                        ny2(Ky2-numFoundy2+1:Ky2,t+1) = 0;
                    end
                end
            end
        end

        %Cell 1 rho
        if resetc1y==0
            if (NNy1(t+1) < NNy1(t))                % diassociation event (particle off)
                oldcoly1 = posy1(minidy1,1:end);
                othercolsy1 = posy1([1:minidy1-1,minidy1+1:Ky1],1:end);
                otherothercolsy1 = posy1(Ky1+1:end,1:end);
                newposy1 = [othercolsy1;oldcoly1;otherothercolsy1];
                posy1 = newposy1;
                ny1(Ky1,t+1) = 0;
            elseif(NNy1(t+1) > NNy1(t))             % association event (on or recruitment)
                rry1 = rand(1,1);
                posy1(Ky1,t+1) = posy1(Ky1,t)+(rry1<pony1)*locy1;               % on event
                posy1(Ky1,t+1) = posy1(Ky1,t)+(rry1>=pony1)*posy1(minidy1,t);    % recruitment event
                ny1(Ky1,t+1) = 1;

                if numRacToRemove>0
                    boundC1Scaled=(L*boundC1/Na);
                    locRemovex1 = find(abs(posx1(:,t+1)-posy1(Ky1,t+1))<epsilon,numRacToRemove);
                    numFoundx1 = length(locRemovex1);
                    if ~isempty(locRemovex1) && boundC1Scaled(1)<=posy1(Ky1,t+1) && boundC1Scaled(end)>=posy1(Ky1,t+1)
                        oldcolx1 = posx1(locRemovex1,1:end); % Find the particle(s) to be removed
                        othercolsx1 = posx1(setdiff(1:Kx1,locRemovex1),1:end); % Gather other "on" particles
                        otherothercolsx1 = posx1(Kx1+1:end,1:end); % Gather "off" particles
                        newposx1 = [othercolsx1;oldcolx1;otherothercolsx1]; % Put removed particle at the end of "on" particles
                        posx1 = newposx1;
                        nx1(Kx1-numFoundx1+1:Kx1,t+1) = 0;
                    end
                end
            end
        end

        %Cell 2 rho
        if resetc2y==0
            if (NNy2(t+1) < NNy2(t))                % diassociation event (particle off)
                oldcoly2 = posy2(minidy2,1:end);
                othercolsy2 = posy2([1:minidy2-1,minidy2+1:Ky2],1:end);
                otherothercolsy2 = posy2(Ky2+1:end,1:end);
                newposy2 = [othercolsy2;oldcoly2;otherothercolsy2];
                posy2 = newposy2;
                ny2(Ky2,t+1) = 0;
            elseif(NNy2(t+1) > NNy2(t))             % association event (on or recruitment)
                rry2 = rand(1,1);
                posy2(Ky2,t+1) = posy2(Ky2,t)+(rry2<pony2)*locy2;               % on event
                posy2(Ky2,t+1) = posy2(Ky2,t)+(rry2>=pony2)*posy2(minidy2,t);    % recruitment event
                ny2(Ky2,t+1) = 1;

                if numRacToRemove>0
                    boundC2Scaled=(L*boundC2/Na);
                    locRemovex2 = find(abs(posx2(:,t+1)-posy2(Ky2,t+1))<epsilon,numRacToRemove);
                    numFoundx2 = length(locRemovex2);
                    if ~isempty(locRemovex2) && boundC2Scaled(1)<=posy2(Ky2,t+1) && boundC2Scaled(end)>=posy2(Ky2,t+1)
                        oldcolx2 = posx2(locRemovex2,1:end); % Find the particle(s) to be removed
                        othercolsx2 = posx2(setdiff(1:Kx2,locRemovex2),1:end); % Gather other "on" particles
                        otherothercolsx2 = posx2(Kx2+1:end,1:end); % Gather "off" particles
                        newposx2 = [othercolsx2;oldcolx2;otherothercolsx2]; % Put removed particle at the end of "on" particles
                        posx2 = newposx2;
                        nx2(Kx2-numFoundx2+1:Kx2,t+1) = 0;
                    end
                end
            end
        end

        [s1,xC1,yC1] = resamplePolarityMolecules(posx1(1:Kx1,t+1),posy1(1:Ky1,t+1),Kx1,Ky1,L,Na);
        [s2,xC2,yC2] = resamplePolarityMolecules(posx2(1:Kx2,t+1),posy2(1:Ky2,t+1),Kx2,Ky2,L,Na);

        xC1all(:,t+1)=xC1;
        yC1all(:,t+1)=yC1;
        xC2all(:,t+1)=xC2;
        yC2all(:,t+1)=yC2;

        %% Update actin filaments
        diffRHSa1 = Hm1*a1;
        diffRHSb1 = Hm1*b1;

        diffRHSa2 = Hm2*a2;
        diffRHSb2 = Hm2*b2;

        cell1_bound=zeros(length(Xa),1);
        cell1_bound(boundC1)=ones(length(boundC1),1);
        cell2_bound=zeros(length(Xb),1);
        cell2_bound(boundC2)=ones(length(boundC2),1);
        abmax=50;

%         if t<=sigswitch_time
%             kaa_ind=2;
%             kbb_ind=3;
%             kcc_ind=3;
%             kdd_ind=2;
%         else
%             kaa_ind=3;
%             kbb_ind=2;
%             kcc_ind=2;
%             kdd_ind=3;
%         end

        


        rxna1 = dt*( F(a1,b1) + Ka1.*(a1.*(1+alpha(1)*xC1 ...
            + ka_vals(ka_ind) * cell1_bound.* (flip(a2).*(flip(a2)<=abmax) + abmax*(flip(a2)>abmax)) ...
            + kb_vals(kb_ind) * cell1_bound.* (flip(b2).*(flip(b2)<=abmax) + abmax*(flip(b2)>abmax)) ...
            + cell1_bound.*kaa_vals(kaa_ind) )) - a1.*a1); %Cell 1 branched
        rxnb1 = dt*( F(b1,a1) + Kb1.*(b1.*(1+1.8*alpha(1)*yC1 ...
            + kc_vals(kc_ind) * cell1_bound.* (flip(a2).*(flip(a2)<=abmax) + abmax*(flip(a2)>abmax)) ...
            + kd_vals(kd_ind) * cell1_bound.* (flip(b2).*(flip(b2)<=abmax) + abmax*(flip(b2)>abmax)) ...
            + cell1_bound.*kbb_vals(kbb_ind) )) - b1.*b1); %Cell 1 bundled
        rxna2 = dt*( F(a2,b2) + Ka2.*(a2.*(1+alpha(1)*xC2 ...
            + ka_vals(ka_ind) * cell2_bound.* (flip(a1).*(flip(a1)<=abmax) + abmax*(flip(a1)>abmax)) ...
            + kb_vals(kb_ind) * cell2_bound.* (flip(b1).*(flip(b1)<=abmax) + abmax*(flip(b1)>abmax)) ...
            + cell2_bound.*kcc_vals(kcc_ind) )) - a2.*a2); %Cell 2 branched
        rxnb2 = dt*( F(b2,a2) + Kb2.*(b2.*(1+1.8*alpha(1)*yC2 ...
            + kc_vals(kc_ind) * cell2_bound.* (flip(a1).*(flip(a1)<=abmax) + abmax*(flip(a1)>abmax)) ...
            + kd_vals(kd_ind) * cell2_bound.* (flip(b1).*(flip(b1)<=abmax) + abmax*(flip(b1)>abmax)) ...
            + cell2_bound.*kdd_vals(kdd_ind) )) - b2.*b2); %Cell 2 bundled



% if c1_ind==1 %KL
%         rxnb1 = dt*( F(b1,a1) + Kb1.*(b1.*(1+(alpha(1)*(yC1).^2)./(1+(yC1).^2) ...
%             + kc_vals(kc_ind) * cell1_bound.* (flip(a2).*(flip(a2)<=abmax) + abmax*(flip(a2)>abmax)) ...
%             + kd_vals(kd_ind) * cell1_bound.* (flip(b2).*(flip(b2)<=abmax) + abmax*(flip(b2)>abmax)) ...
%             + cell1_bound.*kbb_vals(kbb_ind) )) - b1.*b1); %Cell 1 bundled
%         rxnb2 = dt*( F(b2,a2) + Kb2.*(b2.*(1+(alpha(1)*(yC2).^2)./(1+(yC2).^2) ...
%             + kc_vals(kc_ind) * cell2_bound.* (flip(a1).*(flip(a1)<=abmax) + abmax*(flip(a1)>abmax)) ...
%             + kd_vals(kd_ind) * cell2_bound.* (flip(b1).*(flip(b1)<=abmax) + abmax*(flip(b1)>abmax)) ...
%             + cell2_bound.*kdd_vals(kdd_ind) )) - b2.*b2); %Cell 2 bundled
% end

        a1 = Hs1\(diffRHSa1+rxna1);
        b1 = Hs1\(diffRHSb1+rxnb1);
        a2 = Hs2\(diffRHSa2+rxna2);
        b2 = Hs2\(diffRHSb2+rxnb2);

        a1all(:,t)=a1;
        a2all(:,t)=a2;
        b1all(:,t)=b1;
        b2all(:,t)=b2;

        posx1saved(:,t+1)=posx1(:,t+1);
        posy1saved(:,t+1)=posy1(:,t+1);
        posx2saved(:,t+1)=posx2(:,t+1);
        posy2saved(:,t+1)=posy2(:,t+1);

        %Calculate direction angles
        a1New = a1;
        a1New(a1New<1)=0;
        if (a1New(1)~=0 && a1New(end)~=0)
            zeroInda1_1=find(a1New==0,1,'first');
            zeroInda2_1=find(a1New==0,1,'last');
            dirIndexa1=ceil((zeroInda1_1+zeroInda2_1)/2) - 50;
        else
            inda1_1=find(a1New~=0,1,'first');
            inda2_1=find(a1New~=0,1,'last');
            dirIndexa1=ceil((inda1_1+inda2_1)/2);
        end
        b1New = b1;
        b1New(b1New<1)=0;
        if (b1New(1)~=0 && b1New(end)~=0)
            zeroIndb1_1=find(b1New==0,1,'first');
            zeroIndb2_1=find(b1New==0,1,'last');
            dirIndexb1=ceil((zeroIndb1_1+zeroIndb2_1)/2) - 50;
        else
            indb1_1=find(b1New~=0,1,'first');
            indb2_1=find(b1New~=0,1,'last');
            dirIndexb1=ceil((indb1_1+indb2_1)/2);
        end
        a2New = a2;
        a2New(a2New<1)=0;
        if (a2New(1)~=0 && a2New(end)~=0)
            zeroInda1_2=find(a2New==0,1,'first');
            zeroInda2_2=find(a2New==0,1,'last');
            dirIndexa2=ceil((zeroInda1_2+zeroInda2_2)/2) - 50;
        else
            inda1_2=find(a2New~=0,1,'first');
            inda2_2=find(a2New~=0,1,'last');
            dirIndexa2=ceil((inda1_2+inda2_2)/2);
        end
        b2New = b2;
        b2New(b2New<1)=0;
        if (b2New(1)~=0 && b2New(end)~=0)
            zeroIndb1_2=find(b2New==0,1,'first');
            zeroIndb2_2=find(b2New==0,1,'last');
            dirIndexb2=ceil((zeroIndb1_2+zeroIndb2_2)/2) - 50;
        else
            indb1_2=find(b2New~=0,1,'first');
            indb2_2=find(b2New~=0,1,'last');
            dirIndexb2=ceil((indb1_2+indb2_2)/2);
        end
        if dirIndexa1<1
            dirIndexa1=dirIndexa1+101;
        end
        if dirIndexb1<1
            dirIndexb1=dirIndexb1+101;
        end
        if dirIndexa2<1
            dirIndexa2=dirIndexa2+101;
        end
        if dirIndexb2<1
            dirIndexb2=dirIndexb2+101;
        end
        [th,rad] = meshgrid((0:3.6:360)*pi/180,1);

        if move_cells==1
            xshift1(t+1)=xshift1(t);
            yshift1(t+1)=yshift1(t);
            xshift2(t+1)=xshift2(t);
            yshift2(t+1)=yshift2(t);

            if ~isempty(dirIndexa1) && ~isempty(dirIndexb1)
                xshift1(t+1)=xshift1(t+1)+cos(th(dirIndexa1))*0.001;
                yshift1(t+1)=yshift1(t+1)+sin(th(dirIndexa1))*0.001;
            end
            if ~isempty(dirIndexa2) && ~isempty(dirIndexb2)
                xshift2(t+1)=xshift2(t+1)+cos(th(dirIndexa2))*0.001;
                yshift2(t+1)=yshift2(t+1)+sin(th(dirIndexa2))*0.001;
            end

            posn1=[0+xshift1(t+1),0+yshift1(t+1)];
            posn2=[0+xshift2(t+1),-2+yshift2(t+1)];
        end

        if t==(Nt-1)
            % Calculate difference in direction angles
            angTolerance=pi/4;
            strongAngTolerance=pi/5;
            if (isempty(dirIndexa1) || isempty(dirIndexb1)) && (isempty(dirIndexa2) || isempty(dirIndexb2))
                samedirection='2NP';
                outcome='Non-polarized';
                angdiff=NaN;
            elseif (isempty(dirIndexa1) || isempty(dirIndexb1)) || (isempty(dirIndexa2) || isempty(dirIndexb2))
                samedirection='1NP';
                outcome='Non-polarized';
                angdiff=NaN;
            else
                medang1 = th(1,dirIndexa1);
                medang2 = th(1,dirIndexa2);
                angdiff = min(abs(medang1-medang2),abs(2*pi-abs(medang1-medang2)));
                if angdiff < angTolerance
                    samedirection='yes';
                    outcome='Co-alignment';
                elseif (abs(medang1-3*pi/2)<strongAngTolerance && abs(medang2-pi/2)<strongAngTolerance)
                    samedirection='strong no; collision';
                    outcome='Collision';
                else
                    samedirection='no';
                    outcome='No alignment';
                end
            end
            % sprintf('Median angle difference: %d\nSame direction? %s',angdiff,samedirection)
            sprintf('Angle difference: %d\nOutcome: %s',angdiff,outcome);
        end

        % Make plots
        if vid==1
            make_plot=mod(t,tplot)==0 || t==Nt-1;
        else
            make_plot= (t==Nt-1);
        end
        %make_plot=0;
        if make_plot
            % if t==Nt-1

            %Define colors
            colorLength = 50;
            white = [1,1,1];
            darkyellow = [227/256,180/256,76/256];
            yellow2 = [254/256,254/256,98/256];
            pink = [211/256,95/256,183/256];
            darkpink = [141/256,45/256,113/256];
            whiteyellow2 = [linspace(white(1),yellow2(1),colorLength)',linspace(white(2),yellow2(2),colorLength)',linspace(white(3),yellow2(3),colorLength)'];
            yellow2darkyellow = [linspace(yellow2(1),darkyellow(1),colorLength)',linspace(yellow2(2),darkyellow(2),colorLength)',linspace(yellow2(3),darkyellow(3),colorLength)'];
            whitedarkyellow2 = [whiteyellow2;yellow2darkyellow];
            whitepink = [linspace(white(1),pink(1),colorLength)',linspace(white(2),pink(2),colorLength)',linspace(white(3),pink(3),colorLength)'];
            pinkdarkpink = [linspace(pink(1),darkpink(1),colorLength)',linspace(pink(2),darkpink(2),colorLength)',linspace(pink(3),darkpink(3),colorLength)'];
            whitedarkpink = [whitepink;pinkdarkpink];


            branchedColor = whitedarkpink;
            bundledColor = whitedarkyellow2;
            branchedColName = 'Pink';
            bundledColName = 'Yellow';

            % Define circles
            gapsize=0.01;
            [th,rad] = meshgrid((0:3.6:360)*pi/180,0.85:0.01:1);
            [Xcol,Ycol] = pol2cart(th,rad);
            Ycol1=Ycol;
            Ycol2=Ycol;
            Ycol1(:,boundC1)=Ycol1(:,boundC1(1)*ones(1,length(boundC1)));
            Ycol2(:,boundC2)=Ycol2(:,boundC2(1)*ones(1,length(boundC2)));
            Ycol2 = Ycol2 - 2*abs(max(max(Ycol2)))-gapsize;
            ZBranch1 = [a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1 a1]';
            ZBund1 = [b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1 b1]';
            ZBranch2 = [a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2 a2]';
            ZBund2 = [b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2 b2]';
            [th,rad] = meshgrid((0:3.6:360)*pi/180,0.8);
            [Xsm,Ysm] = pol2cart(th,rad);
            Ysm1=Ysm;
            Ysm2=Ysm;
            Ysm1(:,boundC1)=Ysm1(:,boundC1(1)*ones(1,length(boundC1)));
            Ysm2(:,boundC2)=Ysm2(:,boundC2(1)*ones(1,length(boundC2)));
            [th,rad] = meshgrid((0:3.6:360)*pi/180,0.86:0.01:0.93);
            [Xmid,Ymid] = pol2cart(th,rad);

            % Make scatterplots
            scatplot=figure(ppp);
            clf
            subplot(1,2,1); %Cell 1
            plot(Xa,a1,'-o','color',branchedColor(end,:),'linewidth',3); hold on;
            plot(Xa,b1,'-ok','color',bundledColor(end,:),'linewidth',3);
            plot(s1,xC1,'-.','color',branchedColor(end,:),'linewidth',1);
            plot(s1,yC1,'-.k','color',bundledColor(end,:),'linewidth',1);
            set(gca,'fontname','times','fontsize',20); box on;
            lgd = legend('Branched network','Bundled network','Rac','Rho','Location','northeast');
            lgd.NumColumns = 2;
            set(gcf,'color','w');
            title('Cell 1')
            hold off;

            subplot(1,2,2); %Cell 2
            plot(Xa,a2,'-o','color',branchedColor(end,:),'linewidth',3); hold on;
            plot(Xa,b2,'-ok','color',bundledColor(end,:),'linewidth',3);
            plot(s2,xC2,'-.','color',branchedColor(end,:),'linewidth',1);
            plot(s2,yC2,'-.k','color',bundledColor(end,:),'linewidth',1);
            set(gca,'fontname','times','fontsize',20); box on;
            lgd = legend('Branched network','Bundled network','Rac','Rho','Location','northeast');
            lgd.NumColumns = 2;
            set(gcf,'color','w');
            title('Cell 2')
            hold off;


            % Plot cells
            allmax=12;
            showtime=1;
            cellsplot=figure(ppp+1);
            clf
            range=3;
            hold on
            alphaData=ZBranch1+max(0,max(max(ZBranch2))-max(max(ZBranch1)));
            surf(Xcol+xshift1(t),Ycol+yshift1(t),ZBranch1,'AlphaData',alphaData,'FaceAlpha','interp','FaceColor','interp');
            colormap(branchedColor)
            clim([0,12])
            freezeColors;
            shading interp
            alphaData=ZBund1+max(0,max(max(ZBund2))-max(max(ZBund1)));
            surf(Xcol+xshift1(t),Ycol+yshift1(t),ZBund1,'AlphaData',alphaData,'FaceAlpha','interp','FaceColor','interp');
            colormap(bundledColor)
            clim([0,12])
            freezeColors;
            shading interp
            view(2)
            if max(xC1)>=(range+2)
                racxvals1=(range-1)*xC1/max(xC1)+1;
                racyvals1=(range-1)*xC1/max(xC1)+1;
            else
                racxvals1=(range-2)*xC1/max(xC1)+1;
                racyvals1=(range-1)*xC1/max(xC1)+1;
            end
            racxvals1=(racxvals1)'.*cos(2*pi*Xa/L);
            racyvals1=(racyvals1)'.*sin(2*pi*Xa/L);
            plot3(racxvals1+xshift1(t),racyvals1+yshift1(t),(allmax+1)*ones(1,length(racxvals1)),'color',...
                [branchedColor(end,:),1],'LineWidth',3)
            plot3([racxvals1(end)+xshift1(t),racxvals1(1)+xshift1(t)],[racyvals1(end)+yshift1(t),racyvals1(1)+yshift1(t)],...
                [allmax+1,allmax+1],'color',[branchedColor(end,:),1],'LineWidth',3)
            if max(yC1)>=(range+2)
                rhoxvals1=(range-1)*yC1/max(yC1)+1;
                rhoyvals1=(range-1)*yC1/max(yC1)+1;
            else
                rhoxvals1=(range-2)*yC1/max(yC1)+1;
                rhoyvals1=(range-2)*yC1/max(yC1)+1;
            end
            rhoxvals1=(rhoxvals1)'.*cos(2*pi*Xa/L);
            rhoyvals1=(rhoyvals1)'.*sin(2*pi*Xa/L);
            plot3(rhoxvals1+xshift1(t),rhoyvals1+yshift1(t),(allmax+1)*ones(1,length(rhoxvals1)),'color',...
                [bundledColor(end,:),1],'LineWidth',3)
            plot3([rhoxvals1(end)+xshift1(t),rhoxvals1(1)+xshift1(t)],[rhoyvals1(end)+yshift1(t),rhoyvals1(1)+yshift1(t)],...
                [allmax+1,allmax+1],'color',[bundledColor(end,:),1],'LineWidth',3)
            plot3(cos(2*pi*Xa/L)+xshift1(t),sin(2*pi*Xa/L)+yshift1(t),(allmax+2)*ones(1,Na),'color','black','LineWidth',1)
            hold off

            %cell 2
            hold on
            alphaData=ZBranch2+max(0,max(max(ZBranch1))-max(max(ZBranch2)));
            surf(Xcol+xshift2(t),Ycol+yshift2(t)-2-(range-1),ZBranch2,'AlphaData',alphaData,'FaceAlpha','interp','FaceColor','interp');
            colormap(branchedColor)
            clim([0,12])
            cb=colorbar('Location','eastoutside');
            freezeColors;
            freezeColors(cb);
            cbpos=cb.Position;
            % set(cb,'Position',[cbpos(1)+2*cbpos(3),cbpos(2),cbpos(3),cbpos(4)/2])
            set(cb,'Position',[0.9062    0.1097    0.0235    0.4077])
            set(cb,'TickLabels',{});
            cbpos=cb.Position;
            shading interp
            alphaData=ZBund2+max(0,max(max(ZBund1))-max(max(ZBund2)));
            surf(Xcol+xshift2(t),Ycol+yshift2(t)-2-(range-1),ZBund2,'AlphaData',alphaData,'FaceAlpha','interp','FaceColor','interp');
            colormap(bundledColor)
            clim([0,12])
            freezeColors;
            jcb=jicolorbar;
            freezeColors(jcb);
            jcbpos=jcb.Position;
            set(jcb,'Position',[cbpos(1)+cbpos(3),cbpos(2),cbpos(3),cbpos(4)])
            shading interp
            view(2)
            if max(xC2)>=(range+2)
                racxvals2=(range-1)*xC2/max(xC2)+1;
                racyvals2=(range-1)*xC2/max(xC2)+1;
            else
                racxvals2=(range-2)*xC2/max(xC2)+1;
                racyvals2=(range-2)*xC2/max(xC2)+1;
            end
            racxvals2=(racxvals2)'.*cos(2*pi*Xa/L);
            racyvals2=(racyvals2)'.*sin(2*pi*Xa/L);
            plot3(racxvals2+xshift2(t),racyvals2+yshift2(t)-2-(range-1),(allmax+1)*ones(1,length(racxvals2)),'color',...
                [branchedColor(end,:),0.5],'LineWidth',3,'LineStyle','-.')
            plot3([racxvals2(end)+xshift2(t),racxvals2(1)+xshift2(t)],[racyvals2(end)+yshift2(t),racyvals2(1)+yshift2(t)]-2-(range-1),...
                [allmax+1,allmax+1],'color',[branchedColor(end,:),0.5],'LineWidth',3,'LineStyle','-.')
            if max(yC2)>=(range+2)
                rhoxvals2=(range-1)*yC2/max(yC2)+1;
                rhoyvals2=(range-1)*yC2/max(yC2)+1;
            else
                rhoxvals2=(range-2)*yC2/max(yC2)+1;
                rhoyvals2=(range-2)*yC2/max(yC2)+1;
            end
            rhoxvals2=(rhoxvals2)'.*cos(2*pi*Xa/L);
            rhoyvals2=(rhoyvals2)'.*sin(2*pi*Xa/L);
            plot3(rhoxvals2+xshift2(t),rhoyvals2+yshift2(t)-2-(range-1),(allmax+1)*ones(1,length(rhoxvals2)),'color',...
                [bundledColor(end,:),0.5],'LineWidth',3,'LineStyle','-.')
            plot3([rhoxvals2(end)+xshift2(t),rhoxvals2(1)+xshift2(t)],[rhoyvals2(end)+yshift2(t),rhoyvals2(1)+yshift2(t)]-2-(range-1),...
                [allmax+1,allmax+1],'color',[bundledColor(end,:),0.5],'LineWidth',3,'LineStyle','-.')
            plot(cos(2*pi*Xa/L)+xshift2(t),sin(2*pi*Xa/L)+yshift2(t)-2-(range-1),'color','black','LineWidth',1)

            xlim([-3,3])
            ylim([-8,4])
            set(gca,'plotBoxAspectRatio',[6 12 1]);
            hold off

            cbpos=cb.Position;
            if showtime==1
                if t==Nt-1
                    timebox=annotation('textbox', [0.75, cbpos(2), 0.1, 0.05], 'String', "t = " + (Nt)*0.01,'FitBoxToText','on','EdgeColor','none','FontSize',20);
                else
                    timebox=annotation('textbox', [0.75, cbpos(2), 0.1, 0.05], 'String', "t = " + (t)*0.01,'FitBoxToText','on','EdgeColor','none','FontSize',20);
                end
            end

            if ~isempty(dirIndexa1)
                hold on;
                quiver(0+xshift1(t),0+yshift1(t),Xsm(dirIndexa1),Ysm(dirIndexa1),0,'color',[0 0 0],'LineWidth',2,'MaxHeadSize',2);
                hold off;
            end
            if ~isempty(dirIndexa2)
                hold on;
                quiver(0+xshift2(t),-2-(range-1)+yshift2(t),Xsm(dirIndexa2),Ysm(dirIndexa2),0,'color',[0 0 0],'LineWidth',2,'MaxHeadSize',2)
                hold off;
            end

            if signal==1
                [th,rad] = meshgrid((0:3.6:360)*pi/180,1.1);
                [Xsig,Ysig] = pol2cart(th,rad);
                if t<=sigswitch_time
                    hold on;
                    scatter3(Xsig(sigBound2)+xshift2(t),Ysig(sigBound2)+yshift2(t)-2-(range-1),(allmax+3)*ones(1,length(sigBound2)),50,'black','.')
                    hold off;
                else
                    hold on;
                    scatter3(Xsig(sigBound1)+xshift1(t),Ysig(sigBound1)+yshift1(t),(allmax+3)*ones(1,length(sigBound1)),50,'black','.')
                    hold off;
                end
            end

            grid off
            set(gca,'XTick',[],'YTick',[])
            set(gca,'XColor','w')
            set(gca,'YColor','w')
            set(gcf,'color','w');
            set(gcf,'Position',[209   561   682   474])
            ohf = findobj(gcf);
            figaxes = findobj(ohf(1), 'Type', 'axes');
            set(figaxes(1),'Fontsize',15)
            set(figaxes(2),'Fontsize',14)
            camroll(90)


            if vid==1
                scatframe = getframe(scatplot);
                writeVideo(vidObjScatter,scatframe);
                cellsframe = getframe(cellsplot);
                writeVideo(vidObjCells,cellsframe);
            end

            if t==Nt-1 && savefigs==1
                savefig(cellsplot,filenameCells);
                savefig(scatplot,filenameScatter);
            end
        end


    end

    if vid==1
        close(vidObjScatter);
        close(vidObjCells);
    end

    % measure of polarized state (1 if polarized and 0 otherwise)
    %st = 1*( (abs(a(1)-b(end))<1e-3 || abs(a(end)-b(1))<1e-3 ) && abs(a(1)-a(end))>1e-3 && abs(b(1)-b(end))>1e-3 );

    sprintf('Simulation %d done',ppp)
    toc
    if(quit_cond==0)
        if save_matfile==1
            save(strcat(mat_location,int2str(ppp),'.mat'),...
                'boundC1','boundC2','posx1saved','posx2saved','posy1saved','posy2saved','NNx1','NNx2',...
                'NNy1','NNy2','a1all','a2all','b1all','b2all','Xa','Xb','s1','s2',...
                'xC1','xC2','yC1','yC2','xshift1','yshift1','xshift2','yshift2',...
                'posn1','posn2','xC1all','yC1all','xC2all','yC2all','sigBound1','sigBound2','Nt','Tend')
        end
        ppp = ppp + 1;

        if writem==1
            if strcmp(samedirection, 'yes')
                res_counters(1)=res_counters(1)+1;
            elseif strcmp(samedirection, 'strong no; collision')
                res_counters(2)=res_counters(2)+1;
            elseif strcmp(samedirection, '1NP')
                res_counters(3)=res_counters(3)+1;
            elseif strcmp(samedirection, '2NP')
                res_counters(4)=res_counters(4)+1;
            else
                res_counters(5)=res_counters(5)+1;
            end

            anglelf=pi/4;
            if ~isempty(dirIndexa1) && ~isempty(dirIndexa2) && ~isempty(dirIndexb1) && ~isempty(dirIndexb2)
                if xor(abs(medang1-3*pi/2)<anglelf, abs(medang2-pi/2)<anglelf)
                    res_counters(6)=res_counters(6)+1;
                end
            end

            angledist=pi/4;
            if isempty(dirIndexa1) && ~isempty(dirIndexa2) && max(b1)>1
                medang2 = th(1,dirIndexa2);
                if abs(medang2-pi/2)>angledist
                    res_counters(7)=res_counters(7)+1;
                end
            end
            if isempty(dirIndexa2) && ~isempty(dirIndexa1) && max(b2)>1
                medang1 = th(1,dirIndexa1);
                if abs(medang1-3*pi/2)>angledist
                    res_counters(7)=res_counters(7)+1;
                end
            end
            if isempty(dirIndexa1) && isempty(dirIndexa2) && ((max(b1)>1 && max(a2)>1) || (max(b2)>1 && max(a1)>1))
                res_counters(7)=res_counters(7)+1;
            end
        end
    end

    if writem==1
        options=["koffx","koffy","konx","kony"];
        writematrix(res_counters,strcat(savelocation,...
            int2str(coeff_vals(c1coeff_ind)),options(c1_ind),'C1_',...
            int2str(coeff_vals(c2coeff_ind)),options(c2_ind),'C2','.xls'))
        sprintf(int2str(res_counters))
    end
end

%    end
%end
%    end
%end
end