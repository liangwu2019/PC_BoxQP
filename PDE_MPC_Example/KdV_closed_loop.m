%%
clear all; close all; clc

%%
SimPar.N = 128; % spatial scretization (uniform mesh)
SimPar.T = 0.01; % time step

%% Control inputs
x = linspace(-pi,pi,SimPar.N)';
nu = 4; % Number of control inputs

% Gaussian control profiles
v1 = exp(-25*(x-pi/2).^2);
v2 = exp(-25*(x-pi/6).^2);
v3 = exp(-25*(x+pi/6).^2);
v4 = exp(-25*(x+pi/2).^2);

%% Build a Koopman predictor
% % To build a new predictor run KdV_build_predictor.m
load('KdV_data.mat','U','X','Y')
basisFunction = 'rbf';
Nrbf = 2 * SimPar.N;
cent = rand(SimPar.N, Nrbf)*2 - 1; % RBF centers
rbf_type = 'thinplate'; 
% Lifting mapping: the state itself + RBFs 
liftFun = @(xx)( [xx;rbf(xx,cent,rbf_type)] );

Xp = liftFun(X(1:SimPar.N,:));
Yp = liftFun(Y(1:SimPar.N,:));

% Regression for Matrix M = [A,B] (Eq. (22) in the paper)
W = [Xp;U]*[Xp;U]';
V = Yp*[Xp;U]';
M = V*pinv(W);

%% Predictor
n = SimPar.N;
Nlift = numel(liftFun(rand(n,1)));
A = M(:,1:Nlift);
B = M(:,Nlift+1:end);
C = [eye(n,n), zeros(n,Nlift - n)];

%% Set up MPC problem

% Only input control constraints
umin = -ones(nu,1);
umax = ones(nu,1); 

% Weight matrices of MPC
W_N = eye(SimPar.N);
W_x = eye(SimPar.N);
R = 0.01*eye(nu,nu); % Increase to make control less agressive

% Prediction horizon
Tpred = 0.1;
Np = round(Tpred / SimPar.T);

% Condensing MPC construction
AiB = B;
BB = kron(eye(Np),AiB);
for i=1:Np-1
    AiB = A*AiB;
    BB = BB + kron(diag(ones(Np-i,1),-i),AiB);
end
QQ = blkdiag(kron(eye(Np-1),C'*W_x*C),C'*W_N*C);
RR = kron(eye(Np),R);
H = BB'*QQ*BB+RR;

%% Closed-loop simulation

% Intial condition
x0 = zeros(n,1);
X = x0;
U = [];

Run_time_PC_BoxQP = [];
Iters_PC_BoxQP = [];

Run_time_Ref1_BoxQP = [];

% Simulation length
Tsim = 50;
Nsim = Tsim / SimPar.T;

% Spatial refrenence profile (varies in time)
Xref = [];
VIDEO_Opt = 1;
for i=1:Nsim
    if(i < Nsim / 4)
         xr_t = 0.5;
    elseif (i > Nsim / 4 &&  i < 2*Nsim / 4 )
        xr_t = 0.25;
    elseif (i > 2*Nsim / 4 &&  i < 3*Nsim / 4 )
        xr_t = 0.0;
    elseif (i > 3*Nsim / 4 )
        xr_t = 0.75;
    end
    xref = xr_t*ones(SimPar.N,1);
    Xref = [Xref, xr_t];
    %% receving liftFun(X(:,end)) and xref
    ei = A*liftFun(X(:,end));
    ee = ei;
    for k=2:Np
        ei = A*ei;
        ee = [ee; ei];
    end
    h = BB'*(QQ*(ee)-repmat(C'*W_x*xref,Np,1));
    %%
    [z_Ref1,run_time_Ref1] = Ref1_BoxQP_Matlab(H,h,1e-6);
    Run_time_Ref1_BoxQP = [Run_time_Ref1_BoxQP, run_time_Ref1];
    %%
    [z,iters,max_iters,run_time] = PC_BoxQP_Matlab(H,h,1e-6);
    Run_time_PC_BoxQP = [Run_time_PC_BoxQP, run_time];
    Iters_PC_BoxQP = [Iters_PC_BoxQP, iters];
    %%
    u = z(1:nu);
    x_next = kdv_solver(X(:,end),u(1)*v1+u(2)*v2+u(3)*v3+u(4)*v4, SimPar);
    X = [X, x_next];
    U = [U, u];
    if(mod(i,10) == 0 && VIDEO_Opt)
        clf
        plot(x,X(:,i),'-b','Linewidth',3); hold on
        plot(x,ones(size(x))*Xref(i),'color','[0.9 0 0]','linestyle','--','linewidth',3)
        xlabel('$x$','Interpreter','Latex','Fontsize',30)
        title(['t = ' num2str(SimPar.T*i) 's'])
        LEG = legend('$y(t,x)$', 'Reference');
        set(LEG,'interpreter','latex','fontsize',25);
        ylim([-0.5,1])
        xlim([-pi,pi])
        pause(0.01)
    end    
end
avg_Run_time_PC_BoxQP = sum(Run_time_PC_BoxQP)/length(Run_time_PC_BoxQP)
avg_Iters_PC_BoxQP = sum(Iters_PC_BoxQP)/length(Run_time_PC_BoxQP)
std_Iters_PC_BoxQP = std(Iters_PC_BoxQP)
avg_Run_time_Ref1_BoxQP = sum(Run_time_Ref1_BoxQP)/length(Run_time_Ref1_BoxQP)

%%  Plots
% Control inputs
figure
plot([0 Tsim],[-1 -1],'-k','linewidth',2); hold on
plot([0 Tsim],[1 1],'-k','linewidth',2)
h1 = plot([0:1:Nsim-1]*SimPar.T, U(1,:),'linewidth',4); hold on
h2 = plot([0:1:Nsim-1]*SimPar.T,U(2,:),'linewidth',4);
h3 = plot([0:1:Nsim-1]*SimPar.T,U(3,:),'--','linewidth',4);
h4 = plot([0:1:Nsim-1]*SimPar.T,U(4,:),'--','linewidth',4);
ylim([-1.1,1.1])
xlim([0,Tsim])
LEG = legend([h1 h2 h3 h4],'$u_1(t)$','$u_2(t)$','$u_3(t)$','$u_4(t)$');
set(LEG,'interpreter','latex','location','north');
set(gca,'TickLabelInterpreter','latex');
xlabel('t[s]', 'Interpreter','Latex')
set(gca,'FontSize',15);

% Spatial mean
figure
plot([0:1:Nsim]*SimPar.T,mean(X),'linewidth',4); hold on
plot([0:1:Nsim-1]*SimPar.T,Xref,'--r','linewidth',4); hold on
xlim([0,Tsim])
xlabel('t[s]', 'Interpreter','Latex')
set(gca,'FontSize',15);
set(gca,'TickLabelInterpreter','latex');
LEG = legend('Spatial mean of $y(t,x)$', 'Reference','location','northwest');
set(LEG,'interpreter','latex','fontsize',15);


% Surface plot
figure
t = 0:SimPar.T:Tsim;
[TT, XX] = meshgrid(t,x);
surf(TT,XX,X); 
shading flat
colormap('jet')
set(gca,'cameraViewAngle', 8.8);
view([-140.5000   20.0000])
set(gca,'Fontsize',20)
set(gca,'YTick',-pi:pi/2:pi)
set(gca,'YTickLabel',{'$-\pi$','$-\pi/2$','$0$','$\pi/2$','$\pi$'},'TickLabelInterpreter','latex')
xlabel('$t[s]$', 'interpreter','latex')
ylabel('$x$', 'interpreter','latex')
zlabel('$y(t,x)$', 'interpreter','latex')













