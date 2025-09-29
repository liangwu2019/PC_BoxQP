%%
% Collects data and builds the lifting-based predictor for the Kortweg - de Vries PDE
% y_t + yy_x + y_xxx = u, where u is the control input

clear, clc
SimPar.N = 128; % spatial scretization (uniform mesh)
SimPar.T = 0.01; % time step

%% Define inputs
x = linspace(-pi,pi,SimPar.N)';
% Four initial conditions
IC1 = exp(-(((x)-pi/2)).^2); % gaussian initial condition
IC2 = -sin(x/2).^2;
IC3 = exp(-(((x)+pi/2)).^2);
IC4 = cos(x/2).^2;


% Gaussian control input profiles
v1 = exp(-25*(x-pi/2).^2);
v2 = exp(-25*(x-pi/6).^2);
v3 = exp(-25*(x+pi/6).^2);
v4 = exp(-25*(x+pi/2).^2);
m = 4; % number of control inputs

% Transition mapping of the controleld dynamical system
f = @(x,u)(kdv_solver(x,u(1)*v1+u(2)*v2+u(3)*v3+u(4)*v4,SimPar));

% Collect data
disp('Starting data collection')
SimLength = 200; % Number of steps per trajectory
Ntraj = 1000;

umin = -1;
umax = 1;

Ubig= rand(m,SimLength,Ntraj)* (umax-umin)+umin;
disp('run and collect data')

X = []; Y = []; U=[];   % initialize
for i = 1:Ntraj
    xx = [];
    b = rand(4,1); b = b/sum(b);
    % Intial state is a random convex combination of four initial conitions
    xx =b(1)*IC1 + b(2)*IC2 + b(3)*IC3 + b(4)*IC4;
    % Simulate one trajectory
    tic
    fprintf('Trajectory %d out of %d \n',i,Ntraj)
    for j = 1:SimLength
        xx = [xx f(xx(:,end),Ubig(:,j,i))];
        U  = [U,Ubig(:,j,i)];
        % if the solution diverges, go to the next trajectory
        if ~isempty(find(isnan(xx(:,end)),1))
            disp('Bad trajectory')
            break
        end
    end
    toc
    % Store
    X = [X xx(:,1:end-1)];
    Y = [Y xx(:,2:end)];
end
%% save generation data
save('KdV_data.mat','U','X','Y');
