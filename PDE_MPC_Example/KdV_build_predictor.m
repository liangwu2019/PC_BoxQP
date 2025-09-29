%% Build predictor
SimPar.N = 128; % spatial scretization (uniform mesh)
SimPar.T = 0.01; % time step
load('KdV_data.mat','U','X','Y')

%%
basisFunction = 'rbf';
Nrbf = 1*SimPar.N;
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

save('KdV_predictor_2N.mat','M','SimPar','liftFun');
%%
basisFunction = 'rbf';
Nrbf = 3*SimPar.N;
cent = rand(SimPar.N,Nrbf)*2 - 1; % RBF centers
rbf_type = 'thinplate'; 
% Lifting mapping: the state itself + RBFs 
liftFun = @(xx)( [xx;rbf(xx,cent,rbf_type)] );

Xp = liftFun(X(1:SimPar.N,:));
Yp = liftFun(Y(1:SimPar.N,:));

% Regression for Matrix M = [A,B] (Eq. (22) in the paper)
W = [Xp;U]*[Xp;U]';
V = Yp*[Xp;U]';
M = V*pinv(W);

save('KdV_predictor_4N.mat','M','SimPar','liftFun');

%%
basisFunction = 'rbf';
Nrbf = 5*SimPar.N;
cent = rand(SimPar.N,Nrbf)*2 - 1; % RBF centers
rbf_type = 'thinplate'; 
% Lifting mapping: the state itself + RBFs 
liftFun = @(xx)( [xx;rbf(xx,cent,rbf_type)] );

Xp = liftFun(X(1:SimPar.N,:));
Yp = liftFun(Y(1:SimPar.N,:));

% Regression for Matrix M = [A,B] (Eq. (22) in the paper)
W = [Xp;U]*[Xp;U]';
V = Yp*[Xp;U]';
M = V*pinv(W);

save('KdV_predictor_6N.mat','M','SimPar','liftFun');