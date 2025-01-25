%% Life-cycle portfolio-choice model
%
%  2 markov shocks: 
%  - persistent labour shock z1;
%  - permanent retirement shock (death of spouse) z2 (age-dependent)
%
%  2 iid shocks: 
%  - transitory labour shock e1;
%  - transitory retirement shock e2 (e.g., health expenses)
%
%  Tail event: stock market disaster 

%% Intials

Params.agejshifter=19; 
Params.J=100-Params.agejshifter; 

% Grid sizes to use
n_d=[101,51]; 
n_a=101;
n_z=[11,2]; 
n_e=[5,5];
n_u=5;

N_j=Params.J; 
vfoptions.riskyasset=1;
simoptions.riskyasset=1;

vfoptions.refine_d=[0,1,1]; 
simoptions.refine_d=vfoptions.refine_d;

%% Parameters

Params.beta = 0.96;
Params.sigma=5; 

% Prices
Params.w=1;

% Asset returns
Params.r=0.02; 
Params.rp=0.03; 
Params.sigma_u=0.2; 
Params.rho_u=0; 

% Demographics
Params.agej=1:1:Params.J; 
Params.Jr=46;

% Pensions
Params.pension=0.6;

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.8,1.2,Params.Jr-15),linspace(1.2,1,14),ones(1,Params.J-Params.Jr+1)];

% Persistent AR(1) process
Params.rho_z1=0.9;
Params.sigma_epsilon_z1=0.02;

% Transitiory iid normal process e1
Params.rho_e1=0;
Params.sigma_epsilon_e1=0.01; % Implictly, rho_z2=0

% Transitiory iid normal process e2
Params.rho_e2=0;
Params.sigma_epsilon_e2=0.01; % Implictly, rho_z2=0

% Survival probabilities
Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000]; 

Params.sj=1-Params.dj(21:101); 
Params.sj(end)=0; 

%% Grids

a_grid=13*(linspace(0,1,n_a).^3)'; 

% iid process for stock return
[u_grid, pi_u]=discretizeAR1_FarmerToda(Params.rp,Params.rho_u,Params.sigma_u,n_u-1);
pi_u=pi_u(1,:)'; 
% Stock disaster
u_grid=[-0.5; u_grid];
pi_u=[0.01; (1-0.01)*pi_u];

% AR1 process for labour income (z1)
[z1_grid,pi_z1]=discretizeAR1_FarmerToda(0,Params.rho_z1,Params.sigma_epsilon_z1,n_z(1));
z1_grid=exp(z1_grid); 
[mean_z1,~,~,~]=MarkovChainMoments(z1_grid,pi_z1);
z1_grid=z1_grid./mean_z1;

% Permanent retirement shock (death of spouse) z2 (age-dependent)
z2_grid=[1;0.5]; % 50% probability for pension drop
Params.dj_1=Params.dj(21:101);
p_B_j = zeros(1,N_j); % chance that drop remains forever 
pi_z2_J = zeros(n_z(2),n_z(2),N_j);

for j=1:N_j
    prob_j = [Params.dj_1(j),1-Params.dj_1(j);
              p_B_j(j),1-p_B_j(j)];
    pi_z2_J(:,:,j) = prob_j;
end

% Combine z1, z2 together
z_grid=[z1_grid; z2_grid];
z_grid_J=z_grid.*ones(1,N_j);
pi_z_J  = zeros(n_z(1)*n_z(2),n_z(1)*n_z(2),N_j); 
for j=1:N_j
    pi_z_J(:,:,j)  = kron(pi_z2_J(:,:,j),pi_z1);  
end

% iid shock 1: transitory labour income shock (e1)
[e1_grid,pi_e1]=discretizeAR1_FarmerToda(0,0,Params.sigma_epsilon_e1,n_e(1)-1);
e1_grid=exp(e1_grid);
pi_e1=pi_e1(1,:)'; 
e1_grid=[0.8; e1_grid];
pi_e1=[0.1; (1-0.1)*pi_e1];
mean_e1=pi_e1'*e1_grid; 
e1_grid=e1_grid./mean_e1; 

[e2_grid,pi_e2]=discretizeAR1_FarmerToda(0,0,Params.sigma_epsilon_e2,n_e(2)-1);
e2_grid=exp(e2_grid);
pi_e2=pi_e2(1,:)'; 
e2_grid=[0.5; e2_grid];
pi_e2=[0.1; (1-0.1)*pi_e2];
mean_e2=pi_e2'*e2_grid; 
e2_grid=e2_grid./mean_e2; 

% Combine e1, e2 together
e_grid=[e1_grid; e2_grid];
pi_e=kron(pi_e2, pi_e1);

% Share of assets invested in the risky asset
riskyshare_grid=linspace(0,1,n_d(2))'; 
d_grid=[a_grid; riskyshare_grid];

%% Aprime function
% riskyasset: aprime_val=aprimeFn(d,u)
aprimeFn=@(riskyshare,savings, u, r) PortfolioCoiceModel4_1_aprimeFn(riskyshare,savings, u, r);

%% Put the risky asset into vfoptions and simoptions
vfoptions.aprimeFn=aprimeFn;
vfoptions.n_u=n_u;
vfoptions.u_grid=u_grid;
vfoptions.pi_u=pi_u;
simoptions.aprimeFn=aprimeFn;
simoptions.n_u=n_u;
simoptions.u_grid=u_grid;
simoptions.pi_u=pi_u;
simoptions.a_grid=a_grid;
simoptions.d_grid=d_grid;

vfoptions.n_e=n_e;
vfoptions.e_grid=e_grid;
vfoptions.pi_e=pi_e;
simoptions.n_e=n_e;
simoptions.e_grid=e_grid;
simoptions.pi_e=pi_e;

%% Return function 
DiscountFactorParamNames={'beta','sj'};

ReturnFn=@(savings,a,z1,z2,e1,e2,w,sigma,agej,Jr,pension,kappa_j) ...
    PortfolioCoiceModel4_1_ReturnFn(savings,a,z1,z2,e1,e2,w,sigma,agej,Jr,pension,kappa_j)

%% Value function iteration problem
disp('Test ValueFnIter')
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

% Compare
size(V)
% with
[n_a,n_z(1),n_z(2),n_e(1),n_e(2),N_j]
% Policy is
size(Policy)
% which is the same as
[length(n_d)+length(n_a),n_a,n_z(1),n_z(2),n_e(1),n_e(2),N_j]

%% Ploting V and Policy

zind_1=floor(n_z(1)+1)/2; % This will be the median
zind_2=floor(n_z(2))/2; % This will be the median
eind_1=floor(n_e(1)+1)/2; % This will be the median
eind_2=floor(n_e(2)+1)/2; % This will be the median

figure(1)
subplot(2,1,1); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(V(:,zind_1,zind_2,eind_1,eind_2,:),[n_a,Params.J]))
title('Value function: median value of z')
xlabel('Age j')
ylabel('Assets (a)')
subplot(2,1,2); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(Params.agejshifter+(1:1:Params.J)),reshape(V(:,zind_1,zind_2,eind_1,eind_2,:),[n_a,Params.J]))
title('Value function: median value of z')
xlabel('Age in Years')
ylabel('Assets (a)')

figure(2)
subplot(5,1,1); plot(a_grid,V(:,1,1),a_grid,V(:,zind_1,zind_2,eind_1,eind_2,1),a_grid,V(:,end,1)) % j=1
title('Value fn at age j=1')
legend('min z','median z','max z') 
subplot(5,1,2); plot(a_grid,V(:,1,20),a_grid,V(:,zind_1,zind_2,eind_1,eind_2,20),a_grid,V(:,end,20)) % j=20
title('Value fn at age j=20')
subplot(5,1,3); plot(a_grid,V(:,1,45),a_grid,V(:,zind_1,zind_2,eind_1,eind_2,45),a_grid,V(:,end,45)) % j=45
title('Value fn at age j=45')
subplot(5,1,4); plot(a_grid,V(:,1,46),a_grid,V(:,end,46),a_grid,V(:,end,46)) % j=46
title('Value fn at age j=46 (first year of retirement)')
subplot(5,1,5); plot(a_grid,V(:,1,81),a_grid,V(:,zind_1,zind_2,eind_1,eind_2,81),a_grid,V(:,end,81)) % j=81
title('Value fn at age j=81')
xlabel('Assets (a)')

figure(3)
PolicyVals=PolicyInd2Val_Case1_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions);
subplot(2,1,1); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(PolicyVals(1,:,zind_1,zind_2,eind_1,eind_2,:),[n_a,Params.J]))
title('Policy function: savings, median z')
xlabel('Age j')
ylabel('Assets (a)')
zlabel('Savings')
subplot(2,1,2); surf(a_grid*ones(1,Params.J),ones(n_a,1)*(1:1:Params.J),reshape(PolicyVals(2,:,zind_1,zind_2,eind_1,eind_2,:),[n_a,Params.J]))
title('Policy function: riskyshare, median z')
xlabel('Age j')
ylabel('Assets (a)')
zlabel('share of savings invested in risky assets (riskyshare)')

figure(4)
subplot(5,2,1); plot(a_grid,PolicyVals(1,:,1,1),a_grid,PolicyVals(1,:,zind_1,zind_2,eind_1,eind_2,1),a_grid,PolicyVals(1,:,end,1)) % j=1
title('Policy for savings at age j=1')
subplot(5,2,3); plot(a_grid,PolicyVals(1,:,1,20),a_grid,PolicyVals(1,:,zind_1,zind_2,eind_1,eind_2,20),a_grid,PolicyVals(1,:,end,20)) % j=20
title('Policy for savings at age j=20')
subplot(5,2,5); plot(a_grid,PolicyVals(1,:,1,45),a_grid,PolicyVals(1,:,zind_1,zind_2,eind_1,eind_2,45),a_grid,PolicyVals(1,:,end,45)) % j=45
title('Policy for savings at age j=45')
subplot(5,2,7); plot(a_grid,PolicyVals(1,:,1,46),a_grid,PolicyVals(1,:,zind_1,zind_2,eind_1,eind_2,46),a_grid,PolicyVals(1,:,end,46)) % j=46
title('Policy for savings at age j=46 (first year of retirement)')
subplot(5,2,9); plot(a_grid,PolicyVals(1,:,1,81),a_grid,PolicyVals(1,:,zind_1,zind_2,eind_1,eind_2,81),a_grid,PolicyVals(1,:,end,81)) % j=81
title('Policy for savings at age j=81')
xlabel('Assets (a)')
subplot(5,2,2); plot(a_grid,PolicyVals(2,:,1,1),a_grid,PolicyVals(2,:,zind_1,zind_2,eind_1,eind_2,1),a_grid,PolicyVals(2,:,end,1)) % j=1
title('Policy for riskyshare at age j=1')
legend('min z','median z','max z') % Just include the legend once in the top-right subplot
subplot(5,2,4); plot(a_grid,PolicyVals(2,:,1,20),a_grid,PolicyVals(2,:,zind_1,zind_2,eind_1,eind_2,20),a_grid,PolicyVals(2,:,end,20)) % j=20
title('Policy for riskyshare at age j=20')
subplot(5,2,6); plot(a_grid,PolicyVals(2,:,1,45),a_grid,PolicyVals(2,:,zind_1,zind_2,eind_1,eind_2,45),a_grid,PolicyVals(2,:,end,45)) % j=45
title('Policy for riskyshare at age j=45')
subplot(5,2,8); plot(a_grid,PolicyVals(2,:,1,46),a_grid,PolicyVals(2,:,zind_1,zind_2,eind_1,eind_2,46),a_grid,PolicyVals(2,:,end,46)) % j=46
title('Policy for riskyshare at age j=46 (first year of retirement)')
subplot(5,2,10); plot(a_grid,PolicyVals(2,:,1,81),a_grid,PolicyVals(2,:,zind_1,zind_2,eind_1,eind_2,81),a_grid,PolicyVals(2,:,end,81)) % j=81
title('Policy for riskyshare at age j=81')
xlabel('Assets (a)')

%% Initial distribution of agents at birth (j=1)
jequaloneDist=zeros([n_a,n_z,n_e],'gpuArray'); 
jequaloneDist(1,floor((n_z(1)+1)/2),floor((n_z(2)+1)/2),floor((n_e(1)+1)/2),floor((n_e(2)+1)/2))=1; 

%% We now compute the 'stationary distribution' of households
Params.mewj=ones(1,Params.J); 
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); 
AgeWeightsParamNames={'mewj'}; 
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Params,simoptions);

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
FnsToEvaluate.riskyshare=@(savings,riskyshare,a,z1,z2,e1,e2) riskyshare; 
FnsToEvaluate.stockmarketparticpation=@(savings,riskyshare,a,z1,z2,e1,e2) (savings>0)*(riskyshare>0);
FnsToEvaluate.earnings=@(savings,riskyshare,a,z1,z2,e1,e2,w,kappa_j) w*kappa_j*z1*e1; 
FnsToEvaluate.assets=@(savings,riskyshare,a,z1,z2,e1,e2) a; 

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,simoptions);

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets
figure(6)
subplot(4,1,1); plot(1:1:Params.J,AgeConditionalStats.riskyshare.Mean)
title('Life Cycle Profile: Share of savings invested in risky asset (riskyshare)')
subplot(4,1,2); plot(1:1:Params.J,AgeConditionalStats.stockmarketparticpation.Mean)
title('Life Cycle Profile: Stock market participation rate (stockmarketparticpation)')
subplot(4,1,3); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j z)')
subplot(4,1,4); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')
