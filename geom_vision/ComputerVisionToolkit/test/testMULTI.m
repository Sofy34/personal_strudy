
close all
reset_random;

normF = @(x) norm(x,'fro');

n_imm = 8;  % # of images
n_pts = 25; % # of points
density = .8; % holes in the adjacency and visibility

noise =  1e-4;

M = 1.2*(rand(3, n_pts)-.5);  % 3D points

K = [-500 0 200
    0  -500 200
    0    0   1] ; % internal parameters


plot3(M(1,:), M(2,:), M(3,:), 'o'); hold on

% ground truth
G_gt = cell(1,n_imm);
P_gt = cell(1,n_imm);
C_gt = cell(1,n_imm);
m = cell(1,n_imm);

for i = 1:n_imm
    Cop = 5 * (rand(3,1) -.5);
    Cop(3) = 10 + 2*(rand-.5);
    C_gt{i} =  Cop;  %centri
    G_gt{i} = camera(  Cop, rand(3,1) -.5, [0;1;0]);
    P_gt{i}  = K*G_gt{i};
    G_gt{i} = [G_gt{i}; 0 0 0 1];
    plotcam(P_gt{i},.7);
    
    m{i}    = htx(P_gt{i},M)  + noise*randn(2,1); % noise
end


disp(' ');
%-------------------------------------------------------------------------
% SMZ calibration

% planar points on a grid
[Xgrid,Ygrid] = meshgrid(-0.5:0.2:0.4);
M_grid = [Xgrid(:)';Ygrid(:)'; zeros(1,size(Xgrid(:),1))];
plot3(M_grid(1,:), M_grid(2,:), M_grid(3,:), '.')
zlabel('Z')

H_est=cell(1,1);
m_grid{i} =  cell(1,n_imm);
for i = 1:n_imm
    m_grid{i}  = htx(P_gt{i},M_grid)  + noise*randn(size(2,100)); % noise 
    
    % H{i} = dlt(m{i},M);
    H_est{i}  = hom_lin(m_grid{i}, M_grid(1:2,:));
end

[P_est ,K_est] = calibSMZ(H_est);

fprintf('CalibSMZ reproj RMS error:\t\t %0.5g \n',...
    rmse(reproj_res_batch(P_est,M_grid, m_grid)) );

[P_out,M_out] = bundleadj(P_est,M_grid,m_grid,...
    'AdjustCommonInterior','InteriorParameters',5, 'FixedPoints',size(M_grid,2));

fprintf('CalibSMZ reproj RMS error:\t\t %0.5g \n', ...
    rmse(reproj_res_batch(P_out,M_out,m_grid)) );

%--------------------------------------------------------------
% (Auto)Calibrazione da H_infinito LVH

H12= K*eul(rand(1,3))*inv(K); %#ok<*MINV>
H13= K*eul(rand(1,3))*inv(K);
H23= K*eul(rand(1,3))*inv(K);

K_out  = calibLVH( {H12,H13,H23 });


fprintf('H_infty calibration %% error:\t %0.5g \n',100*abs(K_out(1,1)-K(1,1))/K(1,1)) ;

%-------------------------------------------------------------------------
% Triangulation

X_est=triang_lin_batch(P_gt, m);
fprintf('Triangulation batch error:\t %0.5g \n', normF(M-X_est)/n_pts  );

%-------------------------------------------------------------------------
% Projective rec

[P_proj,M_proj] = prec(m);

fprintf('Projective Recon reproj RMS error:\t %0.5g \n',...
    rmse(reproj_res_batch(P_proj,M_proj, m)));

%---------------------------------------------------------------------
% Bundle Adjustment

% random visibility
vis = rand(n_pts,n_imm) < density; % is logical
figure, spy(vis),title('Visibility');ylabel('points');xlabel('images')
if any(sum(vis,2)  < 3)
    error('Not enough visibility')
end

kappa = num2cell(.2*ones(1,n_imm),1);
M_in = X_est;   
% Initial rec
for i = 1:n_imm
    P_in{i} = P_gt{i}; 
    m{i} = rdx(kappa{i},m{i},K);  % add radial
end

fprintf('Bundle Adjustment RMS error (before):\t %0.5g \n',...
    rmse(reproj_res_batch(P_in, M_in, m, 'Visibility', vis)) ); 

%[P_out,M_out] = bundleadj(P_in,M_in,m,'Visibility',vis,'FixedInterior');
[P_out,M_out] = bundleadj(P_in,M_in,m,'Visibility',vis,'FixedInterior', 'DistortionCoefficients', kappa); 
kappa_out = kappa;
fprintf('Bundle Adjustment RMS error (after):\t %0.5g \n', ...
    rmse(reproj_res_batch(P_out,M_out, m, 'Visibility',vis,'DistortionCoefficients', kappa_out) ));


% [P_out,M_out] = bundleadj(P_in,M_in,m,'Visibility',vis,'AdjustCommonInterior','InteriorParameters',5);
[P_out,M_out, kappa_out] = bundleadj(P_in,M_in,m,'Visibility',vis,'AdjustCommonInterior','InteriorParameters',5,'DistortionCoefficients', num2cell(zeros(1,n_imm),1));
fprintf('Bundle Adjustment RMS error (after):\t %0.5g \n', ...
    rmse(reproj_res_batch(P_out,M_out, m, 'Visibility',vis,'DistortionCoefficients', kappa_out) ));

% 
% [P_out,M_out,kappa_out] = bundleadj(P_in,M_in,m,'Visibility',vis,'AdjustSeparateInterior');
 [P_out,M_out,kappa_out] = bundleadj(P_in,M_in,m,'Visibility',vis,'AdjustSeparateInterior','DistortionCoefficients', num2cell(zeros(1,n_imm),1));
fprintf('Bundle Adjustment RMS error (after):\t %0.5g \n', ...
    rmse(reproj_res_batch(P_out,M_out, m, 'Visibility',vis,'DistortionCoefficients', kappa_out) ));


disp(' ');
%---------------------------------------------------------------------
%% Sinchronization

% Compute relative orientations 
X = cell2mat(G_gt(:));
Y = cell2mat(cellfun(@inv,G_gt(:)','uni',0));
Z = X * Y;

% random adjacency matrix
A = rand(n_imm) < density;
A = triu(A,1) + triu(A,1)' + diag(ones(1,n_imm));
figure, spy(A), xlabel(''); title('Adiacenza SFM');

% make holes in Z
Z = Z.*kron(A,ones(4));

% controllo che resti connesso
if any(A^n_imm==0)
    error('grafo non connesso, sincronizzazione impossibile')
end
% ... e paralell-rigid
if ~ParallelRigidityTest(A,3)
    warning('grafo non p.rigido, localizzazione da bearings impossibile')
end

%---------------------------------------------------------------------
%% Rotation synch

% estraggo le rotazioni da Z
Z_rot = Z;
Z_rot(4:4:end,:)=[]; Z_rot(:,4:4:end)=[];
R = rotation_synch(Z_rot,A);

%---------------------------------------------------------------------
%% Translation synch

%  estraggo bearings da Z e calcola baselines applicando le R ad U
U = extract_bearings(Z,R) ;
B = adj2inc(A);
C = translation_synch(U,B);

% calcola errore in SE(3) (totation+translation)
err =0;
for i=1:n_imm
    % altrimenti erano locations
    Gi=[R{i},-R{i}*C{i}; 0 0 0 1];
    err = err + normF(Gi*G_gt{1} - G_gt{i}) ;
end

fprintf('SE3 Synchronization error:\t %0.5g \n',err/n_imm);

%---------------------------------------------------------------------
%% Localization from bearings

% dimentica norma
for k=1:size(U,2)
    U(:,k) = U(:,k)/norm(U(:,k));
end

C = cop_from_bearings(U,B);

% [R,t,s] = opa(reshape(cell2mat(C_gt'),3,[]),reshape(cell2mat(C'),3,[]));
R = G_gt{1}(1:3,1:3)';
s = norm(C_gt{2}-C_gt{1})/norm(R*C{2});
t = C_gt{1}/s;

% calcolo errore
err =0;
for i = 1:n_imm
    err = err + norm(C_gt{i} - s*(R*C{i}+t));
end

fprintf('Localization error:\t\t %0.5g \n',err/n_imm);


%--------------------------------------------------------------
%% Autocalibrazione

% estraggo F da Z
F=cell(1,1);
for j=1:size(A,1)
    for i=j+1:size(A,2)
        if A(i,j) > 0 
             F{i,j} = inv(K)'*skew(Z(4*i-3:4*i-1,4*j))*Z(4*i-3:4*i-1,4*j-3:4*j-1)*inv(K);
        end
    end
end
        
K0 = K + 20*randn(3,3); K0(3,3) = 1; 
K_out = autocal(F,K0)

fprintf('Autocalibration %% error:\t %0.5g \n',100*abs(K_out(1,1)-K(1,1))/abs(K(1,1))) ;

%--------------------------------------------------------------
%% Sincronizzazione omografie

Hgt=cell(1,n_imm);
for i=1:n_imm
    Hgt{i} = randn(3);
    Hgt{i} = Hgt{i}./nthroot(det(Hgt{i}),3);
end

X = cell2mat(Hgt(:));
Y= cell2mat(cellfun(@inv,Hgt(:)','uni',0));
Z = X * Y;

% random adjacency matrix
A = rand(n_imm) < density;
A = triu(A,1) + triu(A,1)' + diag(ones(1,n_imm));

% make holes
Z = Z.*kron(A,ones(3));
figure, spy(A), xlabel(''); title('Adiacenza omografie');

% controllo che resti conneso;
if any(A^n_imm==0)
    error('grafo non connesso, sincronizzazione impossibile')
end

H = hom_synch(Z,A);

err =0;
for i=1:n_imm
    err = err + normF(H{i}*Hgt{1} - Hgt{i}) ;
end

fprintf('H Synchronization error:\t %0.5g \n',err/n_imm);





