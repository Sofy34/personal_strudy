% this script is the driver for self-calibration experiments; taking away
% the computation of the intrinsic parameters, it can be used to check the
% computation of camera motion from correspondences (relative orientation)
% in motion.m or the reconstruction (triangulation) algorithm in rec3d.m

% clear;

% number of points
numberOfPoints = 50

% number of trials to be averaged ( >10 )
numberOftrials = 70

% number of views (or cameras)
numberOfViews = 5

% std dev of gaussian noise added to image
sigma =1.0;

% perturbation of initial value
% pert = 0.0;

aut = [];
avt = [];
u0t = [];
v0t = [];
nt  = [];
ptmean  = [];
rtmean  = [];

for pert = 0:10:1
% for numberOfViews = 3:2:15
% for sigma = 0:0.5:4

  au = [];
  av = [];
  u0 = [];
  v0 = [];
  nconv =0;
  pix_err = []; 
  rec_err = [];
  
  for trial =1:numberOftrials

    %  set of fundamental matrices
    global Fs;
    Fs = zeros(3,3,numberOfViews,numberOfViews);

    global Ws;
    Ws = zeros(numberOfViews,numberOfViews);

    imagePoints = zeros(numberOfPoints,2,numberOfViews);

    PPM = zeros(3,4,numberOfViews);
    PPMGT = zeros(3,4,numberOfViews);

    % punti random in un cubo di lato 0.5
    % che e' conenuto in una sfera di raggio sqrt(3)/2
    data = (rand(numberOfPoints,3)-0.5)./(sqrt(3)/2);

    % plot 3D points
    x = data(:,1);
    y = data(:,2);
    z = data(:,3);
    figure(2)
    plot3(x,y,z,'.');
    hold on

    % camera intrinsic parameters, constant  
    A = [
      800 0   256
      0	  800 256
      0	  0     1];

    % random camera position (first camera) 
    P = rcam(A, [0 0 0], 2.5, 0.25);
    
    G0 = [inv(A) * P; ; 0 0 0 1];
    % G0 e' la trasformazione dei punti dal rif mondo al rif della prima tc 
    
    % applica G0 ai punti 3D
    [x,y,z] = p3t(G0,x,y,z);

    % con i punti espressi in qto sistema, la prima tc e' I|0
    P = A * [eye(3), zeros(3,1)];
    
    % project world points to image points
    [u,v]= projf(P,x,y,z);
    
    % add noise to image points
    u = u + sigma * randn(size(u));
    v = v + sigma * randn(size(v));
    
    imagePoints(:,:,1) = [u,v];
    PPMGT(:,:,1) = P;
    
    % generazione viste successive
    
    for view=2:numberOfViews
      
      % random camera position
      P = rcam(A, [0 0 0], 2.5, 0.25);
            
      % apply world coordinate transformation
      P = P * inv(G0);
      
      % project world points to image points
      [u,v]= projf(P,x,y,z);
      
      % add noise to image points
      u = u + sigma * randn(size(u));
      v = v + sigma * randn(size(v));
      
      imagePoints(:,:,view) = [u,v];
      PPMGT(:,:,view) = P;
      
      % t12 is needed to fix the scale 
      if view == 2
	t12 = inv(A) * P(:,4);
      end
      
      % plot points 
      figure(1)      
      plot(u,v,'+g');
      axis equal;
      % pause
      
    end
    figure(2)
    hold off
    
    % In the real case plug the results of the tracker here in imagePoints
    

    % calcolo le n*(n-1)/2  matrici fondamentali di tutte le coppie di viste

    for i=1:numberOfViews
      for j=i+1:numberOfViews
	
	% calcola la matrice fondamentale ij 
	Fs(:,:,i,j) = ...
	    fm(imagePoints(:,1,i),imagePoints(:,2,i),imagePoints(:,1,j),imagePoints(:,2,j)); 
	
	% check fundamental matrix by computing residuals of m_j * F_ij * m_i = 0
	err = 0;
	for k=1:numberOfPoints
	  err =  err + norm([imagePoints(k,1,j), imagePoints(k,2,j), 1]  * Fs(:,:,i,j) * ...
	      [imagePoints(k,1,i), imagePoints(k,2,i), 1]');
	end
	err =  err/numberOfPoints;
	disp(sprintf('Fundamental matrix %d %d residual: %0.5g',i,j,err));
	
	Ws(i,j) = 1/err;
	
      end
      
    end

    Ws = Ws./sum(sum(Ws));

    % find intrinsic parameters by minimizing residuals with Nelder-Meads
    % the fundamntals matrices are globals

  
    a_true =   [A(1,1), A(1,3), A(2,2), A(2,3)]
    a_conv = fmins('cipolla_cost2',a_true)
    
    a0 = a_true + pert * a_true.*(rand(1,4)-0.0)
    
    % a0 = rand(1,4)*2000
    
    disp(sprintf('initial cost: %0.5g',norm(cipolla_cost2(a0))))
    a = fmins('cipolla_cost2',a0)
    disp(sprintf('solution cost: %0.5g',norm(cipolla_cost2(a))))
    
    if norm(a_conv - a)/norm(a_conv) < 0.1
      nconv = nconv + 1;
    else
      % error('self-calibration did not converged');
      %  disp('self-calibration did not converged');
    end
    
    % this is the resulting Intrinsic parameters mateix 
    
     A = [
       a(1)	0     a(2)
       0	a(3)  a(4)
       0	0   1];
 
     au = [au a(1)];
     av = [av a(3)];
     u0 = [u0 a(2)];
     v0 = [v0 a(4)];
 
 
     % first view
     PPM(:,:,1) = A *[ 
       1     0     0     0
       0     1     0     0
       0     0     1     0 ];
 
 
     % second view
     [R,t] = motion(Fs(:,:,1,2),A,imagePoints(:,1,1),imagePoints(:,2,1),imagePoints(:,1,2),imagePoints(:,2,2));
 
     % t12 comes from the groung truth; it fixes the scale factor
     t = t/norm(t)*norm(t12);
 
     PPM(:,:,2) = A * [R   t];
     t12 = t;
 
     % any view > 3
     for i=3:numberOfViews
       
       % fattorizzazione RS  
       [R,t]     = motion(Fs(:,:,1,i),A,imagePoints(:,1,1),imagePoints(:,2,1),imagePoints(:,1,i),imagePoints(:,2,i));  
       [R2i,t2i] = motion(Fs(:,:,2,i),A,imagePoints(:,1,2),imagePoints(:,2,2),imagePoints(:,1,i),imagePoints(:,2,i));
       
       % normalization factor to obtain coherent projection matrices
       % Explanation: see Luong & Faugeras IJCV 33(3), 1997 pg 32
       % Algorithm: see Zeller & Faugeras, INRIA RR 2793, sec. 6
       k = (norm(cross(R2i*t12,t2i))^2)/(cross(t,t2i)' * cross(R2i*t12,t2i));
       
       % camera matrix 
       PPM(:,:,i) = A * [R, k*t];
       
     end
     
     % at this point, the reconstructed camera matrices are in PPM     
     [xr,yr,zr]=rec3d(PPM,imagePoints);
     
     % bundle adjustment
     % tolto perche' troppo lungo. Forse con fminu va meglio
     % $$$     disp('refining reconstruction');
     % $$$     struct=[xr',yr',zr'];    
     % $$$     struct = fmins('recres',struct,[],[],PPM,imagePoints);
     % $$$     dim = length(struct);
     % $$$     xr = struct(1:dim/3)';
     % $$$     yr = struct(dim/3+1:2*dim/3)';
     % $$$     zr = struct(2*dim/3+1:dim)';
 
     % RMS 3D error on euclidean reconstruction
     eres= sqrt((sum((x-xr).^2 + (y-yr).^2 + (z-zr).^2))/(numberOfPoints-1));      
     disp(sprintf('RMS error on euclidean reconstruction: %0.5g',eres));
     rec_err = [rec_err, eres];
     
     % average RMS pixel error *** viene troppo elevato: perche? ***
     pres = rpe(PPM,imagePoints,xr,yr,zr);
     disp(sprintf('average residual RMS pixels error: %0.5g',pres));
     pix_err = [pix_err, pres];
     
     % visualizzazione proiezione 2D (solo l'ultima)
     i=numberOfViews;
     figure(3) 
     [u,v]= projf(PPM(:,:,i),xr,yr,zr);
     plot(u,v,'or');
     hold on
     plot(imagePoints(:,1,i),imagePoints(:,2,i),'+g')
     axis equal;
     hold off
     
     % visualize 3D sructure with the right scele
     figure(4)
     plot3(xr,yr,zr,'or');
     hold on
     plot3(x,y,z,'+g');
     axis equal;
     hold off
     
     if eres > 1.0 | pres > 100
       disp('3D reconstruction may be wrong');
       pause
     end 
    
    % end of trial
  end
  
  % COMPUTING AND STORING ERRORS
    
   aut = [aut rmse(au, A(1,1)*ones(size(au)))];
   avt = [avt rmse(av, A(2,2)*ones(size(av)))];
   u0t = [u0t rmse(u0, A(1,3)*ones(size(u0)))];
   v0t = [v0t rmse(v0, A(2,3)*ones(size(v0)))];
        nt = [nt nconv];
   
   % trimmed mean (...ish): discard  last ten
   rec_err = sort(rec_err);
   rec_err = rec_err(1:length(rec_err)-10);
   pix_err = sort(pix_err);
   pix_err = pix_err(1:length(pix_err)-10);
   rtmean = [rtmean mean(rec_err)];
   ptmean = [ptmean mean(pix_err)];
   
  % end of outer cycle (on sigma or views)  
end

save aut
save avt
save u0t
save v0t
save nt
save ptmean
save rtmean

A = [
  800 0   256
  0	  800 256
  0	  0     1];

v =  0:0.5:4;


figure(5)
pl1 = plot( v,aut./A(1,1),'x--') ;
hold on
pl2 = plot( v,avt./A(2,2),'s:') ;
pl3 = plot( v,u0t./A(1,3),'+-.') ;
pl4 = plot( v,v0t./A(2,3),'o-'); 
hold off

xlabel('image noise (pixels)') ;     
ylabel('relative RMS error')  ;
title('5 views');
legend([pl1,pl2,pl3,pl4],'au','av','u0','v0');

figure(6)
plot( v,ptmean,'o-')
ylabel('RMS pixel error')  ;  
xlabel('image noise (pixels)') ;   
title('5 views');

figure(7)
plot( v,rtmean,'o-')
ylabel('reconstruction error');
xlabel('image noise (pixels)') ;   
title('5 views');

figure(8)
bar(100*[0:10:1],100*nt/100,'y')
ylabel('percentage of convergence');
xlabel('percentage perturbation of initial value');
title('5 views, 1.0 pixel image noise');

 







 