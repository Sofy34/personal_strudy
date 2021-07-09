function [R,t] = motion(F,A,ul,vl,ur,vr)
%  MOTION SR factorization of the essential matrix
%    The essential matrix E is computed from A and F, and it is then 
%    factorized to yield the camera rotation R and the translation t,
%    between two views
%  
%    Algorithm: Hartley, ECCV'92
%
%    Author: A. Fusiello 1999

% uses p3t, rec3d

imagePoints = zeros(size(ul,1),2,2);
imagePoints(:,:,1) = [ul,vl];
imagePoints(:,:,2) = [ur,vr];

PPM = zeros(3,4,2);

% essential matrix
E = A'*F*A;

%  SVD factorization such that E = U*D*V'
[U,D,V]= svd(E); 

S1=[
  0 -1 0
  1 0 0 
  0 0 0];

R1=[
  0 1 0
  -1 0 0
  0 0 1];

%  left perspective projection matrix
PPM(:,:,1) = A *[ 
  1     0     0     0
  0     1     0     0
  0     0     1     0 ];

for j = 1:4
  
  %  skew symmetric matrix representing translation
  S= (-1)^j * U*S1*U';
  
  %  rotation matrix
  if j<=2   
    R = det(U*V') * U*R1*V';
  end
  
  if j>2   
    R = det(U*V') * U*R1'*V';
  end
  
  %  translation vector
  t=[S(3,2) S(1,3) S(2,1)]';
  t = t / norm(t);
  
  % 3D points distance from BOTH focal planes must be positive 
  PPM(:,:,2) = A * [R   t];
  [x,y,zL]=rec3d(PPM,imagePoints(1:8,:,:));
  G0 = [R,t; 0 0 0 1];
  [x,y,zR] = p3t(G0,x,y,zL);
  
  if min(zL)>= 0 & min(zR)>= 0
    break 
  end
  
end

if min(zL)<0 | min(zR)<0   
%  error('no way to get positive depths'); 
end;


% check whether R is a rotation
if abs(det(R)-1) > 0.000001 & norm(R*R'-eye(3,3)) > 0.000001
  det(R)  
  error('R is not a rotation matrix'); 
end;


% $$$  disp(E ./norm(E))
% $$$    
% $$$   
% $$$  E = [
% $$$       0 -t(3) t(2)
% $$$      t(3) 0 -t(1)
% $$$       -t(2) t(1) 0] * R;
% $$$   
% $$$  
% $$$  disp(E ./norm(E))
% $$$    










