function [xp,yp,zp] = rec3d(PPM,imagePoints)
%  REC3D compute 3D reconstruction by triangulation from a set of views
%    Given a set of projection matrices (PPM) and a set of corresponding
%    image points (the position gives the correspondence) the 3D position of
%    the 3D points that projects onto the image points is computed. Since
%    rays may not intersect, the mid-point of the common perpendicular to
%    the two rays is chosen (the mid-point method)


%    Algorithm: Beardsley, Ziserman, Murray IJCV 23(3),1997
%
%    Author: A. Fusiello 1999


numberOfPoints = size(imagePoints,1);
numberOfViews =  size(imagePoints,3);
struc = [];
I = eye(3,3);

for i = 1:numberOfPoints
  
  A =  zeros(3,3);
  b =  zeros(3,1);
  
  for view = 1:numberOfViews
    
    Q =  inv(PPM(:,1:3,view));
    c =  -Q*PPM(:,4,view);
    d =   Q*[imagePoints(i,:,view), 1]';
    d  = d/norm(d);
    
    A = A + I - d*d';
    b = b + c - (c'*d)*d;
    
  end
  
  w = inv(A) * b;
  struc = [struc  w];
  
  xp = struc(1,:)';
  yp = struc(2,:)';
  zp = struc(3,:)';
  
end









