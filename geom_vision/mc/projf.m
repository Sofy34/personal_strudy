function [u,v] = projf(P,x,y,z)
% PROJ  compute perspective projection (from 3D to pixel coordinates)
%   pixel positions are returned with floating point precision
%
%   See also PROJE

c3d = [x,y,z];
h3d =[c3d ones(size(c3d,1),1)]';
h2d = P*h3d ;
c2d = h2d(1:2,:)./ [h2d(3,:)' h2d(3,:)']'; 

u = c2d(1,:)';
v = c2d(2,:)'; 
