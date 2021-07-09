function [r,phi] = ikan(R)
% IKAN: compute the axis and angle of a rotation matrix

phi = acos((trace(R)-1)/2);
r0 = [R(3,2)-R(2,3), R(1,3)-R(3,1), R(2,1)-R(1,2)]';
r = r0/norm(r0);

% this is not well defined for small angles
% according to Jeff, if R-R'=0, this does not work --> CHECK
