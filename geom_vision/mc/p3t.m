function [xp,yp,zp] = p3t(T,x,y,z)
% P3T apply a Projective 3D Transform

tmp = T * [x, y, z, ones(size(x,1),1)]' ; 
xp = (tmp(1,:) ./tmp(4,:))';
yp = (tmp(2,:) ./tmp(4,:))';
zp = (tmp(3,:) ./tmp(4,:))';

