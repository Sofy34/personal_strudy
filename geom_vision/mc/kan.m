function R = kan(r,phi)
% KAN: compute a rotation matrix given the axis and angle

c = cos(phi);
s = sin(phi);
v = 1-cos(phi);

r=r/norm(r);

R = [ 	r(1)*r(1)*v+c,		r(1)*r(2)*v-r(3)*s,	r(1)*r(3)*v+r(2)*s
	r(1)*r(2)*v+r(3)*s,	r(2)*r(2)*v+c,		r(2)*r(3)*v-r(1)*s
	r(1)*r(3)*v-r(2)*s,	r(2)*r(3)*v+r(1)*s,	r(3)*r(3)*v+c  ];



