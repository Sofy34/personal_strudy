function P =  rcam(A,lookp,ad,sd)
% RCAM generate a random camera
%    generate a random camera pointing to lookp, positioned at an average 
%    distance ad form the origin, with a std dev of sd 
%    A is the intrinsic parameters matrix


eyep = rand(1,3)-0.5;
eyep = eyep/norm(eyep) * (ad + sd*randn);
R(3,:) = lookp - eyep/norm(lookp - eyep);
R(2,:) = cross(R(3,:),rand(1,3));
R(2,:) = R(2,:)/norm(R(2,:));
R(1,:) = cross(R(2,:),R(3,:));
P = A*[R, -R*eyep'];    


% plot camera (ONLY FOR DEBUG)
figure(2)
% figure number must be the same where 3d point are plotted and holded
a = (lookp+eyep)/2;
b = [R(1,:)+eyep;R(2,:)+eyep;R(3,:)+eyep]/2;
plot3([a(1,1), b(3,1)],[a(1,2), b(3,2)],[a(1,3), b(3,3)],'-c');
plot3([a(1,1), b(2,1)],[a(1,2), b(2,2)],[a(1,3), b(2,3)],'-c');
plot3([a(1,1), b(1,1)],[a(1,2), b(1,2)],[a(1,3), b(1,3)],'-c');
axis equal
rotate3d on


