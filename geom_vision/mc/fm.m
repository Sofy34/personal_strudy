function F = fm(u1,v1,u2,v2)
% FM compute fundametal matrix from point correspondence
%    at least 8 points are needed. Use data standardization and linear
%    algorithm by Longuett-Higgins and Hartley.

%    Algorithm: Hartley PAMI,1997
%
%    Author: A. Fusiello 1999 (adapted from original code by C. Plakas)


n_points = size(u1,1);
if n_points < 8
  error('8 points at least are needed')
end

% scale the point coordinates so that barycenter is in origin and average distance is sqr(2)
avgx1 = (sum(u1)/n_points);
avgy1 = (sum(v1)/n_points);
avgx2 = (sum(u2)/n_points);
avgy2 = (sum(v2)/n_points);
u1 = u1 - avgx1;
v1 = v1 - avgy1;
u2 = u2 - avgx2;
v2 = v2 - avgy2;
tscale1 = sum(sqrt(u1.^2 + v1.^2))/n_points/sqrt(2);
tscale2 = sum(sqrt(u2.^2 + v2.^2))/n_points/sqrt(2);
u1 = u1/tscale1;
v1 = v1/tscale1;
u2 = u2/tscale2;
v2 = v2/tscale2;
SM1 = [1/tscale1 0 -avgx1/tscale1; 0 1/tscale1 -avgy1/tscale1; 0 0 1];
SM2 = [1/tscale2 0 -avgx2/tscale2; 0 1/tscale2 -avgy2/tscale2; 0 0 1];
% end of scaling

% Preparing the equation matrix; must have at least 9 rows
A = zeros(n_points,9);
A(:,1) = u2 .* u1;
A(:,2) = u1 .* v2;
A(:,3) = u1;
A(:,4) = v1 .* u2;
A(:,5) = v1 .* v2;
A(:,6) = v1;
A(:,7) = u2;
A(:,8) = v2;
A(:,9) = 1;

% ready

[U D V] = svd(A);
% diag(D) is guaranteed to be ordered

% solution vector corresponding to the 
% least singular value
f = V(:,9)/norm(V(:,9));

% essential matrix from f vector
F = reshape(f,3,3);

% Enforce singularity constraint on F
[U D V] = svd(F);
D(3,3) = 0;
F = U * D * V';

% apply the inverse scaling
F = SM2' * F * SM1;               













