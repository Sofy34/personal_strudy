function cost= cipolla_cost2(params)

% F(:,:,i,j) is the 3x3 fundamental matrix between views i and j
% and is global
% params is the a 5 vector containing the intrinsic parameters
% of all the cameras

% %%% CONSTANT PARAMETERS 

global Fs
global Ws

n = size(Fs,3);

Aj = [ params(1)  0  params(2)
  0 params(3) params(4)
  0 0 1];

Ai = [ params(1)  0 params(2)
  0 params(3) params(4)
  0 0 1];

cost = 0.0;

for i=1:n
  for j=i+1:n
    
    
    E = Aj'* Fs(:,:,i,j) * Ai;
    
    [U,S,V] = svd(Aj'* Fs(:,:,i,j) * Ai);
    
    cost = cost + Ws(i,j) * (S(1,1)-S(2,2))/(S(1,1)+S(2,2));
    
  end
end





