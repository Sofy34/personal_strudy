function r = rmse(x,y)
% RMSE    Standard deviation.
%   For vectors, RMSE(X,Y) returns the root mean square error of X wirh
%   respect to the true value Y.
%
%   RMSE(X,Y) normalizes by (N-1) where N is the sequence length.  This
%   makes RMSE(X,Y).^2 the best unbiased estimate of the variance if X
%   is a sample from a normal distribution centered in Y.
%

r = sqrt(sum((x - y).^2)/(length(x)-1));
