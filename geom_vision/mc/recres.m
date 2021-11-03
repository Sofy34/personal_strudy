function r = recres(struct,PPM,imagePoints)
% 

dim = length(struct);

x = struct(1:dim/3)';
y = struct(dim/3+1:2*dim/3)';
z = struct(2*dim/3+1:dim)';


numberOfPoints = size(imagePoints,1);
numberOfViews =  size(imagePoints,3);

r=0;

for view = 1:numberOfViews
 
  [u v] = projf(PPM(:,:,view),x,y,z);
  r = r +  sqrt((sum((u-imagePoints(:,1,view) ).^2 + (v-imagePoints(:,2,view)).^2))/(numberOfPoints-1));

end

r = r / numberOfViews;


