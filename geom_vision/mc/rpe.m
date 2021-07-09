function r = rpe(PPM,imagePoints,x,y,z)
% RPE compute reconstruction RMS Pixel Error 

numberOfPoints = size(imagePoints,1);
numberOfViews =  size(imagePoints,3);

r=0;

for view = 1:numberOfViews
 
  [u v] = projf(PPM(:,:,view),x,y,z);
  r = r +  sqrt((sum((u-imagePoints(:,1,view) ).^2 + (v-imagePoints(:,2,view)).^2))/(numberOfPoints));

end

r = r / numberOfViews;


