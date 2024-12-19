clear all

% Create terrain
xLimits         = [900 1200]; % x-axis limits of terrain (m)
yLimits         = [-200 200]; % y-axis limits of terrain (m)
roughnessFactor = 1.75;       % Roughness factor
initialHgt      = 0;          % Initial height (m)
initialPerturb  = 200;        % Overall height of map (m) 
numIter         = 8;          % Number of iterations


minX = xLimits(1);
maxX = xLimits(2);
minY = yLimits(1);
maxY = yLimits(2);
initialHeight = initialHgt;
f = roughnessFactor;

% Generate random terrain
dX = (maxX-minX)/2;
dY = (maxY-minY)/2;
[x,y] = meshgrid(minX:dX:maxX,minY:dY:maxY);
terrain = ones(3,3)*initialHeight;
perturb = initialPerturb;
for ii = 2:numIter
    perturb = perturb/f;
    oldX = x;
    oldY = y;
    dX = (maxX-minX)/2^ii;
    dY = (maxY-minY)/2^ii;
    [x,y] = meshgrid(minX:dX:maxX,minY:dY:maxY);
    terrain = griddata(oldX,oldY,terrain,x,y);
    terrain = terrain + perturb*random('norm',0,1,1+2^ii,1+2^ii);
    terrain(terrain < 0) = 0; 
end

A = terrain;

A(A < 0) = 0; % Fill-in areas below 0
xvec = x(1,:); 
yvec = y(:,1);
resMapX = mean(diff(xvec));
resMapY = mean(diff(yvec));


A_2 = zeros(257);
xvec_2 = xvec;
yvec_2 = yvec;

for i = 1:257
    A_2(i,i) = 200;
    A_2(i,258-i) = 200;
    for a = 1:10
        if i-a > 0
            A_2(i-a,i) = 200-(20*a);
            A_2(i,i-a) = 200-(20*a);
        end
        if i+a < 258
            A_2(i+a,i) = 200-(20*a);
            A_2(i,i+a) = 200-(20*a);
        end
        if 258-i-a > 0 && i-a > 0
            A_2(i-a,258-i) = 200-(20*a);
            A_2(i,258-i-a) = 200-(20*a);
        end
        if 258-i+a < 257 && i+a < 258
            A_2(i+a,258-i) = 200-(20*a);
            A_2(i,258-i+a) = 200-(20*a);
        end
    end
end

    helperPlotSimulatedTerrain(xvec_2,yvec_2,A_2)
helperPlotSimulatedTerrain(xvec,yvec,A)



function helperPlotSimulatedTerrain(xvec,yvec,A)
% Plot simulated terrain

figure()
hS = surf(xvec,yvec,A);
hS.EdgeColor = 'none';
hC = colorbar;
hC.Label.String = 'Elevation (m)';
landmap = landColormap(64);
colormap(landmap); 
xlabel('X (m)')
ylabel('Y (m)')
axis equal;
title('Simulated Terrain')
view([78 78])
drawnow
pause(0.25)
end

function cmap = landColormap(n)
%landColormap Colormap for land surfaces
% cmap = landColormap(n)
%
% Inputs: 
%    - n     = Number of samples in colormap
%
% Output: 
%    - cmap  = n-by-3 colormap

c = hsv2rgb([5/12 1 0.4; 0.25 0.2 1; 5/72 1 0.4]);
cmap = zeros(n,3);
cmap(:,1) = interp1(1:3,c(:,1),linspace(1,3,n)); 
cmap(:,2) = interp1(1:3,c(:,2),linspace(1,3,n));
cmap(:,3) = interp1(1:3,c(:,3),linspace(1,3,n)); 
colormap(cmap);
end

