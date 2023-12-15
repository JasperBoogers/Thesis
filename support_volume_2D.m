clear; clc; close

syms x1 y1 x2 y2 x3 y3 x4 y4 theta real

%% define points
p1 = [x1; y1];
p2 = [x2; y2];
p3 = [x3; y3];
p4 = [x4; y4];
points = [p1, p2, p3, p4];
points_val = [-1, -1, 1, -1, 1, 1, -1, 1];

plane = -4;

%% rotate and extract points
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];
points_ = R*points;
points_ = subs(points_, [x1, y1, x2, y2, x3, y3, x4, y4], points_val);

p1_ = points_(:,1);
p2_ = points_(:,2);
p3_ = points_(:,3);
p4_ = points_(:,4);



%% calculate support volumes
S1 = 0.5*( p1_(1)-p4_(1) )*( p4_(2)-p1_(2) );
S2 = ( p1_(1)-p4_(1) )*( p1_(2)-plane );
S3 = ( p2_(1)-p1_(1) )*( p1_(2)-plane );
S4 = 0.5*( p2_(1)-p1_(1) )*( p2_(2)-p1_(2) );

S = simplify(S1 + S2 + S3 + S4);
dSdt = diff(S, theta);

S_sub = subs(S, [x1, y1, x2, y2, x3, y3, x4, y4], points_val);
dSdt_sub = subs(dSdt, [x1, y1, x2, y2, x3, y3, x4, y4], points_val);

figure(1);
subplot(1, 2, 1);
fplot(S_sub, [0, pi/2]);
title('Support volume')
subplot(1, 2, 2);
fplot(dSdt_sub, [0, pi/2]);
title('Derivative of support volume')
