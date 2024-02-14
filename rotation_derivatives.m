clear; clc

syms a b g real
syms x1 y1 z1 x2 y2 z2 x3 y3 z3 u v w real

P1 = [x1; y1];
P2 = [x2; y2];

eval_var = [x1, y1, x2, y2];
eval_sub = [-1, -1, 1, -1]/2;

% Rotation matrices
R = [cos(a) -sin(a); sin(a) cos(a)];
Rx = [1, 0, 0; 0 cos(a) -sin(a); 0 sin(a) cos(a)];
Ry = [cos(b) 0 sin(b); 0 1 0; -sin(b) 0 cos(b)];
Rz = [cos(g) -sin(g) 0; sin(g) cos(g) 0; 0 0 1];

% construct total rotation matrix
Rxyz = Rz * Ry * Rx;
Rxy = Ry * Rx;

% rotation matrix derivatives
dR = diff(R, a);
dRda = diff(Rxy, a);
dRdb = diff(Rxy, b);

%% 2D - fixed projection distance
k = [0; v];

% rotate points and take derivative
p1 = R*P1;
p2 = R*P2;
dp1 = dR*P1;
dp2 = dR*P2;
dk = dR*(R'*k);

% calculate area
A1 = (p2(1)-p1(1))*(p2(2)-p1(2))/2;
A2 = (p2(1)-p1(1))*(p1(2)-w);
A = A1+A2;

A_eval = subs(A, eval_var, eval_sub);

% caculate area derivative
dA1_ = diff(A1, a);
dA1 = (dp2(1)-dp1(1))*(p2(2)-p1(2))/2 + (p2(1)-p1(1))*(dp2(2)-dp1(2))/2;
dA2_ = diff(A2, a);
dA2 = (dp2(1)-dp1(1))*(p1(2)-w) + (p2(1)-p1(1))*dp1(2);
assert (dA1_ == dA1);
assert (dA2_ == dA2);

dA = dA1 + dA2;

%% 2D - projection to lowest point

k = [u; v];

% rotate points and take derivative
p1 = R*P1;
p2 = R*P2;
dp1 = dR*P1;
dp2 = dR*P2;
dk = dR*(R'*k);

% calculate area
A1 = (p2(1)-p1(1))*(p2(2)-p1(2))/2;
A2 = (p2(1)-p1(1))*(p1(2)-proj(2));
A = A1+A2;

% caculate area derivative
dA1 = (dp2(1)-dp1(1))*(p2(2)-p1(2))/2 + (p2(1)-p1(1))*(dp2(2)-dp1(2))/2;
dA2 = (dp2(1)-dp1(1))*(p1(2)-proj(2)) + (p2(1)-p1(1))*(dp1(2)-dproj(2));
dA = dA1 + dA2;

%% 3D - single triangle

% rotate arbitrary points
P1 = [x1; y1; z1];
P2 = [x2; y2; z2];
P3 = [x3; y3; z3];

P1_ = Rxy * P1;
P2_ = Rxy * P2;
P3_ = Rxy * P3;

k = [u; v; w];
k0 = Rxy'*k;
dk_da = simplify(dRda*k0)
dk_db = simplify(dRdb*k0)

% A0 = compute_A(P1_, P2_, P3_);
% h = (P1_(3) + P2_(3) + P3_(3))/3 + 1.4142135381698608;
% 
% % compute derivatives of individual rotation matrices
% dRda = Ry*diff(Rx, a);
% dRdb = diff(Ry, b)*Rx;
% dP1 = dRda*P1;
% dP2 = dRda*P2;
% dP3 = dRda*P3;
% 
% p1_eval = [-1, 2, -1];
% p2_eval = [-1, -2, -1];
% p3_eval = [1, -2, -1];
% ev = [deg2rad(28), 0];
% 
% dP1_ = eval(subs(dP1, [x1, y1, z1, a, b], [p1_eval, ev]));
% dP2_ = eval(subs(dP2, [x2, y2, z2, a, b], [p2_eval, ev]));
% dP3_ = eval(subs(dP3, [x3, y3, z3, a, b], [p3_eval, ev]));
% 
% % % % compute derivative components
% % % t1 = 0.5*(P2_(2) - P3_(2))*dP1(1);
% % % t2 = 0.5*(-P1_(2) + P3_(2))*dP2(1);
% % % t3 = 0.5*(-P2_(2) + P1_(2))*dP3(1);
% % % t4 = 0.5*(-P2_(1) + P3_(1))*dP1(2);
% % % t5 = 0.5*(P1_(1) - P3_(1))*dP2(2);
% % % t6 = 0.5*(P2_(1) - P1_(1))*dP3(2);
% % % dAda = (t1+t2+t3+t4+t5+t6);
% % % dhda = 1/3*(dP1(3) + dP2(3) + dP3(3));
% 
% dVda = A*dhda + h*dAda;
% 
% dAda_ = eval(subs(dAda, [P1; P2; P3; a; b], [p1_eval, p2_eval, p3_eval, ev]'))
% dhda_ = eval(subs(dhda, [P1; P2; P3; a; b], [p1_eval, p2_eval, p3_eval, ev]'))
% dVda_ = eval(subs(dVda, [P1; P2; P3; a; b], [p1_eval, p2_eval, p3_eval, ev]'))
% 
% function s = S(v)
%     [x, y, z] = deal(v(1), v(2), v(3));
%     s = [0 -z y; z 0 -x; -y x 0];
% end
% 
% function a = compute_A(p1, p2, p3)
%     a = det([p1(1:2), p2(1:2)]) + ...
%         det([p2(1:2), p3(1:2)]) + ...
%         det([p3(1:2), p1(1:2)]);
%     a = abs(a/2);
% end