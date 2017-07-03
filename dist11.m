function n1 = dist11(X, m, n)
%DIST1	Calculates squared distance between two sets of points.

[d,N] = size(X);
n1 = zeros(N,N);

Xm = 2*(X(m,n)*ones(1,N)-X(m,:));
n1(n,:) = Xm;
n1(:,n) = Xm';

end