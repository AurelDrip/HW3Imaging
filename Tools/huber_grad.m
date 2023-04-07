function g=huber_grad(x,delta)
% compute the gradient of the Huber function with parameter delta
% x: vector of size N
% g: vector of size N
g = zeros(size(x));
g(x < -delta) = delta * sign(x(x<-delta));
g(x > delta) = delta * sign(x(x>delta));
g(abs(x) <= delta) = x((abs(x)<=delta));


end