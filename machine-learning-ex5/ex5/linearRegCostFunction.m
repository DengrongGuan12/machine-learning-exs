function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
sizeOfTheta = size(theta,1);
h = X * theta;
J = sum((h - y) .^ 2)/(2 * m) + sum(theta(2:sizeOfTheta) .^ 2) * lambda/(2*m);
theta1 = [0;theta(2:sizeOfTheta,1)];
grad = (1/m) * X' * (h - y) + (lambda/m) * theta1; 
% You need to return the following variables correctly 
% sig = sigmoid(X * theta);
% theta(1) = 0;
% J = -(1/m)*(y' * log(sig) + (1 - y)' * log(1 - sig)) + sum(theta .^ 2) * lambda/(2 * m);
% grad = (1/m) * X' * (sig - y) + (lambda/m) * theta;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
