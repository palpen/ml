function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% sigmoid function
h = sigmoid(X * theta);

% cost function for regularized logistic regression
J = (1 / m) * sum(-y.*log(h) - (1 - y).*log(1 - h)) + ...
    (lambda / (2 * m)) * sum(theta(2:end,:).^2);

% gradient vector (note: size(X)=mxn; size(h-y)=mx1)
% note omission of first row containing the data for theta_0 (which shouldn't be regularized)
X_theta0 = X(:, 1); %mx1
grad(1) = (1 / m) * (X_theta0' * (h - y));

X_notheta0 = X(:, 2:end); %mx(n-1)
grad(2:end) = (1 / m) * ((X_notheta0' * (h - y)) + lambda * theta(2:end));

end
