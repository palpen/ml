function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% create matrix of hypothesis by taking matrix product of the covariates matrix (with 1s for constant term) and the parameter vector
Xt = X * theta;

% create residual matrix
residual = Xt - y;

% sum squared residual (transpose residual so matrix dimensions agee)
sum_sqr_resid = (residual.') * (residual);

% calcualte cost function
J = (1 / (2 * m)) * sum_sqr_resid

end
