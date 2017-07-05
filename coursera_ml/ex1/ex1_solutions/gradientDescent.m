function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % save corresponding theta to a scalar (note matlab indexing starts at 1, not 0)
    theta0 = theta(1);
    theta1 = theta(2);

    % derivative of cost function wrt each feature j
    J_deriv0 = (X * theta - y).*X(:, 1);
    J_deriv1 = (X * theta - y).*X(:, 2);

    % simultaneously update all theta values before next iteration
    theta0 = theta0 - alpha * (1 / m) * sum(J_deriv0);
    theta1 = theta1 - alpha * (1 / m) * sum(J_deriv1);

    % save update theta values to theta vector
    theta = [theta0;theta1];

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

    % more elegant solution from https://github.com/anirudhjayaraman/Machine-Learning
    % theta = theta - (alpha/m)*(X')*(X*theta - y);

end

end
