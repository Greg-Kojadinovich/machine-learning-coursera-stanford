function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
htheta = sigmoid(X*theta);
%h [n x 1] X [n x 3] theta [3 x 1] n = 2

Pos_Cost = -y' * log(htheta);
% y' [1 x 100] htheta [n x 1]

Neg_Cost = (1-y') * log(1-htheta);

J = (1/m) * (Pos_Cost - Neg_Cost);

grad = (1/m) * (X' * (htheta - y));
%X' [3 x 100] ntheta [3 x 1] y [100 x 1]






% =============================================================

end
