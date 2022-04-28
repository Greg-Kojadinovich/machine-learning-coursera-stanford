function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%adding bias layer to inputs 
X = [ones(m,1), X];

%Recoding labels as vectors of only 0 and 1 for outputs y 
K = num_labels;
eye_matrix = eye(K);
yk = eye_matrix(y,:);

%% Feedforward

%Input layer
a1 = X';
%Hidden Layer 
z2 = Theta1*a1;
a2 = [ones(m,1), sigmoid(z2)'];
%Output layer 
z3 = Theta2*a2';
a3 = sigmoid(z3);

%Sending hypothesis to htheta variable so math looks familiar
h_theta = a3;

%Calcualte Cost Function J - Unreguilarized
%J = (1/m) * sum(sum(-yk'.*log(h_theta) - (1-yk').*log(1-h_theta)));

%Removing the bias from Theta1 and Theta2 since they should not be included in regularization 

Theta1_reg = Theta1(:,2:end);
Theta2_reg = Theta2(:,2:end);

%Calcualte Cost Function J - With regularization
J = (1/m) * sum(sum(-yk'.*log(h_theta) - (1-yk').*log(1-h_theta))) + lambda/(2*m)*( sum(sum(Theta1_reg.^2)) + sum(sum(Theta2_reg.^2)));

%% BackProp

%for each output unit k in layer 3 (output layer) 
delta3 = h_theta'-yk;

%finding delta for hidden layer 2

t = Theta2' * delta3' ;
delta2 = t(2:end,:).*sigmoidGradient(z2);

Delta2 = delta3' * a2;
Delta1 = delta2 * a1';

Theta2_grad = (1/m) * Delta2;
Theta1_grad = (1/m) * Delta1;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
