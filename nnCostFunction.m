function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% This function returns the cost and partial derivates for a 2 layer neural network

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for the 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
       
% Initialize values to be returned
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Compute the cost by feedforward method

% Add bias input to layer 1
a1 = [ones(m,1) X];
z2 = a1*Theta1';

% Add bias to layer 2 and compute sigmoid of weighted sum
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2*Theta2';
a3 = sigmoid(z3);

% Prepare new_y matrix for error calcualtion
new_y = zeros(m,num_labels);
for i = 1:m
  new_y(i, y(i)) = 1;
end

% Cost function
J1 = (-1/m)*sum((sum(new_y.*log(a3)+((1-new_y).*(log(1-a3))))));
J2 = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+(sum(sum(Theta2(:,2:end).^2))));
J = J1 + J2;

% Back propagation
a_1_bp = a1;
a_2_bp = sigmoid(a_1_bp*Theta1');
a_2_bp = [ones(m,1) a_2_bp];
a_3_bp = sigmoid(a_2_bp*Theta2');
delta_3 = a_3_bp - new_y;
delta_2 = delta_3*Theta2 .* sigmoidGradient([ones(m,1) z2]);
accumulated_delta1 = (delta_2(:,2:end)' * a_1_bp);
accumulated_delta2 = (delta_3' * a_2_bp);
p1 = lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)];
p2 = lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)];
Theta1_grad = (accumulated_delta1/m) + p1/m;
Theta2_grad = (accumulated_delta2/m) + p2/m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
