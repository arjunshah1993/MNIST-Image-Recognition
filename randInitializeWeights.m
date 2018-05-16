function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in

% Initialize weights matrix
W = zeros(L_out, 1 + L_in);

% Epsilon calculated as below
epsilon = sqrt(6)/sqrt(L_in+L_out);

W = rand(L_out, L_in+1)*2*epsilon-epsilon;

end
