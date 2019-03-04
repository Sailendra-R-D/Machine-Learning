% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
% First 10 examples from the dataset
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

% Scale features and set them to zero mean
[X, mu, sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];

% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Display gradient descent's result
fprintf('Theta computed from gradient descent:%f,%f\n\n',theta(1),theta(2))


% Plot the convergence graph
#plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
#xlabel('Number of iterations');
#ylabel('Cost J');

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================

price = [1650 3]; % Enter your price formula here
price = (price -mu)./sigma;
price = [ones(1,1) price];
price = price*theta;
% ============================================================

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $%f\n', price);


% Run gradient descent:
% Choose some alpha value
alpha = 1;
num_iters = 50;

% Init Theta and Run Gradient Descent 
theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

price = [1650 3]; % Enter your price formula here
price = (price -mu)./sigma;
price = [ones(1,1) price];
price = price*theta;

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): $%f\n', price);

% Plot the convergence graph
#plot(1:num_iters, J_history, '-b', 'LineWidth', 2);
#xlabel('Number of iterations');
#ylabel('Cost J');



theta = normalEqn(X,y);

price = [1650 3]; % Enter your price formula here
price = (price -mu)./sigma;
price = [ones(1,1) price];
price = price*theta;

fprintf('Predicted price of a 1650 sq-ft, 3 br house (using normal eq): $%f\n', price);
