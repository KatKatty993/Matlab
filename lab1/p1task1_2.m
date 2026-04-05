clear; clc; close all; 
% очищення робочої області, консолі та закриття всіх графіків

X = [-0.3 -2.0 -2.0 -0.1; 
      1.5  1.0 -2.0 -1.0];

T = [0 1 1 0];

net = perceptron;

% Встановлення правила навчання
net.inputWeights{1,1}.learnFcn = 'learnpn';
net.biases{1}.learnFcn = 'learnpn';

net = train(net,X,T);

Y = net(X);

% Побудова графіка точок
figure;
plotpv(X,T); hold on;

% Побудова розділяючої прямої
plotpc(net.IW{1,1},net.b{1});

title('Task 1 - Variant 2');
grid on;