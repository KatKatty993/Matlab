clear; clc; close all; 
% Жокеї
J = [160 154 162 168 170; 
     55  62  58  64  65];

% Баскетболісти
B = [185 193 200 195 171; 
     85  92  95  88  79];

% Об'єднання даних
Xlab = [J B];

% Мітки класів (0 – жокеї, 1 – баскетболісти)
Tlab = [zeros(1,5) ones(1,5)];

netA = perceptron;
netA.inputWeights{1,1}.learnFcn = 'learnpn';
netA.biases{1}.learnFcn = 'learnpn';
netA = train(netA,Xlab,Tlab);

Ylab = netA(Xlab);

figure;
plotpv(Xlab,Tlab); hold on;

% Розділяюча пряма
plotpc(netA.IW{1,1},netA.b{1});

title('Athletes classification (height/weight)');
grid on;
