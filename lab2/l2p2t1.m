clc; clear; close all;

P = [0 0 1 1;
     0 1 0 1];

T = [0 1 1 0];

% Створення мережі
net = feedforwardnet(2,'trainlm');

% Параметри навчання
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-5;

% Навчання
net = train(net,P,T);

% Тестування
Y = net(P);

fprintf('Результати роботи мережі:\n');
disp(Y);
