clc; clear; close all;

% Вхідні дані XOR
P = [0 0 1 1;
    0 1 0 1];

% Цільові значення
T = [0 1 1 0];

% Створення нейромережі
net = feedforwardnet(2, 'trainlm');

% Параметри навчання
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-5;

% Навчання мережі
net = train(net, P, T);

% Тестування
Y = net(P);

fprintf('Результати мережі:\n');
disp(Y);

fprintf('Округлені результати:\n');
disp(round(Y));