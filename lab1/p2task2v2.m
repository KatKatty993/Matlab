clear; clc; close all;

X_black = [0,  -0.5,  0.5,  0.0, -0.3,  0.3;
           0,   0.5,  0.5,  0.8,  0.2,  0.2];

T_black = [1 1 1 1 1 1;
           1 1 1 1 1 1;
           1 1 1 1 1 1];

X_white = [0,  -2,   2,   1.5, -1.5,  0.0;
           2,  -1,  -1,   1.5,  1.5, -1.5];

T_white = [0 1 1 0 1 1;
           1 0 1 1 0 1;
           1 1 0 1 1 0];

% Об'єднуємо вибірку
X = [X_black, X_white];
T = [T_black, T_white];

% Створюємо та навчаємо персептрон (3 нейрони)
net = newp(X, T, 'hardlim', 'learnpn');
net.trainParam.epochs = 1000;
net = train(net, X, T);

% Тестування
test_point = [0; 0.5];
result = sim(net, test_point);

disp('Model testing:');
fprintf('Test point: (%.1f, %.1f)\n', test_point(1), test_point(2));
fprintf('Network output: [%d; %d; %d]\n', result(1), result(2), result(3));

if all(result == [1; 1; 1])
    disp('Class: BLACK point');
else
    disp('Class: WHITE point');
end;

figure;
plot(X_black(1,:), X_black(2,:), 'wo', ...
     'MarkerFaceColor', 'none', 'MarkerSize', 10, 'LineWidth', 1.5);
hold on;
plot(X_white(1,:), X_white(2,:), 'ko', ...
     'MarkerFaceColor', 'w', 'MarkerSize', 10, 'LineWidth', 1.5);
plotpc(net.IW{1,1}, net.b{1});
grid on;
axis([-3 3 -2 3]);
title('Complex area classification');