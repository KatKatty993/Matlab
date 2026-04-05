clear; clc; close all; 
X3 = [ -1.5  -1.5 -1.5  -1.5 -1.5  -1.5 -1.5  -1.5;
       -1.5 -1.5  1.5   1.5 -1.5 -1.5  1.5   1.5;
       -1.5 -1.5 -1.5  -1.5  1.5  1.5  1.5   1.5 ];

T3 = [1 0 0 0 1 1 1 0];

% Створення та навчання персептрона
net3 = perceptron;
net3.inputWeights{1,1}.learnFcn = 'learnpn';
net3.biases{1}.learnFcn = 'learnpn';
net3 = train(net3,X3,T3);

Y3 = net3(X3);

figure;

% Індекси точок різних класів
idx0 = find(T3==0);
idx1 = find(T3==1);

% Побудова точок у 3D
scatter3(X3(1,idx0),X3(2,idx0),X3(3,idx0),'filled'); hold on;
scatter3(X3(1,idx1),X3(2,idx1),X3(3,idx1),'filled');

% Отримання параметрів площини розділення
w = net3.IW{1,1};
b = net3.b{1};

% Побудова площини
[xg,yg] = meshgrid(-4:0.5:4, -4:0.5:4);
zg = (-w(1)*xg - w(2)*yg - b) / w(3);
surf(xg,yg,zg,'FaceAlpha',0.3,'EdgeColor','none');

title('Task 3 - 3D separation');
xlabel('x1'); ylabel('x2'); zlabel('x3');
grid on; view(3);
