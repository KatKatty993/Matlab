clear; clc; close all;

X4 = [ 0.5  0.8  0.9  1.2  1.5  0.2  0.1 -0.6 -1.0;
       1.3  1.6  1.8  0.6  0.8  0.1  0.5 -0.8 -1.2 ];

T4 = [ 0 0 0 1 1 0 0 1 1;
       1 1 1 1 1 0 0 0 0 ];

net4 = perceptron;

% Налаштування входів і нейронів
net4.inputs{1}.size = 2;
net4.layers{1}.size = 2;

% Правило навчання
net4.inputWeights{1,1}.learnFcn = 'learnpn';
net4.biases{1}.learnFcn = 'learnpn';

net4 = train(net4, X4, T4);

Y4 = net4(X4);

% Формування номерів класів (0..3)
classes = 2*Y4(1,:) + Y4(2,:);

figure; hold on;

% Побудова точок різних класів
colors = ['r','g','b','y'];

for c = 0:3
    idx = find(classes==c);
    scatter(X4(1,idx), X4(2,idx), 80, colors(c+1), 'filled');
end

% Отримання ваг і зміщень
W = net4.IW{1};
b = net4.b{1};

% Побудова ліній розділення
x = linspace(-2,2,100);

for i = 1:2
    y = -(W(i,1)*x + b(i)) / W(i,2);
    plot(x,y,'LineWidth',2);
end

title('Task 5 - Four classes');
xlabel('x1');
ylabel('x2');
grid on;
legend('Class 0','Class 1','Class 2','Class 3','Decision line 1','Decision line 2');

hold off;