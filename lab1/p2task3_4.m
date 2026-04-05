clear; clc; close all;

n = 200;
x = rand(2,n)*2-1; % координати від -1 до 1

t = zeros(2,n); % 2 класи

for i = 1:n
    r = x(1,i)^2 + x(2,i)^2;
    
    if r < 0.3
        t(:,i) = [1;0]; % чорний клас
    else
        t(:,i) = [0;1]; % білий клас
    end
end

% Побудова нейронної мережі

net = patternnet(6); % 6 нейронів у прихованому шарі

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-5;

net = train(net,x,t);

% Перевірка мережі

y = net(x);
[~,class] = max(y); % визначення класу

figure
hold on

for i = 1:n
    if class(i) == 1
        % чорний клас
        plot(x(1,i),x(2,i),'ko', ...
            'MarkerFaceColor','b', ...
            'MarkerSize',8)
    else
        % білий клас
        plot(x(1,i),x(2,i),'ko', ...
            'MarkerFaceColor','w', ...
            'MarkerSize',8)
    end
end

xlabel('X1')
ylabel('X2')
title('Classification of points by neural network')
grid on
axis equal