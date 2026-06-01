clc; clear; close all;

% Генерація 48 векторів (6 кластерів)
centers = [4 2;
    3 8;
    5 5;
    8 1;
    8 8;
    5 2]';

std_dev = 0.4;
points_per_cluster = 8;
data = [];

for i = 1:6
    cluster_points = centers(:,i) + ...
        std_dev * randn(2, points_per_cluster);
    data = [data cluster_points];
end

% Початкові дані
figure;
plot(data(1,:), data(2,:), 'w+', ...
    'MarkerSize', 7, ...
    'LineWidth', 1.5);
hold on;
grid on;
title('Вхідні дані та процес навчання нейронів');
xlabel('X1');
ylabel('X2');

% Формування шару Кохонена 
net = competlayer(6);
net.trainParam.epochs = 5;

% Навчання кожні 5 епох 
colors = lines(10);

for i = 1:10

    net = train(net, data);

    weights = net.IW{1};

    plot(weights(:,1), weights(:,2), 'o', ...
        'MarkerSize', 10, ...
        'MarkerEdgeColor', colors(i,:), ...
        'LineWidth', 2);

    pause(0.5);

end

% Відображення  кластерів
figure;

Y = net(data);

clusterIndex = vec2ind(Y);

gscatter(data(1,:)', ...
    data(2,:)', ...
    clusterIndex);

hold on;

weights = net.IW{1};

plot(weights(:,1), weights(:,2), ...
    'wo', ...
    'MarkerSize', 14, ...
    'LineWidth', 3);

grid on;
title('Результат кластеризації');
xlabel('X1');
ylabel('X2');

legend('Кластер 1', ...
    'Кластер 2', ...
    'Кластер 3', ...
    'Кластер 4', ...
    'Кластер 5', ...
    'Кластер 6', ...
    'Нейрони');

% Тестові вектори. 
test_data = [3 5; 6 6; 5 1; 2 8]';

outputs = net(test_data);

% Конвертація виходу в індекси кластерів
cluster_indices = vec2ind(outputs);

disp('Тестові точки:');
disp(test_data);
disp('Призначені кластери:');
disp(cluster_indices);

plot(test_data(1,:), test_data(2,:), 'r*', ...
    'MarkerSize', 12, ...
    'LineWidth', 2);

% Підбір топології карти SOM

% Варіант 1: Гексагональна топологія
net_hex = selforgmap([2 3], 100, 3, 'hextop');
net_hex = train(net_hex, data);

% Варіант 2: Прямокутна топологія
net_grid = selforgmap([2 3], 100, 3, 'gridtop');
net_grid = train(net_grid, data);

% Візуалізація результатів
plotsompos(net_hex, data);
title('Гексагональна топологія [2 3]');

plotsompos(net_grid, data);
title('Прямокутна топологія [2 3]');

% відстані між сусідніми нейронами
plotsomnd(net_hex); 
title('SOM Neighbor Distances - Hex');

plotsomnd(net_grid);
title('SOM Neighbor Distances - Grid');

% Підготовка 3D даних 
PR = [0 10; 0 10; 0 10]; 
clusters_num = 4;        
points_num = 6;          
std_dev_3d = 0.8;     

rng(1); 

centers_3d = PR(:,1) + (PR(:,2) - PR(:,2)) .* rand(3, clusters_num);
data_3d = [];

for i = 1:clusters_num
    % Навколо кожного центру генеруємо хмару точок
    cluster = centers_3d(:, i) + std_dev_3d * randn(3, points_num);
    data_3d = [data_3d, cluster];
end

figure;
plot3(data_3d(1,:), data_3d(2,:), data_3d(3,:), 'w.', ...
    'MarkerSize', 25);
grid on;
title('Розподіл тривимірних векторів');
xlabel('X1');
ylabel('X2');
zlabel('X3');

% Підбір топології для 3D даних 

% Розмір 2x2 (4 нейрони) з гексагональною топологією
net_3d = selforgmap([2 2], 100, 3, 'hextop');
net_3d = train(net_3d, data_3d);

plotsompos(net_3d, data_3d);
title('SOM Weight Positions (3D Data)');

% Перевірка hits
plotsomhits(net_3d, data_3d);