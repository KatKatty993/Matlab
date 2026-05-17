clc; clear; close all;

x1 = -3 : 0.5 : 3;
x2 = -2 : 0.5 : 2;

[X1, X2] = meshgrid(x1, x2);

% Формула функції
F = 2 * exp(2 - 5 * X2.^2) + 5 * (X1 - X2.^2);

% Формування вхідних векторів
P = [X1(:)'; X2(:)'];

% Цільові значення
T = F(:)';

goals = [0.1 0.01 0.001];

neurons_count = zeros(size(goals));

rbf_nets = cell(length(goals),1);

fprintf('RBF МЕРЕЖІ\n');

for i = 1:length(goals)

    goal = goals(i);

    % Створення RBF мережі
    net = newrb(P, T, goal);

    % Збереження мережі
    rbf_nets{i} = net;

    % Кількість нейронів
    neurons_count(i) = net.layers{1}.size;

    fprintf('Goal = %.3f ---> Нейронів = %d\n', ...
            goal, neurons_count(i));

end

figure;

plot(goals, neurons_count, ...
     'r-o', ...
     'LineWidth', 2, ...
     'MarkerSize', 8);

grid on;

xlabel('Значення помилки goal');
ylabel('Кількість нейронів');

title('Залежність кількості нейронів від goal');

set(gca,'XDir','reverse');

%  Тестові дані
x1_test = -3 : 0.25 : 3;
x2_test = -2 : 0.25 : 2;

[X1t, X2t] = meshgrid(x1_test, x2_test);

% Реальна функція
F_real = 2 * exp(2 - 5 * X2t.^2) + ...
         5 * (X1t - X2t.^2);

P_test = [X1t(:)'; X2t(:)'];

%  Використання найточнішої мережі
best_net = rbf_nets{3};

Y = best_net(P_test);

F_approx = reshape(Y, size(X1t));

%  Масив помилок
Errors = F_real - F_approx;

%  Апроксимована функція
figure;

surf(X1t, X2t, F_approx);

xlabel('x1');
ylabel('x2');
zlabel('f(x1,x2)');

title('Апроксимована функція');

shading interp;
grid on;

%  Графік помилки апроксимації
figure;

surf(X1t, X2t, Errors);

xlabel('x1');
ylabel('x2');
zlabel('Помилка');

title('Помилка апроксимації');

shading interp;
grid on;

%  Створення алфавіту
alphabet = round(rand(35,26));

% Шаблон літери Q
q_pattern = [0 1 1 1 0;
             1 0 0 0 1;
             1 0 0 0 1;
             1 0 0 0 1;
             1 0 1 0 1;
             1 0 0 1 0;
             0 1 1 0 1];

alphabet(:,17) = q_pattern(:);

%  Цільові класи
targets = ind2vec(1:26);

%  Створення PNN мережі
spread = 0.1;

pnn_net = newpnn(alphabet, targets, spread);

fprintf('PNN МЕРЕЖА СТВОРЕНА\n');

%  Перевірка без шуму
Y_pnn = pnn_net(alphabet);

[~, predicted] = max(Y_pnn);

fprintf('\nРезультати класифікації:\n');

disp(predicted);

%  Дослідження стійкості до шуму
noise_levels = 0 : 0.05 : 0.5;

mean_errors = zeros(size(noise_levels));

for i = 1:length(noise_levels)

    std_dev = noise_levels(i);

    total_error = 0;

    for char_idx = 1:26

        for trial = 1:10

            % Генерація шуму
            noise = std_dev * randn(35,1);

            % Обмеження [-1 ; 1]
            noise(noise > 1) = 1;
            noise(noise < -1) = -1;

            % Зашумлений вхід
            noisy_input = alphabet(:,char_idx) + noise;

            % Робота мережі
            y_out = pnn_net(noisy_input);

            % Помилка
            error_value = norm(targets(:,char_idx) - y_out);

            total_error = total_error + error_value;

        end
    end

    mean_errors(i) = total_error / (26 * 10);

end

%  Графік залежності помилки від шуму
figure;

plot(noise_levels, ...
     mean_errors, ...
     'b-o', ...
     'LineWidth', 2, ...
     'MarkerSize', 7);

grid on;

xlabel('Рівень шуму');
ylabel('Середня помилка');

title('Стійкість PNN мережі до шуму');

set(gca,'FontSize',11);

%  Відображення літери q
figure;

imagesc(reshape(alphabet(:,17),7,5));

colormap(gray);

axis equal;
axis tight;

title('Шаблон літери Q');