clc; clear; close all;

% 26 виходів для 26 літер
T = eye(26);


% 35 ознак (матриця 7x5)
% 26 літер латинського алфавіту

alphabet = round(rand(35,26));

q_pattern = [0 1 1 1 0;
             1 0 0 0 1;
             1 0 0 0 1;
             1 0 0 0 1;
             1 0 1 0 1;
             1 0 0 1 0;
             0 1 1 0 1];

alphabet(:,17) = q_pattern(:);

net = feedforwardnet(31,'trainlm');

% Використовувати всі дані для навчання
net.divideFcn = '';

% Максимальна кількість епох
net.trainParam.epochs = 1000;

% Цільова помилка
net.trainParam.goal = 1e-6;

net = train(net, alphabet, T);

Y = net(alphabet);

[~, predicted] = max(Y);
[~, target] = max(T);

fprintf('Результати розпізнавання:\n');

disp(predicted);

fprintf('Очікувані класи:\n');

disp(target);

fprintf('Коректність розпізнавання:\n');

disp(predicted == target);

% Рівні шуму від 0 до 0.5
noise_levels = 0:0.05:0.5;

% Масив середніх помилок
mean_errors = zeros(size(noise_levels));

for i = 1:length(noise_levels)

    std_dev = noise_levels(i);

    total_error = 0;

    % Перебір усіх літер
    for char_idx = 1:26

        % 10 зашумлених прикладів
        for trial = 1:10

            % Генерація шуму

            noise = std_dev * randn(35,1);

            % Обмеження шуму в межах (-1;1)
            noise(noise > 1) = 1;
            noise(noise < -1) = -1;

            % Додавання шуму
            noisy_input = alphabet(:,char_idx) + noise;

            % Вихід мережі
            y_out = net(noisy_input);

            % Обчислення евклідової помилки
            error_value = norm(T(:,char_idx) - y_out);

            total_error = total_error + error_value;

        end
    end

    % Середня помилка
    mean_errors(i) = total_error / (26 * 10);

end

figure;

plot(noise_levels, ...
     mean_errors, ...
     'b-o', ...
     'LineWidth', 2, ...
     'MarkerSize', 7);

grid on;

xlabel('Рівень шуму (std dev)', ...
       'FontSize', 12);

ylabel('Середня помилка', ...
       'FontSize', 12);

title('Залежність помилки від інтенсивності шуму', ...
      'FontSize', 13);

set(gca,'FontSize',11);

figure;

imagesc(reshape(alphabet(:,17),7,5));

colormap(gray);

axis equal;
axis tight;

title('Шаблон літери Q');
