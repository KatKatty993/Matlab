clc; clear; close all;

% Створення ідеальної одиничної матриці для цілей (26 класів)
T = eye(26);

% Створення матриці алфавіту (35 ознак на 26 літер)
alphabet = round(rand(35, 26));

% Додавання шаблону Q у 17-й стовпець
q_pattern = [0 1 1 1 0;
             1 0 0 0 1;
             1 0 0 0 1;
             1 0 0 0 1;
             1 0 1 0 1;
             1 0 0 1 0;
             0 1 1 0 1];

alphabet(:, 17) = q_pattern(:);

% 31 нейрон у прихованому шарі
net = feedforwardnet(31, 'trainlm');

% Використання даних для навчання
net.divideFcn = '';

% Налаштування параметрів навчання
net.trainParam.goal = 1e-6;

net = train(net, alphabet, T);

% Дослідження впливу шуму
noise_levels = 0:0.05:0.5;
mean_errors = zeros(size(noise_levels));

for i = 1:length(noise_levels)

    std_dev = noise_levels(i);
    total_error = 0;

    for char_idx = 1:26

        for trial = 1:10

            % Додавання шуму
            noisy_input = alphabet(:, char_idx) + ...
                          std_dev * randn(35, 1);

            % Моделювання виходу
            y_out = net(noisy_input);

            % Обчислення евклідової норми помилки
            total_error = total_error + ...
                          norm(T(:, char_idx) - y_out);

        end
    end

    % Середня помилка для поточного рівня шуму
    mean_errors(i) = total_error / (26 * 10);

end

figure;

plot(noise_levels, mean_errors, ...
     'b-o', ...          
     'LineWidth', 2, ...
     'MarkerSize', 6);

grid on;

xlabel('Рівень шуму (std dev)', 'FontSize', 12);
ylabel('Середня помилка (Euclidean norm)', 'FontSize', 12);

title('Стійкість розпізнавання алфавіту до зашумлення', ...
      'FontSize', 13);

set(gca, 'FontSize', 11);