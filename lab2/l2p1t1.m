clc; clear; close all;

% Вхідні та цільові дані
P = [5 4 3 2];
T = [10 20 30 40];

% Перетворення у послідовність
X = con2seq(P);
Y = con2seq(T);

% Створення лінійної мережі з двома блоками затримки
net = linearlayer(1:2,0.01);

% Підготовка даних
[Xs,Xi,Ai,Ts] = preparets(net,X,Y);

% Параметри навчання
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-5;

% Навчання мережі
net = train(net,Xs,Ts,Xi,Ai);

% Моделювання
Out = sim(net,Xs,Xi);

% Перетворення результатів
T_real = cell2mat(Ts);
Y_real = cell2mat(Out);

% Помилка
E = T_real - Y_real;

figure;
plot(T_real,'b-o');
hold on;
plot(Y_real,'r-*');
grid on;
legend('Цільові значення','Вихід мережі');
title('Результат роботи мережі');

% Дослідження впливу кількості блоків затримки
errors = [];

delays = 1:length(P)-1;

for d = delays

    net = linearlayer(1:d,0.01);

    [Xs,Xi,Ai,Ts] = preparets(net,X,Y);

    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-5;

    net = train(net,Xs,Ts,Xi,Ai);

    Out = sim(net,Xs,Xi);

    T_real = cell2mat(Ts);
    Y_real = cell2mat(Out);

    err = mse(T_real - Y_real);

    errors = [errors err];
end

figure;
plot(delays,errors,'o-');
grid on;
xlabel('Кількість блоків затримки');
ylabel('Помилка MSE');
title('Вплив кількості блоків затримки');