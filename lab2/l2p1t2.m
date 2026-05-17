clc; clear; close all;

% Формування сигналу
Fs = 50;
Tmax = 2.5;
time = 0:1/Fs:Tmax;

x = sin(2*pi*time);
y = 2*x + 3;

% Перетворення в cell array
X = con2seq(x);
T = con2seq(y);

% Створення мережі
net = linearlayer(1:2,0.01);

% Підготовка даних
[Xs,Xi,Ai,Ts] = preparets(net,X,T);

% Навчання
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-5;

net = train(net,Xs,Ts,Xi,Ai);

% Моделювання
Out = sim(net,Xs,Xi);

% Результати
Y_real = cell2mat(Out);
T_real = cell2mat(Ts);

% Помилка
E = T_real - Y_real;

figure;
plot(E);
grid on;
title('Помилка мережі');
xlabel('Номер відліку');
ylabel('Помилка');

% Дослідження кількості затримок
errors = [];
delays = 1:8;

for d = delays
    net = linearlayer(1:d,0.01);

    [Xs,Xi,Ai,Ts] = preparets(net,X,T);

    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-5;

    net = train(net,Xs,Ts,Xi,Ai);

    Out = sim(net,Xs,Xi);

    Y_real = cell2mat(Out);
    T_real = cell2mat(Ts);

    err = mse(T_real - Y_real);
    errors = [errors err];
end

figure;
plot(delays,errors,'o-');
grid on;
title('Вплив кількості блоків затримки');
xlabel('Кількість блоків');
ylabel('MSE');
