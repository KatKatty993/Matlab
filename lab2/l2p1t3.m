clc; clear; close all;

% Перша ділянка сигналу
Fs1 = 100;
t1 = 0:1/Fs1:3;
x1 = sin(2*pi*t1);

% Друга ділянка сигналу
Fs2 = 40;
t2 = 0:1/Fs2:2;
x2 = sin(4*pi*t2);

% Об'єднання
x = [x1 x2];

% Формування затриманого сигналу
x_delay = [0 x(1:end-1)];
y = 3*x_delay + 1;

% Перетворення
X = con2seq(x);
T = con2seq(y);

% Мережа
net = linearlayer(1:2,0.01);

[Xs,Xi,Ai,Ts] = preparets(net,X,T);

net.trainParam.epochs = 1500;
net.trainParam.goal = 1e-5;

net = train(net,Xs,Ts,Xi,Ai);

% Моделювання
Out = sim(net,Xs,Xi);

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

% Дослідження затримок
errors = [];
delays = 1:8;

for d = delays
    net = linearlayer(1:d,0.01);

    [Xs,Xi,Ai,Ts] = preparets(net,X,T);

    net.trainParam.epochs = 1500;
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
