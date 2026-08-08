clc; clear; close all;

P = [0 0 1 0 1 1 0 1 1 1 0 0 1 1 0 1];

% 1, якщо поточне і попереднє значення дорівнюють 1
T = zeros(size(P));

for i = 2:length(P)
    if P(i) == 1 && P(i-1) == 1
        T(i) = 1;
    end
end

% Перетворення у послідовний формат
Pseq = con2seq(P);
Tseq = con2seq(T);

% Створення мережі Елмана
hidden_neurons = 5;
net = layrecnet(1, hidden_neurons, 'trainlm');

net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;

[Xs, Xi, Ai, Ts] = preparets(net, Pseq, Tseq);

net = train(net, Xs, Ts, Xi, Ai);

Ptest = [1 0 1 1 0 0 1 1 1 0 1 0 1 1];
Ttest = zeros(size(Ptest));

for i = 2:length(Ptest)
    if Ptest(i) == 1 && Ptest(i-1) == 1
        Ttest(i) = 1;
    end
end

Ptest_seq = con2seq(Ptest);

[Xtest, Xitest, Aitest] = preparets(net, Ptest_seq);
Y = net(Xtest, Xitest, Aitest);

Ynum = cell2mat(Y);
Yround = round(Ynum);

disp('Вхідна тестова послідовність:');
disp(Ptest);

disp('Очікуваний результат:');
disp(Ttest);

disp('Результат мережі:');
disp(Yround);

figure;
plot(Ttest, 'bo-', 'LineWidth', 2);
hold on;
plot(Yround, 'r*-', 'LineWidth', 2);
grid on;
legend('Очікуваний результат', 'Результат мережі');
title('Розпізнавання двох одиниць підряд');
xlabel('Номер елемента послідовності');
ylabel('Значення');