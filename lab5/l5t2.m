clc; clear; close all;

P = [0 0 1 0 1 1 0 1 1 1 0 0 1 0 0 1];

% XOR для двох попередніх значень входу
T = zeros(size(P));

for i = 3:length(P)
    T(i) = xor(P(i-1), P(i-2));
end

% Перетворення у послідовний формат
Pseq = con2seq(P);
Tseq = con2seq(T);

% Створення мережі Елмана
hidden_neurons = 6;
net = layrecnet(1:2, hidden_neurons, 'trainlm');

net.trainParam.epochs = 400;
net.trainParam.goal = 1e-5;

[Xs, Xi, Ai, Ts] = preparets(net, Pseq, Tseq);

net = train(net, Xs, Ts, Xi, Ai);

Ptest = [1 0 0 1 1 0 1 0 0 1 1 1 0 1];
Ttest = zeros(size(Ptest));

for i = 3:length(Ptest)
    Ttest(i) = xor(Ptest(i-1), Ptest(i-2));
end

Ptest_seq = con2seq(Ptest);

[Xtest, Xitest, Aitest] = preparets(net, Ptest_seq);
Y = net(Xtest, Xitest, Aitest);

Ynum = cell2mat(Y);
Yround = round(Ynum);

disp('Вхідна тестова послідовність:');
disp(Ptest);

disp('Очікуваний результат XOR:');
disp(Ttest);

disp('Результат мережі:');
disp(Yround);

figure;
plot(Ttest, 'bo-', 'LineWidth', 2);
hold on;
plot(Yround, 'r*-', 'LineWidth', 2);
grid on;
legend('Очікуваний результат', 'Результат мережі');
title('Динамічне перетворення XOR мережею Елмана');
xlabel('Номер елемента послідовності');
ylabel('Значення');