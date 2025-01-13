clc; clear; close all;

testIdx = 2;

load(sprintf('lin_x_%d.mat', testIdx))
load(sprintf('xsim_%d.mat', testIdx))
load(sprintf('usim_%d.mat', testIdx))
load(sprintf('xref_%d.mat', testIdx))
load(sprintf('uref_%d.mat', testIdx))

t = time;
umin = [15; -0.8]; umax = [30; 0.8];

WPNum = 5;
WPListState = cell(1, WPNum);

switch(testIdx)
    case 1
        WPListState{1} = [0; 50; 100; pi/6]; 
        WPListState{2} = [10; 250; 200; pi/6];
        WPListState{3} = [20; 450; 300; pi/6];
        WPListState{4} = [30; 650; 400; pi/6];
        WPListState{5} = [40; 850; 500; pi/6];
    
    case 2
        WPListState{1} = [0; 50; 100; pi/6]; 
        WPListState{2} = [10; 250; 200; pi/4];
        WPListState{3} = [20; 450; 300; pi/3];
        WPListState{4} = [30; 650; 400; pi/3];
        WPListState{5} = [40; 850; 500; pi/6];


    case 3
        WPListState{1} = [0; 50; 100; pi/6]; 
        WPListState{2} = [10; 240; 185; pi/4];
        WPListState{3} = [20; 435; 325; -pi/20];
        WPListState{4} = [30; 630; 420; -pi/10];
        WPListState{5} = [40; 850; 500; pi/12];

    case 4
        WPListState{1} = [0; 50; 100; pi/2.3];
        WPListState{2} = [10; 180; 230; pi/1.6];
        WPListState{3} = [20; 290; 380; pi/2.3];
        WPListState{4} = [30; 440; 450; pi/2.2];
        WPListState{5} = [40; 490; 640; pi/1.12];
end
%% Plot Trajectory
figure; hold on; grid on;
plot(x_ref(1, :), x_ref(2, :));
for i = 1 : WPNum
    stem(WPListState{i}(2), WPListState{i}(3), '--r', 'HandleVisibility', 'off')
end
stem(xsim(1, end), xsim(2, end), '--r', 'HandleVisibility', 'off')
xlabel("x (m)"); ylabel("y (m)");
title("Position evolution")

saveas(gcf, sprintf('images/gen_trajectory_pe_%d', testIdx), 'epsc');

figure; 
subplot(3, 1, 1); hold on; grid on;
plot(t, x_ref(1, :));
for i = 1 : WPNum
    stem(WPListState{i}(1), WPListState{i}(2), '--r', 'HandleVisibility', 'off')
end
stem(t(end), xsim(1, end), '--r', 'HandleVisibility', 'off')
title("Position x"); xlabel("Time"); ylabel("x(m)");

subplot(3, 1, 2); hold on; grid on;
plot(t, x_ref(2, :));
for i = 1 : WPNum
    stem(WPListState{i}(1), WPListState{i}(3), '--r', 'HandleVisibility', 'off')
end
stem(t(end), xsim(2, end), '--r', 'HandleVisibility', 'off')
title("Position y"); xlabel("Time"); ylabel("y (m)")

subplot(3, 1, 3); hold on; grid on;
plot(t, x_ref(3, :));
for i = 1 : WPNum
    stem(WPListState{i}(1), WPListState{i}(4), '--r', 'HandleVisibility', 'off')
end
stem(t(end), xsim(3, end), '--r', 'HandleVisibility', 'off')
title("Yaw angle"); xlabel("Time"); ylabel("Yaw (rad)")

saveas(gcf, sprintf('images/gen_trajectory_states_%d', testIdx), 'epsc');

figure;
subplot(2, 1, 1); hold on; grid on;
plot(t, u_ref(1, :));
legend("v_a");
yline(umin(1), '--r', 'HandleVisibility', 'off');
yline(umax(1), '--r', 'HandleVisibility', 'off');
title("Air Speed"); xlabel("Time"); ylabel("v_a");

subplot(2, 1, 2); hold on; grid on;
plot(t, u_ref(2, :));
legend("\phi");
yline(umin(2), '--r', 'HandleVisibility', 'off');
yline(umax(2), '--r', 'HandleVisibility', 'off');
title("Roll Angle"); xlabel("Time"); ylabel("Roll")

saveas(gcf, sprintf('images/gen_trajectory_commands_%d', testIdx), 'epsc');

%% Plot results
figure; hold on; grid on;
plot(xsim(1, :), xsim(2, :));
plot(x_ref(1, :), x_ref(2, :));
for i = 1 : WPNum
    stem(WPListState{i}(2), WPListState{i}(3), '--r', 'HandleVisibility', 'off')
end
stem(xsim(1, end), xsim(2, end), '--r', 'HandleVisibility', 'off')
legend("sim", "ref")
xlabel("x (m)"); ylabel("y (m)");
title("Position evolution")

saveas(gcf, sprintf('images/results_pe_%d', testIdx), 'epsc');

%% States
figure; 
subplot(3, 1, 1); hold on; grid on;
plot(t, xsim(1, :));
plot(t, x_ref(1, :));
legend("Simulation", "Reference", "Location", "southeast");
for i = 1 : WPNum
    stem(WPListState{i}(1), WPListState{i}(2), '--r', 'HandleVisibility', 'off')
end
stem(t(end), xsim(1, end), '--r', 'HandleVisibility', 'off')
title("Position x"); xlabel("Time"); ylabel("x(m)");

subplot(3, 1, 2); hold on; grid on;
plot(t, xsim(2, :));
plot(t, x_ref(2, :));
for i = 1 : WPNum
    stem(WPListState{i}(1), WPListState{i}(3), '--r', 'HandleVisibility', 'off')
end
stem(t(end), xsim(2, end), '--r', 'HandleVisibility', 'off')
legend("Simulation", "Reference", "Location", "southeast");
title("Position y"); xlabel("Time"); ylabel("y (m)")

subplot(3, 1, 3); hold on; grid on;
plot(t, xsim(3, :));
plot(t, x_ref(3, :));
for i = 1 : WPNum
    stem(WPListState{i}(1), WPListState{i}(4), '--r', 'HandleVisibility', 'off')
end
stem(t(end), xsim(3, end), '--r', 'HandleVisibility', 'off')
legend("Simulation", "Reference", "Location", "southeast");
title("Yaw angle"); xlabel("Time"); ylabel("Yaw (rad)")

saveas(gcf, sprintf('images/results_states_%d', testIdx), 'epsc');

%% Commands
figure;
subplot(2, 1, 1); hold on; grid on;
plot(t, usim(1, :));
plot(t, u_ref(1, :));
legend("v_a");
yline(umin(1), '--r', 'HandleVisibility', 'off');
yline(umax(1), '--r', 'HandleVisibility', 'off');
legend("Simulation", "Reference");
title("Air Speed"); xlabel("Time"); ylabel("v_a");

subplot(2, 1, 2); hold on; grid on;
plot(t, usim(2, :));
plot(t, u_ref(2, :));
legend("\phi");
yline(umin(2), '--r', 'HandleVisibility', 'off');
yline(umax(2), '--r', 'HandleVisibility', 'off');
legend("Simulation", "Reference");
title("Roll Angle"); xlabel("Time"); ylabel("Roll")

saveas(gcf, sprintf('images/results_commands_%d', testIdx), 'epsc');

%% Linearization Comparison
figure;
subplot(3, 1, 1); hold on; grid on;
plot(t(1:length(lin_x(2, :))), xsim(1, 1:length(lin_x(2, :))));
plot(t(1:length(lin_x(2, :))), lin_x(1, :));
legend("Nonlinear", "Linearized", "Location", "southeast");
for i = 1 : WPNum
    stem(WPListState{i}(1), WPListState{i}(2), '--r', 'HandleVisibility', 'off')
end
stem(t(end), xsim(1, end), '--r', 'HandleVisibility', 'off')
ylim([100, 200])
title("Position x"); xlabel("Time"); ylabel("x (m)");

subplot(3, 1, 2); hold on; grid on;
plot(t(1:length(lin_x(2, :))), xsim(2, 1:length(lin_x(2, :))));
plot(t(1:length(lin_x(2, :))), lin_x(2, :));
for i = 1 : WPNum
    stem(WPListState{i}(1), WPListState{i}(3), '--r', 'HandleVisibility', 'off')
end
stem(t(end), xsim(2, end), '--r', 'HandleVisibility', 'off')
legend("Nonlinear", "Linearized", "Location", "southeast");
title("Position y"); xlabel("Time"); ylabel("y (m)")

subplot(3, 1, 3); hold on; grid on;
plot(t(1:length(lin_x(2, :))), xsim(3, 1:length(lin_x(2, :))));
plot(t(1:length(lin_x(2, :))), lin_x(3, :));
for i = 1 : WPNum
    stem(WPListState{i}(1), WPListState{i}(4), '--r', 'HandleVisibility', 'off')
end
stem(t(end), xsim(3, end), '--r', 'HandleVisibility', 'off')
legend("Nonlinear", "Linearized", "Location", "southeast");
title("Yaw angle"); xlabel("Time"); ylabel("Yaw")

%%
state_err = xsim(:, :) - x_ref(:, :);

err_x = mean(abs(state_err(1, :)))
err_y = mean(abs(state_err(2, :)))
err_yaw = mean(abs(state_err(3, :)))