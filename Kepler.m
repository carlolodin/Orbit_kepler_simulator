clear; clc; close all;
format short e

%% ====Condizioni Iniziali====

% Sistema Sole - Terra
%m1 = 1.989e30;          % Massa del Sole [kg]
%m2 = 5.972e24;          % Massa della Terra [kg]
%r = [1.47e11; 0; 0];    % Posizione iniziale in 3D [m]
%v = [0; 30.29e3; 0];    % Velocità iniziale in 3D [m/s]

%m1 = 1.989e30;          % Massa del Sole [kg]
%m2 = 5.972e24;
%r = [1.47e11; 0; 0];    % Posizione iniziale in 3D [m]
%v = [-5e3; 12.3e3; 2e3];    % Velocità iniziale in 3D [m/s]

m1 = 1.989e30;          % Massa del Sole [kg]
m2 = 5.972e24;
r = [1.47e11; 0; 0];    % Posizione iniziale in 3D [m]
v = [-5e4; 12.3e3; 2e4];    % Velocità iniziale in 3D [m/s]

%% ====Costanti fisiche e parametri globali====
G = 6.67430e-11;    % Costante gravitazionale [m^3 kg^-1 s^-2]
M = m1 + m2;        % Massa totale
mu = (m1 * m2) / M; % Massa ridotta



% Parametri di simulazione
t_max = 365 * 24 * 3600;        % Tempo totale [s] (Ad esempio qui un anno terrestre) 
dt = 3600;                       % Passo temporale [s]
N = floor(t_max / dt);          % Numero di passi

% Inizializzazione delle variabili
r_traj = zeros(3, N);
E_tot = zeros(1, N);
L_tot = zeros(1, N);
a = -G * M * r / norm(r)^3;     % Computa l'accelerazione iniziale

%% ======= SIMULAZIONE =========
disp('Inizio della simulazione...');
tic;

for i = 1:N
    r_traj(:, i) = r;                   % Salva la posizione attuale
    r = r + v * dt + 0.5 * a * dt^2;    % Aggiorna posizione
    a_new = -G * M * r / norm(r)^3;     % Calcola accelerazione nel nuovo punto    
    v = v + 0.5 * (a + a_new) * dt;     % Aggiorna velocità con la nuova accelerazione    
    a = a_new;                          % Aggiorna il vettore accelerazione

    T = 0.5 * mu * norm(v)^2;           % Calcola energia totale
    U = -G * M * mu / norm(r);
    E_tot(i) = T + U;
    L_tot(i) = norm(cross(r,v));
end

sim_time = toc;
disp(['Simulazione completata in ', num2str(sim_time), ' secondi.']);

% Inizializzazione array per le aree dei triangoli
area_triangoli = zeros(1, N-1); % Un triangolo per ogni coppia di punti consecutivi

% Calcolo delle aree dei triangoli
for i = 1:N-1
    % Due punti consecutivi della traiettoria
    r1 = r_traj(:, i);
    r2 = r_traj(:, i+1);

    % Calcolo del prodotto vettoriale
    cross_prod = cross(r1, r2);

    % Calcolo dell'area del triangolo
    area_triangoli(i) = 0.5 * norm(cross_prod);
end
var_relativa_area = 100 * (max(area_triangoli) - min(area_triangoli)) / mean(area_triangoli);

%% ======= Calcolo della conica con il miglior fit =======
disp('Calcolo della conica con il miglior fit');
tic;

% Caricamento delle posizioni della traiettoria simulata
x = r_traj(1, :);
y = r_traj(2, :);
z = r_traj(3, :);

%PCA per trovare il piano
xyz = [x; y; z]';               % Matrice delle posizioni
xyz_mean = mean(xyz, 1);        % Centro della traiettoria
xyz_centered = xyz - xyz_mean;  % Centra i dati

plane_vectors = pca(xyz_centered);      

% Base ortonormale del piano
plane_x = plane_vectors(:,1);  % Primo autovettore (direzione principale)
plane_y = plane_vectors(:,2);  % Secondo autovettore
plane_normal = plane_vectors(:,3);  % Normale al piano

% Proiezione dei punti sul piano
proj_x = xyz_centered * plane_x;
proj_y = xyz_centered * plane_y; 

% Fit della conica in 2D
M = [proj_x.^2, proj_x .* proj_y, proj_y.^2, proj_x, proj_y, ones(size(proj_x))];   % Matrice con termini quadratici, lineari e costanti presi a partire dalle coordinate nel piano
[~, ~, V] = svd(M, 0);                                                              % Decomposizione ai valori singolari
params = V(:, end);                                                                 % L'ultima colonna di V è la soluzione ottimale

A = params(1);  %Parametri della conica A*x'^2 + B*x'*y' + C*y'^2 +  D*x' + E*y' + F
B = params(2);
C = params(3);
D = params(4);
E = params(5);
F = params(6);

% Calcolo una serie di punti della conica in coordinate 2D
theta = linspace(0, 2*pi, 200); % Angoli per la parametrizzazione
cos_t = cos(theta);
sin_t = sin(theta);

% Creiamo una griglia di punti per il plot della conica
[X_plot, Y_plot] = meshgrid(linspace(min(proj_x), max(proj_x), 100), linspace(min(proj_y), max(proj_y), 100));

% Equazione della conica
Z_plot = A * X_plot.^2 + B * X_plot .* Y_plot + C * Y_plot.^2 + D * X_plot + E * Y_plot + F;

% Troviamo i punti (x', y') che soddisfano la conica (livello Z = 0)
contour_pts = contourc(X_plot(1,:), Y_plot(:,1), Z_plot, [0 0]);
contour_pts = contour_pts(:, 2:end); % Rimuoviamo il primo valore (header di contourc)
proj_x_conic = contour_pts(1, :);
proj_y_conic = contour_pts(2, :);

% Riporta i punti nel sistema di riferimento originale
conic_3D = xyz_mean' + plane_x * proj_x_conic + plane_y * proj_y_conic;

%Determino il tipo di conica
discriminante = B^2 - 4 * A * C;
if discriminante < 0
    conic_type = 'Ellisse';
elseif discriminante == 0
    conic_type = 'Parabola';
else
    conic_type = 'Iperbole';
end

% Calcolo dell'eccentricità
if discriminante < 0
    % Per un'ellisse, i semi-assi a e b sono legati ai coefficienti A, B, C.
    M_conica = [A B/2; B/2 C];
    eig_values = eig(M_conica);
    
    % I semi-assi sono le radici degli autovalori
    a = sqrt(1 / max(eig_values));  % Semi-asse minore
    b = sqrt(1 / min(eig_values));  % Semi-asse maggiore
    
    % Calcoliamo la distanza dei fuochi dal centro
    c = sqrt(b^2 - a^2);
    
    % Calcoliamo l'eccentricità
    eccentricity = sqrt(1 - (b^2 / a^2));
    conic_type = strcat(conic_type , ', e = ' , string(eccentricity));
    disp(strcat('La traiettoria è un-ellisse con eccentricità e= ', string(eccentricity) ))

    % Coordinate dei fuochi nel sistema 2D
    F1_2D = c * plane_vectors(:,1);
    F2_2D = -c * plane_vectors(:,1);

    % Ricostruiamo le coordinate originali 3D dai punti proiettati
    F1_3D = xyz_mean' + (plane_x./norm(plane_x)) * F1_2D(1) + (plane_y./norm(plane_y)) * F1_2D(2);
    F2_3D = xyz_mean' + (plane_x./norm(plane_x)) * F2_2D(1) + (plane_y./norm(plane_y)) * F2_2D(2);

    % Calcola il raggio medio
    distanze = sqrt(sum(r_traj.^2, 1));
    distanza_media = mean(distanze);
    focus_error = 100 * min(norm(F1_3D), norm(F2_3D)) / distanza_media;
    disp(strcat('Il raggio medio dell-orbita è:', string(distanza_media), ' metri'))
    disp(strcat('La distanza tra il fuoco simulato e quello teorico è il:', string(focus_error), '% del raggio medio'));

else
    F1_3D = [NaN, NaN, NaN];
    F2_3D = [NaN, NaN, NaN];
end

% Plot
figure;
hold on;
grid on;
axis equal;

% Scatter dei punti originali
scatter3(x, y, z, 25, 'bo', 'filled');

% Disegno del piano trovato
[X_plane, Y_plane] = meshgrid(linspace(min(x), max(x), 20), linspace(min(y), max(y), 20));
Z_plane = (-plane_normal(1) * (X_plane - xyz_mean(1)) - plane_normal(2) * (Y_plane - xyz_mean(2))) / plane_normal(3) + xyz_mean(3);
surf(X_plane, Y_plane, Z_plane, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', 'g'); % Piano semi-trasparente

plot3(conic_3D(1, :), conic_3D(2, :), conic_3D(3, :), 'r', 'LineWidth', 2);                 %Plot della conica

plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'y');                             % Corpo principale

% Plot dei fuochi in verde
plot3(F1_3D(1), F1_3D(2), F1_3D(3), 'go', 'MarkerSize', 5, 'MarkerFaceColor', 'g');
plot3(F2_3D(1), F2_3D(2), F2_3D(3), 'go', 'MarkerSize', 5, 'MarkerFaceColor', 'g');


xlabel('X');
ylabel('Y');
zlabel('Z');
title('Fit della conica in 3D');
if discriminante < 0 && focus_error < 10
    legend('Punti dati', 'Piano stimato', strcat('Conica stimata (', conic_type, ')'), 'Sole', strcat('Fuochi stimati. Errore rispetto al raggio:', string(focus_error), '%' ));
else
    legend('Punti dati', 'Piano stimato', strcat('Conica stimata (', conic_type, ')'), 'Sole');
end
view(3);

sim_time = toc;
disp(['Conica calcolata in  ', num2str(sim_time), ' secondi.']);

% Equazione del piano
plane_eq = sprintf('%.3fx + %.3fy + %.3fz = %.3f', ...
    plane_normal(1), plane_normal(2), plane_normal(3), ...
    dot(plane_normal, xyz_mean));

text(min(x), min(y), max(z), plane_eq, 'FontSize', 12, 'Color', 'k', 'BackgroundColor', 'w');
%text(max(x), min(y), max(z), strcat('Fuoco 1 in ', string(F1_3D)), 'FontSize', 12, 'Color', 'k', 'BackgroundColor', 'w');
%text(min(x), max(y), max(z), strcat('Fuoco 2 in ', string(F2_3D)), 'FontSize', 12, 'Color', 'k', 'BackgroundColor', 'w');

%% ======= Plot energia totale, momento angolare e area dei triangoli=======
figure
subplot(3,1,1);
h_energy = plot((0:N-1) * dt / (24 * 3600), E_tot, 'b', 'LineWidth', 1.5);
xlabel('Tempo [giorni]');
ylabel('Energia Totale [J]');
title('Conservazione dell’Energia Totale');
grid on;

% Calcola la variazione relativa dell'energia
delta_E_rel = 100 * (max(E_tot) - min(E_tot)) / mean(E_tot);
text(N * dt / (24 * 3600) * 0.7, min(E_tot) + 0.05 * (max(E_tot) - min(E_tot)), sprintf('\\DeltaE/E_{media} = %.6f %%', delta_E_rel), 'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');
if norm(delta_E_rel) > 2
    disp('========================')
    disp('WARNING! La variazione percentuale di energia è maggiore del 2% attorno alla media.')
    disp('La simulazione potrebbe essere inaffidabile!')
    disp('========================')

end


subplot(3,1,2);
h_momentum = plot((0:N-1) * dt / (24 * 3600), L_tot, 'b', 'LineWidth', 1.5);
xlabel('Tempo [giorni]');
ylabel('Momento angolare [Kg * m^2 / s]');
title('Conservazione del Momento Angolare Totale');
grid on;

% Calcola la variazione relativa del momento
delta_L_rel = 100 * (max(L_tot) - min(L_tot)) / mean(L_tot);
text(N * dt / (24 * 3600) * 0.7, min(L_tot) + 0.05 * (max(L_tot) - min(L_tot)), sprintf('\\DeltaL/L_{medio} = %.6f %%', delta_L_rel), 'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold');


% ======= Plot aree dei triangoli =======
subplot(3,1,3);
plot((1:N-1) * dt / (24 * 3600), area_triangoli, 'g', 'LineWidth', 1.5);
xlabel('Tempo [giorni]');
ylabel('Area del Triangolo [m^2]');
title(['Aree dei Triangoli Formati dalla Traiettoria (Errore Medio Relativo: ', num2str(var_relativa_area), ')']);
grid on;

%% ======= ANIMAZIONE =========
disp('Avvio dell''animazione...');

figure;
hold on;
set(gcf, 'Units', 'normalized', 'OuterPosition', [0 0 1 1]);                    % Imposta la finestra a schermo intero
h_orbit = plot3(NaN, NaN, NaN, 'r', 'LineWidth', 1.5);                          % Linea dell'orbita
h_body = plot3(0, 0, 0, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');        % Corpo principale
h_point = plot3(NaN, NaN, NaN, 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');  % Corpo secondario
grid on;
axis equal;

% Determina i limiti degli assi basati sulla traiettoria massima
max_r = max(max(abs(r_traj), [], 2));
xlim([-max_r, max_r]);
ylim([-max_r, max_r]);
zlim([-max_r, max_r]);

xlabel('Posizione X [m]');
ylabel('Posizione Y [m]');
zlabel('Posizione Z [m]');
title('Problema dei due corpi - Coordinate relative (3D)');
view(3);

% Animazione della traiettoria in 3D
for i = 1:N
    set(h_orbit, 'XData', r_traj(1, 1:i), 'YData', r_traj(2, 1:i), 'ZData', r_traj(3, 1:i));    % Aggiorna l'orbita
    set(h_point, 'XData', r_traj(1, i), 'YData', r_traj(2, i), 'ZData', r_traj(3, i));          % Aggiorna la posizione del corpo secondario
    pause(0.0);                                                                                 % Pausa per creare l'effetto animato
end

disp('Animazione completata.');
