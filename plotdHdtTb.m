dhdt_dir = '/data-archive/download-extended/ICESat1_ICESat2_mass_change_updated_2_2021 (1)/ICESat1_ICESat2_mass_change_updated_2_2021/dhdt/';
Tb_dir   = '/home/donglaiyang/Documents/Georgia-Tech/Research/GMR-inference/data/';
Tpmp_dir = '/home/donglaiyang/Documents/Georgia-Tech/Research/thermal-model/Amundsen-thermal-output-Yang/thermal-training-data/Thwaites-PIG/training/gridded/';

dhdt_file = [dhdt_dir 'ais_dhdt_grounded.tif'];
Tb_file   = [Tb_dir 'posterior_mean_Tb.tif'];
mask_file = [Tpmp_dir 'training_mask_domain_continuous.mat'];
H_file    = [Tpmp_dir 'H_gridded.mat'];
coord_file= [Tpmp_dir 'trainingAll_image_coord.mat'];

% load smith et al 2020 data
[dhdt,R] = readgeoraster(dhdt_file,'OutputType','double');
[X,Y] = worldGrid(R);

% load basal temperature
[Tb,R] = readgeoraster(Tb_file,'OutputType','double');
[X_ase,Y_ase] = worldGrid(R);

% interpolate dhdt data onto ASE coordinate
dhdt_ase = interp2(X,Y,dhdt,X_ase,Y_ase);

% load other data
load(mask_file)
load(H_file)
load(coord_file)

Tpmp = computePMP(H_struct.H);

%%
levels1 = [-0.3,-0.2,-0.1,0];

figure;

% ── Left plot: dhdt_ase ──────────────────────────────────────────
subplot(1,2,1);
imagesc(X_ase(1,:), Y_ase(:,1), dhdt_ase);
set(gca,'YDir','normal');
clim([-1.5, 0]);
colormap(gca, cmocean('ice'));
colorbar;
hold on;

[C1, h1] = contour(X_ase(1,:), Y_ase(:,1), dhdt_ase, ...
                   levels1, ...
                   'LineColor', 'r', ...
                   'LineWidth', 3);
if ~isempty(C1)
    clabel(C1, h1, 'Color', 'r', 'FontSize', 15, 'LabelSpacing', 200);
end
hold off;
title('dh/dt (m/yr)');

% ── Right plot: Tpmp - Tb ─────────────────────────────────────────
dT = Tpmp - Tb;   % positive = below melting point, 0 = at melting

subplot(1,2,2);
imagesc(X_ase(1,:), Y_ase(:,1), dT);
set(gca,'YDir','normal');
colormap(gca, cmocean('matter'));
colorbar;
hold on;

% Contour where dT < 1 (i.e. draw the boundary AT dT = 1)
[C2, h2] = contour(X_ase(1,:), Y_ase(:,1), dT, ...
                   [1, 1], ...
                   'LineColor', [0.2 0.9 0.3], ...
                   'LineWidth', 3.0);
if ~isempty(C2)
    clabel(C2, h2, 'Color', [0.2 0.9 0.3], 'FontSize', 15, 'LabelSpacing', 200);
end
hold on
[C1, h1] = contour(X_ase(1,:), Y_ase(:,1), dhdt_ase, ...
                   levels1, ...
                   'LineColor', 'r', ...
                   'LineWidth', 3);
if ~isempty(C1)
    clabel(C1, h1, 'Color', 'r', 'FontSize', 15, 'LabelSpacing', 200);
end
hold off;
title('T_{pmp} - T_b  (K)');
exportgraphics(gcf,'figs/dhdt_Tb.png','Resolution',300)