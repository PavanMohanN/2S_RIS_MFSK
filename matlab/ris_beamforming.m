% ris_beamforming.m — Stage 2: R-RIS Beamforming and Phase Quantization
% ======================================================================
% Models the 16×16 UPA R-RIS radiation pattern, beam steering,
% and phase quantization sensitivity (1–4 bits).
%
% Paper equations:
%   ψ_n   = k_c · x_n · sin(θ1)                         [Eq. 9]
%   AF(θ) = (1/N) Σ_n Γ·exp(jψ_n) · exp(-jk_c·x_n·sinθ) [Eq. 12]
%   Quantised phase: ψ_q = round(ψ_n / Δψ) × Δψ,  Δψ = 2π/2^B
%
% Generates Figure F9 — two-panel:
%   (a) Cartesian: array factor dB vs azimuth, all quantisation levels
%   (b) Polar:     linear array factor, same curves
%
% Exports: data/ris_beamforming_data.mat
%
% Usage:
%   run ris_beamforming.m        (standalone)
%   Called by run_all_matlab.m

clear; clc; close all;

%% ══════════════════════════════════════════════════════════════════════════════
%  1. System parameters
%% ══════════════════════════════════════════════════════════════════════════════
C_LIGHT   = 2.998e8;
FC        = 28e9;
lambda    = C_LIGHT / FC;        % ≈ 10.71 mm
k_c       = 2*pi / lambda;       % wavenumber [rad/m]

N_X = 16;  N_Y = 16;
N   = N_X * N_Y;                 % Total elements per stage = 256
d   = lambda / 2;                % Element spacing [m]

theta1_deg = 60;
theta1_rad = deg2rad(theta1_deg);

% Quantization levels to compare
BITS = [1, 2, 3, 4];
N_BITS = length(BITS);

% Reflection magnitude (−1 dB loss)
Gamma = 10^(-1.0/20);

% Scan angle vector
theta_deg = linspace(-90, 90, 3601);
theta_rad = deg2rad(theta_deg);

% Colour palette (colour-blind safe)
COL_IDEAL = [0.2 0.2 0.2];   % dark grey for ideal
COL_BITS  = [0.102 0.435 0.686;   % 1-bit: blue
             0.757 0.153 0.176;   % 2-bit: red
             0.180 0.545 0.341;   % 3-bit: green
             0.851 0.467 0.024];  % 4-bit: amber
LS = {'-', '--', '-.', ':'};      % linestyles

%% ══════════════════════════════════════════════════════════════════════════════
%  2. Element positions (1-D ULA along x, for azimuth pattern)
%     For the full 16×16 UPA the AF is the product of two 1-D AFs;
%     plotting the azimuth cut at elevation = 0 reduces to the 1-D case.
%% ══════════════════════════════════════════════════════════════════════════════
x_n = (0 : N_X-1) * d;          % Element x-positions [m]

%% ══════════════════════════════════════════════════════════════════════════════
%  3. Ideal (continuous phase) array factor
%% ══════════════════════════════════════════════════════════════════════════════
psi_ideal = k_c * x_n * sin(theta1_rad);   % Phase gradient [rad/element]

AF_ideal = zeros(1, length(theta_deg));
for i = 1:length(theta_deg)
    w = Gamma * exp(1j*psi_ideal) .* exp(-1j*k_c*x_n*sin(theta_rad(i)));
    AF_ideal(i) = abs(sum(w)) / N_X;
end
AF_ideal_norm = AF_ideal / max(AF_ideal);
AF_ideal_dB   = 20*log10(AF_ideal_norm + 1e-10);

%% ══════════════════════════════════════════════════════════════════════════════
%  4. Quantised array factors — 1 through 4 bits
%% ══════════════════════════════════════════════════════════════════════════════
AF_q      = zeros(N_BITS, length(theta_deg));
AF_q_dB   = zeros(N_BITS, length(theta_deg));
hpbw      = zeros(1, N_BITS);
sll       = zeros(1, N_BITS);
bpe       = zeros(1, N_BITS);     % Beam pointing error [deg]
peak_gain_dB = zeros(1, N_BITS);  % Peak gain loss vs ideal [dB]

for b = 1:N_BITS
    B      = BITS(b);
    dPsi   = 2*pi / 2^B;          % Phase step [rad]
    psi_q  = round(psi_ideal / dPsi) * dPsi;   % Quantised phase

    for i = 1:length(theta_deg)
        w = Gamma * exp(1j*psi_q) .* exp(-1j*k_c*x_n*sin(theta_rad(i)));
        AF_q(b, i) = abs(sum(w)) / N_X;
    end

    AF_q_norm  = AF_q(b,:) / max(AF_ideal);    % Normalise to ideal peak
    AF_q_dB(b,:) = 20*log10(AF_q_norm + 1e-10);
    peak_gain_dB(b) = 20*log10(max(AF_q(b,:)) / max(AF_ideal));

    % HPBW
    af_n = AF_q(b,:) / max(AF_q(b,:));
    hpbw(b) = compute_hpbw(theta_deg, af_n);

    % SLL
    [~, pk_idx] = max(AF_q(b,:));
    sll(b) = compute_sll(theta_deg, af_n, pk_idx);

    % Beam pointing error
    [~, pk_idx_q] = max(AF_q(b,:));
    bpe(b) = abs(theta_deg(pk_idx_q) - theta1_deg);
end

%% ══════════════════════════════════════════════════════════════════════════════
%  5. Beam squint across M = 12 tones  [Eq. 14]
%% ══════════════════════════════════════════════════════════════════════════════
M = 12;  delta_f = 25e6;
fm_k       = (1:M) * delta_f;               % Tone offsets [Hz]
squint_deg = rad2deg(asin(sin(theta1_rad) * fm_k / FC));  % [deg]

% HPBW of 16-element ULA at theta1
hpbw_ideal_deg = rad2deg(0.886*lambda / (N_X*d*cos(theta1_rad)));

%% ══════════════════════════════════════════════════════════════════════════════
%  6. Console summary table
%% ══════════════════════════════════════════════════════════════════════════════
fprintf('\n%s\n', repmat('═', 1, 62));
fprintf('  R-RIS Beamforming — Quantization Sensitivity Summary\n');
fprintf('  Steering angle θ₁ = %d°,  N = %d×%d = %d elements\n', ...
        theta1_deg, N_X, N_Y, N);
fprintf('%s\n', repmat('─', 1, 62));
fprintf('  Bits | Peak gain (dB) | HPBW (°) | SLL (dB) | BPE (°)\n');
fprintf('%s\n', repmat('─', 1, 62));
for b = 1:N_BITS
    fprintf('    %d  |    %6.2f      |   %5.2f  |   %5.1f  |  %.3f\n', ...
            BITS(b), peak_gain_dB(b), hpbw(b), sll(b), bpe(b));
end
fprintf('%s\n', repmat('─', 1, 62));
fprintf('  Ideal (∞ bits) HPBW = %.2f°\n', hpbw_ideal_deg);
fprintf('  Max beam squint (k=12, 300 MHz): %.3f°  (%.1f%% of HPBW)\n', ...
        squint_deg(end), squint_deg(end)/hpbw_ideal_deg*100);
fprintf('%s\n\n', repmat('═', 1, 62));

%% ══════════════════════════════════════════════════════════════════════════════
%  7. Figure F9 — Two-panel: (a) Cartesian, (b) Polar
%% ══════════════════════════════════════════════════════════════════════════════
fig9 = figure('Units', 'inches', ...
              'Position',      [1 1 7.16 3.0], ...
              'PaperUnits',    'inches', ...
              'PaperSize',     [7.16 3.0], ...
              'PaperPosition', [0 0 7.16 3.0]);
set(fig9, 'DefaultAxesFontName', 'Times New Roman', ...
          'DefaultTextFontName', 'Times New Roman');

%% ── Panel (a): Cartesian dB pattern ────────────────────────────────────────
ax1 = subplot(1, 2, 1);

% Ideal curve
hc(1) = plot(theta_deg, AF_ideal_dB, '-', 'Color', COL_IDEAL, ...
             'LineWidth', 2, 'DisplayName', 'Ideal (continuous)');
hold on;

% Quantised curves
for b = 1:N_BITS
    hc(b+1) = plot(theta_deg, AF_q_dB(b,:), LS{b}, 'Color', COL_BITS(b,:), ...
                   'LineWidth', 2, ...
                   'DisplayName', sprintf('%d-bit', BITS(b)));
end

% Steering angle reference
xline(theta1_deg, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, ...
      'HandleVisibility', 'off');
text(theta1_deg - 11 , -32, sprintf('θ₁=%d°', theta1_deg), ...
     'FontSize', 20, 'Color', [0 0 0]);

% −3 dB reference
yline(-3, ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, ...
      'HandleVisibility', 'off');
text(-88, -3 + 0.6, '−3 dB', 'FontSize', 20, 'Color', [0 0 0]);

xlabel('Azimuth angle (degrees)',FontSize=11);
ylabel('Normalised array factor (dB)',FontSize=11);
xlim([-90 90]);   ylim([-35 5]);
xticks(-90:30:90);  yticks(-30:10:0);
ax1.FontSize   = 20;
ax1.LineWidth  = 0.8;

ax1.TickDir    = 'out';
legend(ax1, 'Location', 'northwest', 'FontSize', 15, 'Box', 'on');
legend.Location = 'none';
legend.Position = [30 0.78 0.15 0.10];

grid on;
% title('(a) Cartesian pattern', 'FontSize', 8, 'FontWeight', 'normal');

%% ── Panel (b): Polar pattern (linear scale) ─────────────────────────────────
ax2 = subplot(1, 2, 2);

% Convert to polar: θ in rad, r = linear AF
% MATLAB's polarplot works with angle [rad] and radius [linear]
hp_ideal = polarplot(deg2rad(theta_deg), AF_ideal_norm, '-', ...
                     'Color', COL_IDEAL, 'LineWidth', 1.4);
hold on;
hp_q = gobjects(N_BITS, 1);
for b = 1:N_BITS
    af_n = AF_q(b,:) / max(AF_ideal);
    hp_q(b) = polarplot(deg2rad(theta_deg), max(af_n, 0), LS{b}, ...
                        'Color', COL_BITS(b,:), 'LineWidth', 2);
end

% Configure polar axes
ax2p = gca;
ax2p.ThetaLim           = [-90 90];
ax2p.ThetaZeroLocation  = 'top';
ax2p.ThetaDir           = 'clockwise';
ax2p.ThetaTick          = -90:30:90;
ax2p.RLim               = [0 1.05];
ax2p.RTick              = [0.25 0.5 0.707 1.0];
ax2p.RTickLabel         = {'-12 dB', '-6 dB', '-3 dB', '0 dB'};
ax2p.FontSize           = 20;
ax2p.FontName           = 'Times New Roman';
ax2p.GridColor          = [0.65 0.65 0.65];
ax2p.GridAlpha          = 0.9;
ax2p.GridLineStyle      = ':';

title('(b) Polar pattern (linear)', 'FontSize', 20, 'FontWeight', 'normal');

%% ── Shared legend on panel (b) ───────────────────────────────────────────────
leg_labels = ['Ideal', arrayfun(@(b) sprintf('%d-bit', b), BITS, ...
              'UniformOutput', false)];
legend([hp_ideal, hp_q'], leg_labels, 'Location', 'southeast', ...
       'FontSize', 20, 'Box', 'on');

%% ── Save ─────────────────────────────────────────────────────────────────────
if ~exist('figures', 'dir'), mkdir('figures'); end
set(fig9, 'PaperPositionMode', 'auto');
print(fig9, 'figures/F9_beamforming_quantization', '-dpdf', '-r300', '-painters');
print(fig9, 'figures/F9_beamforming_quantization', '-dpng', '-r300');
fprintf('  Saved: figures/F9_beamforming_quantization (.pdf + .png)\n\n');

%% ══════════════════════════════════════════════════════════════════════════════
%  8. Export data for Python pipeline
%% ══════════════════════════════════════════════════════════════════════════════
if ~exist('data', 'dir'), mkdir('data'); end
save('data/ris_beamforming_data.mat', ...
     'theta_deg', 'theta_rad', ...
     'AF_ideal', 'AF_ideal_norm', 'AF_ideal_dB', ...
     'AF_q', 'AF_q_dB', ...
     'BITS', 'hpbw', 'sll', 'bpe', 'peak_gain_dB', ...
     'squint_deg', 'fm_k', 'hpbw_ideal_deg', ...
     'lambda', 'k_c', 'N_X', 'N_Y', 'N', 'd', ...
     'theta1_deg', 'theta1_rad', 'Gamma', '-v7');
fprintf('  Exported: data/ris_beamforming_data.mat\n');
fprintf('  Stage 2 (MATLAB — beamforming) complete.\n\n');

%% ══════════════════════════════════════════════════════════════════════════════
%  Local helper functions
%% ══════════════════════════════════════════════════════════════════════════════

function hpbw_deg = compute_hpbw(theta_scan, af_norm)
    % Compute 3-dB beamwidth from normalised (peak = 1) AF.
    half_pwr  = 1 / sqrt(2);
    [~, pk]   = max(af_norm);
    right_idx = find(af_norm(pk:end) < half_pwr, 1);
    left_idx  = find(af_norm(pk:-1:1) < half_pwr, 1);
    if isempty(right_idx) || isempty(left_idx)
        hpbw_deg = NaN;
        return;
    end
    right_th = theta_scan(pk + right_idx - 2);
    left_th  = theta_scan(pk - left_idx  + 1);
    hpbw_deg = right_th - left_th;
end

function sll_dB = compute_sll(theta_scan, af_norm, peak_idx)
    % Compute first SLL [dB] relative to main lobe.
    [pks, locs] = findpeaks(af_norm, 'MinPeakDistance', 5);
    dist = abs(locs - peak_idx);
    sl_pks = pks(dist > max(dist)*0.1);   % exclude main-lobe vicinity
    if isempty(sl_pks)
        sll_dB = -Inf;
    else
        sll_dB = 20*log10(max(sl_pks) / af_norm(peak_idx));
    end
end
