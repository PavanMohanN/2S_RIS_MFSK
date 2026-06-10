% pin_diode_circuit.m — Stage 1: PIN Diode Equivalent Circuit Model
% ==================================================================
% Implements the PIN diode lumped-element model for 28 GHz T-RIS elements.
% Generates publication-ready Figures F4, F5, F6 and exports data for the
% Python pipeline.
%
% Paper equations:
%   Z1(f,state) = Rs + RJ / (1 + j*omega*CJ*RJ)    [Eq. 1]
%   Tn(f,state) = 2*Z0 / (2*Z0 + Z1)               [Eq. 2]
%   Tmean(f)    = (Tn_on + Tn_off) / 2              [Eq. 5]
%   Tsb(f)      = |Tn_on - Tn_off| / 2              [Eq. 8]
%
% Output:
%   figures/F4_transmission_amplitude.pdf/.png
%   figures/F5_phase_response.pdf/.png
%   figures/F6_impedance_magnitude.pdf/.png
%   data/pin_diode_data.mat  (for Python pipeline import via scipy.io.loadmat)
%
% Usage:
%   run pin_diode_circuit.m          (standalone)
%   run_all_matlab.m calls this as a function
%
% Verified against paper: |Tn(on)| = -0.26 dB, |Tn(off)| = -8.02 dB,
%   |Tsb| = -7.17 dB, |Tmean| = -4.46 dB, ΔPhase = -64.5° at 28 GHz

clear; clc; close all;

%% ══════════════════════════════════════════════════════════════════════════════
%  1. Parameters
%% ══════════════════════════════════════════════════════════════════════════════
RS     = 2.0;          % Series resistance [Ω]
CJ     = 25e-15;       % Junction capacitance [F] = 25 fF
RJ_ON  = 1.0;          % Forward-bias junction resistance [Ω]
RJ_OFF = 8e3;          % Reverse-bias junction resistance [Ω] = 8 kΩ
Z0     = 50.0;         % Characteristic impedance [Ω]

FC      = 28e9;        % Carrier frequency [Hz]
F_START = 20e9;        % Sweep start [Hz]
F_STOP  = 35e9;        % Sweep stop  [Hz]
N_PTS   = 1501;        % Frequency points

% Colour palette (colour-blind safe, matches Python config)
COL_ON      = [0.102 0.435 0.686];   % Blue
COL_OFF     = [0.757 0.153 0.176];   % Red
COL_SB      = [0.180 0.545 0.341];   % Green
COL_CARRIER = [0.851 0.467 0.024];   % Amber
COL_GRAY    = [0.333 0.333 0.333];   % Gray

%% ══════════════════════════════════════════════════════════════════════════════
%  2. Frequency sweep
%% ══════════════════════════════════════════════════════════════════════════════
f     = linspace(F_START, F_STOP, N_PTS);   % [Hz]
omega = 2*pi*f;
f_GHz = f / 1e9;

%% ══════════════════════════════════════════════════════════════════════════════
%  3. Impedance Z1(f, state)  [Eq. 1]
%% ══════════════════════════════════════════════════════════════════════════════
% ON state:  RJ_ON is small → near-short, dominated by Rs
Z1_on  = RS + RJ_ON  ./ (1 + 1j*omega*CJ*RJ_ON);

% OFF state: RJ_OFF large → capacitive behaviour at mmWave
Z1_off = RS + RJ_OFF ./ (1 + 1j*omega*CJ*RJ_OFF);

%% ══════════════════════════════════════════════════════════════════════════════
%  4. Transmission coefficients  [Eq. 2]
%% ══════════════════════════════════════════════════════════════════════════════
Tn_on  = (2*Z0) ./ (2*Z0 + Z1_on);
Tn_off = (2*Z0) ./ (2*Z0 + Z1_off);

%% ══════════════════════════════════════════════════════════════════════════════
%  5. Derived waveform quantities  [Eq. 5, 8]
%% ══════════════════════════════════════════════════════════════════════════════
Tsb   = abs(Tn_on - Tn_off) / 2;          % Sideband conversion coeff
Tmean = abs((Tn_on + Tn_off) / 2);        % Carrier feedthrough amplitude

% dB conversion
Ton_dB   = 20*log10(abs(Tn_on));
Toff_dB  = 20*log10(abs(Tn_off));
Tsb_dB   = 20*log10(Tsb);
Tmean_dB = 20*log10(Tmean);

% Phase [degrees]
Ph_on  = angle(Tn_on)  * (180/pi);
Ph_off = angle(Tn_off) * (180/pi);

%% ══════════════════════════════════════════════════════════════════════════════
%  6. Operating-point summary at fc
%% ══════════════════════════════════════════════════════════════════════════════
[~, idx_fc] = min(abs(f - FC));

fprintf('\n%s\n', repmat('═', 1, 58));
fprintf('  PIN Diode — Operating point at fc = %.0f GHz\n', FC/1e9);
fprintf('%s\n', repmat('═', 1, 58));
fprintf('  |Tn(on)|          = %6.2f dB\n',  Ton_dB(idx_fc));
fprintf('  |Tn(off)|         = %6.2f dB\n',  Toff_dB(idx_fc));
fprintf('  |Tsb|             = %6.2f dB   <- sideband amplitude\n', Tsb_dB(idx_fc));
fprintf('  |Tmean|           = %6.2f dB   <- carrier feedthrough\n', Tmean_dB(idx_fc));
fprintf('  angle Tn(on)      = %6.1f deg\n', Ph_on(idx_fc));
fprintf('  angle Tn(off)     = %6.1f deg\n', Ph_off(idx_fc));
fprintf('  Delta Phase       = %6.1f deg\n', Ph_off(idx_fc)-Ph_on(idx_fc));
fprintf('  |Z1(on)|          = %6.2f Ohm  (<<Z0=%g Ohm)\n', abs(Z1_on(idx_fc)), Z0);
fprintf('  |Z1(off)|         = %6.1f Ohm\n', abs(Z1_off(idx_fc)));
fprintf('  Tsb variation     = %6.2f dB over 20-35 GHz\n', ...
        max(Tsb_dB) - min(Tsb_dB));
fprintf('%s\n\n', repmat('═', 1, 58));

%% ══════════════════════════════════════════════════════════════════════════════
%  7. Figure F4 — Transmission amplitude vs frequency
%% ══════════════════════════════════════════════════════════════════════════════
fig4 = ieee_figure(3.5, 2.6);

% Main curves
hp(1) = plot(f_GHz, Ton_dB,   '-',  'Color', COL_ON,      'LineWidth', 1.5, ...
             'DisplayName', '$|T_n(f,\mathrm{on})|$');   hold on;
hp(2) = plot(f_GHz, Toff_dB,  '--', 'Color', COL_OFF,     'LineWidth', 1.5, ...
             'DisplayName', '$|T_n(f,\mathrm{off})|$');
hp(3) = plot(f_GHz, Tsb_dB,   '-.', 'Color', COL_SB,      'LineWidth', 1.5, ...
             'DisplayName', '$|T_{\mathrm{sb}}(f)|$');
hp(4) = plot(f_GHz, Tmean_dB, ':',  'Color', COL_CARRIER, 'LineWidth', 1.5, ...
             'DisplayName', '$|T_{\mathrm{mean}}(f)|$');

% fc reference
xline(28, ':', 'Color', COL_GRAY, 'LineWidth', 0.7, 'Alpha', 0.7, ...
      'HandleVisibility', 'off');

% Operating-point markers
markers = {'o', 's', '^', 'd'};
colors_op = {COL_ON, COL_OFF, COL_SB, COL_CARRIER};
keys_dB   = {Ton_dB, Toff_dB, Tsb_dB, Tmean_dB};
for i = 1:4
    plot(28, keys_dB{i}(idx_fc), markers{i}, 'Color', colors_op{i}, ...
         'MarkerSize', 5, 'MarkerFaceColor', colors_op{i}, 'HandleVisibility', 'off');
end

% Axis annotation
text(27.7, -20.8, '$f_c\!=\!28$ GHz', 'Interpreter', 'latex', ...
     'FontSize', 6.5, 'Color', COL_GRAY, 'HorizontalAlignment', 'center');

xlabel('Frequency (GHz)', 'FontSize', 9);
ylabel('Transmission coefficient (dB)', 'FontSize', 9);
xlim([20 35]);   ylim([-22 2]);
xticks(20:5:35); yticks(-20:5:2);
ax = gca;
ax.XMinorTick = 'on';   ax.YMinorTick = 'on';
legend('Location', 'southwest', 'NumColumns', 2, 'FontSize', 7, ...
       'Interpreter', 'latex', 'Box', 'on');
grid on;

save_ieee_fig(fig4, 'figures/F4_transmission_amplitude');

%% ══════════════════════════════════════════════════════════════════════════════
%  8. Figure F5 — Phase response vs frequency
%% ══════════════════════════════════════════════════════════════════════════════
fig5 = ieee_figure(3.5, 2.6);

plot(f_GHz, Ph_on,  '-',  'Color', COL_ON,  'LineWidth', 1.5, ...
     'DisplayName', '$\angle T_n(f,\mathrm{on})$');    hold on;
plot(f_GHz, Ph_off, '--', 'Color', COL_OFF, 'LineWidth', 1.5, ...
     'DisplayName', '$\angle T_n(f,\mathrm{off})$');

% fc reference
xline(28, ':', 'Color', COL_GRAY, 'LineWidth', 0.7, 'Alpha', 0.7, ...
      'HandleVisibility', 'off');

% Operating-point markers
plot(28, Ph_on(idx_fc),  'o', 'Color', COL_ON,  'MarkerSize', 5, ...
     'MarkerFaceColor', COL_ON,  'HandleVisibility', 'off');
plot(28, Ph_off(idx_fc), 's', 'Color', COL_OFF, 'MarkerSize', 5, ...
     'MarkerFaceColor', COL_OFF, 'HandleVisibility', 'off');

% Double-arrow for phase difference
delta_ph = Ph_off(idx_fc) - Ph_on(idx_fc);
ph_mid   = (Ph_on(idx_fc) + Ph_off(idx_fc)) / 2;
annotation_double_arrow(fig5, 28, Ph_on(idx_fc), Ph_off(idx_fc), ...
                        [F_START/1e9, F_STOP/1e9], ylim_est());
text(28.35, ph_mid, sprintf('$\\Delta\\phi = %.1f^\\circ$', delta_ph), ...
     'Interpreter', 'latex', 'FontSize', 7, 'VerticalAlignment', 'middle');

% fc label
text(27.7, min(Ph_on(idx_fc), Ph_off(idx_fc)) - 8, '$f_c$', ...
     'Interpreter', 'latex', 'FontSize', 7, 'Color', COL_GRAY, ...
     'HorizontalAlignment', 'center');

xlabel('Frequency (GHz)', 'FontSize', 9);
ylabel('Phase (degrees)', 'FontSize', 9);
xlim([20 35]);
xticks(20:5:35);
ax = gca;
ax.XMinorTick = 'on';   ax.YMinorTick = 'on';
legend('Location', 'northeast', 'FontSize', 7, 'Interpreter', 'latex');
grid on;

save_ieee_fig(fig5, 'figures/F5_phase_response');

%% ══════════════════════════════════════════════════════════════════════════════
%  9. Figure F6 — Impedance magnitude vs frequency (log-y)
%% ══════════════════════════════════════════════════════════════════════════════
fig6 = ieee_figure(3.5, 2.6);

semilogy(f_GHz, abs(Z1_on),  '-',  'Color', COL_ON,   'LineWidth', 1.5, ...
         'DisplayName', '$|Z_1(f,\mathrm{on})|$');    hold on;
semilogy(f_GHz, abs(Z1_off), '--', 'Color', COL_OFF,  'LineWidth', 1.5, ...
         'DisplayName', '$|Z_1(f,\mathrm{off})|$');

% Z0 reference
yline(50, ':', 'Color', COL_GRAY, 'LineWidth', 0.9, ...
      'DisplayName', sprintf('$Z_0 = %g\\,\\Omega$', Z0));

% fc reference
xline(28, ':', 'Color', COL_GRAY, 'LineWidth', 0.7, 'Alpha', 0.7, ...
      'HandleVisibility', 'off');

% Annotate values at fc
plot(28, abs(Z1_on(idx_fc)),  'o', 'Color', COL_ON,  'MarkerSize', 5, ...
     'MarkerFaceColor', COL_ON,  'HandleVisibility', 'off');
plot(28, abs(Z1_off(idx_fc)), 's', 'Color', COL_OFF, 'MarkerSize', 5, ...
     'MarkerFaceColor', COL_OFF, 'HandleVisibility', 'off');

text(28.4, abs(Z1_on(idx_fc))*1.6, ...
     sprintf('%.1f $\\Omega$', abs(Z1_on(idx_fc))),  ...
     'Interpreter', 'latex', 'FontSize', 6.5, 'Color', COL_ON);
text(28.4, abs(Z1_off(idx_fc))*0.7, ...
     sprintf('%.0f $\\Omega$', abs(Z1_off(idx_fc))), ...
     'Interpreter', 'latex', 'FontSize', 6.5, 'Color', COL_OFF);

xlabel('Frequency (GHz)', 'FontSize', 9);
ylabel('Impedance magnitude $(\Omega)$', 'Interpreter', 'latex', 'FontSize', 9);
xlim([20 35]);
xticks(20:5:35);
ax = gca;
ax.XMinorTick = 'on';
legend('Location', 'northeast', 'FontSize', 7, 'Interpreter', 'latex');
grid on;

save_ieee_fig(fig6, 'figures/F6_impedance_magnitude');

%% ══════════════════════════════════════════════════════════════════════════════
%  10. Export data for Python pipeline (scipy.io.loadmat compatible)
%% ══════════════════════════════════════════════════════════════════════════════
if ~exist('data', 'dir'), mkdir('data'); end

save('data/pin_diode_data.mat', ...
     'f', 'f_GHz', ...
     'Z1_on', 'Z1_off', ...
     'Tn_on', 'Tn_off', ...
     'Tsb', 'Tmean', ...
     'Ton_dB', 'Toff_dB', 'Tsb_dB', 'Tmean_dB', ...
     'Ph_on', 'Ph_off', ...
     'RS', 'CJ', 'RJ_ON', 'RJ_OFF', 'Z0', 'FC', ...
     'idx_fc', '-v7');   % -v7 for scipy.io.loadmat compatibility

fprintf('  Exported: data/pin_diode_data.mat\n\n');
fprintf('  Stage 1 (MATLAB) complete.\n\n');

%% ══════════════════════════════════════════════════════════════════════════════
%  Local helper functions
%% ══════════════════════════════════════════════════════════════════════════════

function fig = ieee_figure(w_in, h_in)
    % Create figure pre-configured for IEEE TVT publication.
    fig = figure('Units', 'inches', ...
                 'Position',      [1 1 w_in h_in], ...
                 'PaperUnits',    'inches', ...
                 'PaperSize',     [w_in h_in], ...
                 'PaperPosition', [0 0 w_in h_in]);
    set(fig, ...
        'DefaultAxesFontName',  'Times New Roman', ...
        'DefaultAxesFontSize',  8, ...
        'DefaultTextFontName',  'Times New Roman', ...
        'DefaultAxesLineWidth', 0.8, ...
        'DefaultAxesTickDir',   'out', ...
        'DefaultAxesBox',       'on');
    axes('Parent', fig);
end

function save_ieee_fig(fig, base_path)
    % Save figure as PDF (vector) and PNG (300 DPI raster).
    dir_part = fileparts(base_path);
    if ~isempty(dir_part) && ~exist(dir_part, 'dir')
        mkdir(dir_part);
    end
    % Ensure tight bounding box
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, base_path, '-dpdf',  '-r300', '-painters');
    print(fig, base_path, '-dpng',  '-r300');
    fprintf('  Saved: %s (.pdf + .png)\n', base_path);
end

function yl = ylim_est()
    % Return current y-axis limits for use before ylim is set.
    ax = gca;
    yl = ax.YLim;
    if all(yl == [0 1])
        yl = [-180 0];   % sensible phase default before plot data is rendered
    end
end

function annotation_double_arrow(fig, x_data, y_lo, y_hi, x_data_lim, y_data_lim)
    % Draw a double-headed arrow in data coordinates using annotations.
    % Maps data coords → normalised figure coords.
    ax = gca;
    drawnow;  % ensure axes limits are finalised
    xl = ax.XLim;   yl = ax.YLim;
    axpos = ax.Position;   % [left bottom width height] in normalised fig units

    norm_x = axpos(1) + (x_data - xl(1)) / (xl(2)-xl(1)) * axpos(3);
    norm_y_lo = axpos(2) + (y_lo - yl(1)) / (yl(2)-yl(1)) * axpos(4);
    norm_y_hi = axpos(2) + (y_hi - yl(1)) / (yl(2)-yl(1)) * axpos(4);

    annotation(fig, 'doublearrow', ...
               [norm_x norm_x], [norm_y_lo norm_y_hi], ...
               'HeadStyle', 'vback2', 'HeadLength', 5, 'HeadWidth', 5, ...
               'Color', 'black', 'LineWidth', 0.8);
end
