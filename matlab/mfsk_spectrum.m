% mfsk_spectrum.m — Stage 2: M-FSK Spectrum and Time-Frequency Spectrogram
% =========================================================================
% Simulates the M-FSK waveform produced by the T-RIS via DDS switching.
% Generates the analytical spectral line plot (F7) and the time-frequency
% spectrogram (F8).
%
% Paper equations:
%   s_n(t) = Tmean·cos(2πfc·t)
%           + Tsb/2·cos[2π(fc+fm,k)·t]    <- desired upper sideband
%           + Tsb/2·cos[2π(fc-fm,k)·t]    <- image  lower sideband  [Eq. 7]
%
% Requires pin_diode_circuit.m to have been run first (loads Tsb, Tmean).
% If data/pin_diode_data.mat does not exist, uses paper values directly.
%
% Generates:
%   figures/F7_mfsk_spectrum.pdf/.png
%   figures/F8_mfsk_spectrogram.pdf/.png
%
% Exports:
%   data/mfsk_waveform_data.mat
%
% Usage:
%   run mfsk_spectrum.m          (standalone)
%   Called by run_all_matlab.m

clear; clc; close all;

%% ══════════════════════════════════════════════════════════════════════════════
%  1. System parameters
%% ══════════════════════════════════════════════════════════════════════════════
FC      = 28e9;         % Carrier frequency [Hz]
M       = 12;           % Number of FSK tones
DELTA_F = 25e6;         % Tone spacing [Hz]
FM_K    = (1:M)*DELTA_F;% Tone offsets [Hz]: 25, 50, …, 300 MHz
TSYM    = 1.40e-6;      % Symbol period [s]

% BPF passband
BPF_LO  = FC + 12.5e6;    % [Hz]
BPF_HI  = FC + 312.5e6;   % [Hz]

% PIN diode values at fc (from paper / pin_diode_circuit.m)
if exist('data/pin_diode_data.mat', 'file')
    P = load('data/pin_diode_data.mat');
    [~, idx_fc] = min(abs(P.f - FC));
    Tsb_dB   = P.Tsb_dB(idx_fc);
    Tmean_dB = P.Tmean_dB(idx_fc);
    fprintf('  Loaded PIN diode data from data/pin_diode_data.mat\n');
else
    % Paper-verified values
    Tsb_dB   = -7.17;
    Tmean_dB = -4.46;
    fprintf('  Using paper values: |Tsb| = %.2f dB, |Tmean| = %.2f dB\n', ...
            Tsb_dB, Tmean_dB);
end

Tsb   = 10^(Tsb_dB   / 20);
Tmean = 10^(Tmean_dB / 20);

% Colour palette
COL_CARRIER = [0.851 0.467 0.024];   % Amber
COL_SB      = [0.180 0.545 0.341];   % Green
COL_OFF     = [0.757 0.153 0.176];   % Red
COL_GRAY    = [0.333 0.333 0.333];   % Gray

fc_GHz    = FC / 1e9;
fu_GHz    = (FC + FM_K) / 1e9;       % Upper sideband frequencies [GHz]
fl_GHz    = (FC - FM_K) / 1e9;       % Lower sideband (image) frequencies [GHz]
bpf_lo_GHz= BPF_LO / 1e9;
bpf_hi_GHz= BPF_HI / 1e9;

%% ══════════════════════════════════════════════════════════════════════════════
%  2. Figure F7 — Analytical M-FSK Spectrum
%% ══════════════════════════════════════════════════════════════════════════════
fig7 = ieee_figure(3.5, 2.6);

%% BPF shading
fill([bpf_lo_GHz bpf_hi_GHz bpf_hi_GHz bpf_lo_GHz], ...
     [-35 -35 5 5], COL_SB, ...
     'FaceAlpha', 0.09, 'EdgeColor', 'none', ...
     'DisplayName', 'BPF passband');
hold on;
plot([bpf_lo_GHz bpf_lo_GHz], [-35 5], ':', 'Color', COL_SB, ...
     'LineWidth', 0.9, 'HandleVisibility', 'off');
plot([bpf_hi_GHz bpf_hi_GHz], [-35 5], ':', 'Color', COL_SB, ...
     'LineWidth', 0.9, 'HandleVisibility', 'off');

%% Carrier feedthrough
stem_plot(fc_GHz, Tmean_dB, 'd', COL_CARRIER, 1.5, 5, ...
          '$T_{\mathrm{mean}}$ (carrier)');

%% Upper sidebands (desired — inside BPF)
for i = 1:M
    if i == 1
        stem_plot(fu_GHz(i), Tsb_dB, '^', COL_SB, 1.3, 4.5, ...
                  'Upper sideband (desired)');
    else
        stem_plot(fu_GHz(i), Tsb_dB, '^', COL_SB, 1.3, 4.5, '');
    end
end

%% Lower sidebands (image — outside BPF)
for i = 1:M
    if i == 1
        stem_plot(fl_GHz(i), Tsb_dB, 'v', COL_OFF, 1.2, 4.5, ...
                  'Lower sideband (image)', '--');
    else
        stem_plot(fl_GHz(i), Tsb_dB, 'v', COL_OFF, 1.2, 4.5, '', '--');
    end
end

%% fc reference
plot([fc_GHz fc_GHz], [-35 0], ':', 'Color', COL_GRAY, 'LineWidth', 0.7, ...
     'HandleVisibility', 'off');
text(fc_GHz, -33.5, '$f_c\!=\!28$ GHz', 'Interpreter', 'latex', ...
     'FontSize', 6.5, 'Color', COL_GRAY, 'HorizontalAlignment', 'center');

%% Tone labels
text(fu_GHz(1),   Tsb_dB + 1.8, '+25 MHz',  'FontSize', 5.5, ...
     'HorizontalAlignment', 'center', 'Color', COL_GRAY);
text(fu_GHz(end), Tsb_dB + 1.8, '+300 MHz', 'FontSize', 5.5, ...
     'HorizontalAlignment', 'center', 'Color', COL_GRAY);
text(fl_GHz(1),   Tsb_dB + 1.8, '−25 MHz',  'FontSize', 5.5, ...
     'HorizontalAlignment', 'center', 'Color', COL_GRAY);

xlabel('Frequency (GHz)');
ylabel('Normalised amplitude (dB)');
xlim([fc_GHz - 0.44, fc_GHz + 0.44]);
ylim([-35 5]);

% Custom x-ticks at fc ± k·0.1 GHz
x_ticks = fc_GHz + (-0.4:0.1:0.4);
xticks(x_ticks);
xticklabels(arrayfun(@(x) sprintf('%.1f', x), x_ticks, 'UniformOutput', false));
ax7 = gca;
ax7.XTickLabelRotation = 30;
yticks(-35:5:5);

legend('Location', 'northeast', 'FontSize', 6.5, 'Interpreter', 'latex', ...
       'Box', 'on');
grid on;
ax7.FontSize  = 8;
ax7.LineWidth = 0.8;

save_ieee_fig(fig7, 'figures/F7_mfsk_spectrum');

%% ══════════════════════════════════════════════════════════════════════════════
%  3. Figure F8 — Time-frequency spectrogram
%     Generate baseband M-FSK signal (2 full sweeps) and compute STFT.
%% ══════════════════════════════════════════════════════════════════════════════

%% Baseband signal parameters
T_SEGS = 2 * M;                        % Symbol periods to generate (24)
fs_bb  = 4 * (M + 2) * DELTA_F;        % Baseband sampling rate ≈ 1.4 GHz
N_tot  = round(T_SEGS * TSYM * fs_bb);
t_vec  = (0 : N_tot-1) / fs_bb;        % Time vector [s]

sig = zeros(1, N_tot);
sym_t_us  = zeros(1, T_SEGS);
sym_fm_MHz= zeros(1, T_SEGS);

for seg = 0 : T_SEGS-1
    k_idx = mod(seg, M) + 1;
    fm_k  = FM_K(k_idx);
    t_s   = seg * TSYM;
    t_e   = (seg+1) * TSYM;
    mask  = (t_vec >= t_s) & (t_vec < t_e);
    n_seg = sum(mask);
    if n_seg < 4, continue; end
    win = hanning(n_seg)';
    % Complex baseband: upper at +fm, image at -fm
    t_m = t_vec(mask);
    sig(mask) = (exp( 1j*2*pi*fm_k*t_m) + ...
                 exp(-1j*2*pi*fm_k*t_m)) .* win;
    sym_t_us(seg+1)   = t_s * 1e6;
    sym_fm_MHz(seg+1) = fm_k / 1e6;
end

%% STFT / spectrogram
n_per_seg  = max(64, round(fs_bb * TSYM * 0.80));
n_overlap  = floor(n_per_seg / 2);
win_stft   = hanning(n_per_seg);

[S, F_s, T_s] = spectrogram(sig, win_stft, n_overlap, n_per_seg, fs_bb, 'centered');

S_dB  = 10*log10(abs(S).^2 + 1e-30);
F_MHz = F_s / 1e6;
T_us  = T_s * 1e6;

vmin = prctile(S_dB(:), 68);
vmax = max(S_dB(:));

%% Plot
fig8 = ieee_figure(3.5, 2.6);

imagesc(T_us, F_MHz, S_dB);
axis xy;
colormap(hot);

cb = colorbar;
cb.Label.String   = 'PSD (dB)';
cb.Label.FontSize = 7;
cb.FontSize       = 6;
clim([vmin vmax]);

hold on;

%% Annotate tone steps (first sweep only)
for k = 1:M
    t_mid = sym_t_us(k) + TSYM*1e6*0.5;
    if t_mid < max(T_us)
        text(t_mid, sym_fm_MHz(k)+15, sprintf('%d', round(sym_fm_MHz(k))), ...
             'Color', 'white', 'FontSize', 5.2, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center', 'FontName', 'Times New Roman');
    end
end

%% DC reference and BPF edges
yline(0, '--', 'Color', [1 1 1 0.45], 'LineWidth', 0.6, 'HandleVisibility', 'off');
bpf_bb_lo = (BPF_LO - FC) / 1e6;
bpf_bb_hi = (BPF_HI - FC) / 1e6;
yline(bpf_bb_lo, ':', 'Color', [0 1 1 0.55], 'LineWidth', 0.7, ...
      'HandleVisibility', 'off');
yline(bpf_bb_hi, ':', 'Color', [0 1 1 0.55], 'LineWidth', 0.7, ...
      'HandleVisibility', 'off');
text(max(T_us)*0.97, bpf_bb_lo + 6, 'BPF', 'Color', 'cyan', ...
     'FontSize', 5.5, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
text(max(T_us)*0.97, bpf_bb_hi - 6, 'BPF', 'Color', 'cyan', ...
     'FontSize', 5.5, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'top');

xlabel('Time (\mus)');
ylabel('Baseband frequency (MHz)');
ylim([-340 340]);
yticks(-300:100:300);
ax8 = gca;
ax8.FontSize  = 8;
ax8.LineWidth = 0.8;

save_ieee_fig(fig8, 'figures/F8_mfsk_spectrogram');

%% ══════════════════════════════════════════════════════════════════════════════
%  4. Export waveform data
%% ══════════════════════════════════════════════════════════════════════════════
if ~exist('data', 'dir'), mkdir('data'); end
save('data/mfsk_waveform_data.mat', ...
     'FC', 'M', 'DELTA_F', 'FM_K', 'TSYM', ...
     'Tsb', 'Tmean', 'Tsb_dB', 'Tmean_dB', ...
     'BPF_LO', 'BPF_HI', ...
     'fu_GHz', 'fl_GHz', 'fc_GHz', '-v7');
fprintf('  Exported: data/mfsk_waveform_data.mat\n');
fprintf('  Stage 2 (MATLAB — waveform) complete.\n\n');

%% ══════════════════════════════════════════════════════════════════════════════
%  Local helper functions
%% ══════════════════════════════════════════════════════════════════════════════

function stem_plot(x, y, marker, col, lw, ms, lbl, ls)
    % Draw a single stem line with optional linestyle and legend label.
    if nargin < 8, ls = '-'; end
    line([x x], [-35 y], 'Color', col, 'LineWidth', lw, 'LineStyle', ls, ...
         'HandleVisibility', 'off');
    if ~isempty(lbl)
        plot(x, y, marker, 'Color', col, 'MarkerFaceColor', col, ...
             'MarkerSize', ms, 'DisplayName', lbl);
    else
        plot(x, y, marker, 'Color', col, 'MarkerFaceColor', col, ...
             'MarkerSize', ms, 'HandleVisibility', 'off');
    end
end

function fig = ieee_figure(w_in, h_in)
    fig = figure('Units', 'inches', ...
                 'Position',      [1 1 w_in h_in], ...
                 'PaperUnits',    'inches', ...
                 'PaperSize',     [w_in h_in], ...
                 'PaperPosition', [0 0 w_in h_in]);
    set(fig, 'DefaultAxesFontName', 'Times New Roman', ...
             'DefaultAxesFontSize', 8, ...
             'DefaultTextFontName', 'Times New Roman', ...
             'DefaultAxesLineWidth', 0.8, ...
             'DefaultAxesTickDir', 'out', ...
             'DefaultAxesBox', 'on');
    axes('Parent', fig);
end

function save_ieee_fig(fig, base_path)
    dir_part = fileparts(base_path);
    if ~isempty(dir_part) && ~exist(dir_part, 'dir'), mkdir(dir_part); end
    set(fig, 'PaperPositionMode', 'auto');
    print(fig, base_path, '-dpdf', '-r300', '-painters');
    print(fig, base_path, '-dpng', '-r300');
    fprintf('  Saved: %s (.pdf + .png)\n', base_path);
end
