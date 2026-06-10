% run_all_matlab.m  —  Stage 8b: Master MATLAB Runner
% =====================================================
% Executes all four MATLAB simulation scripts in dependency order
% and reports timing for each.
%
% Usage (from the ris_mfsk_sim/matlab/ directory):
%   matlab -batch "run('run_all_matlab.m')"
%   matlab -nodisplay -r "run('run_all_matlab.m'); exit"
%
% Or from the ris_mfsk_sim/ root:
%   matlab -batch "cd matlab; run('run_all_matlab.m')"
%
% Requires MATLAB R2021a or later (uses 'matlab -batch' syntax).
% All scripts write outputs to:
%   ../figures/   — PDF and PNG figures
%   ../data/      — .mat data files

clear; clc;
fprintf('\n%s\n', repmat('═', 1, 62));
fprintf('  RIS M-FSK Sensing — MATLAB Simulation Suite\n');
fprintf('  IEEE Transactions on Vehicular Technology\n');
fprintf('%s\n\n', repmat('═', 1, 62));

% ─── Execution order and script registry ────────────────────────────────────
SCRIPTS = {
    'pin_diode_circuit',  'Phase 1: PIN diode circuit model      (F_circ_*)';
    'ris_beamforming',    'Phase 2: RIS beam pattern analysis     (F_beam_*)';
    'mfsk_spectrum',      'Phase 2: M-FSK spectrum (MATLAB)       (F_spec_*)';
    'near_field_coupling','Phase 3: Near-field coupling analysis  (F_NFC_*)';
};

n_scripts = size(SCRIPTS, 1);
t_start_all = tic;

% Change to the matlab/ subdirectory if not already there
script_dir = fileparts(mfilename('fullpath'));
if ~isempty(script_dir)
    cd(script_dir);
end

% Ensure output directories exist relative to parent
if ~exist('../figures', 'dir'), mkdir('../figures'); end
if ~exist('../data',    'dir'), mkdir('../data');    end
if ~exist('../tables',  'dir'), mkdir('../tables');  end

% ─── Run each script ─────────────────────────────────────────────────────────
results = struct('script', {}, 'status', {}, 'elapsed_s', {}, 'error', {});

for k = 1 : n_scripts
    script_name = SCRIPTS{k, 1};
    description = SCRIPTS{k, 2};

    fprintf('%s\n', repmat('─', 1, 60));
    fprintf('  [%d/%d] %s\n', k, n_scripts, description);
    fprintf('%s\n', repmat('─', 1, 60));

    t0 = tic;
    err_msg = '';
    ok = false;

    try
        run(script_name);
        ok = true;
    catch ME
        err_msg = ME.message;
        fprintf('\n  [ERROR] %s failed:\n  %s\n\n', script_name, err_msg);
    end

    elapsed = toc(t0);
    results(k).script    = script_name;
    results(k).status    = ok;
    results(k).elapsed_s = elapsed;
    results(k).error     = err_msg;

    if ok
        fprintf('\n  ↳ %s finished in %.1f s\n\n', script_name, elapsed);
    end

    close all;   % close any figures the script left open
end

% ─── Summary report ──────────────────────────────────────────────────────────
total_elapsed = toc(t_start_all);

fprintf('\n%s\n', repmat('═', 1, 62));
fprintf('  MATLAB Runner — Summary\n');
fprintf('%s\n', repmat('═', 1, 62));
fprintf('  %-28s  %-8s  %s\n', 'Script', 'Status', 'Time (s)');
fprintf('  %s\n', repmat('─', 1, 58));

n_ok   = 0;
n_fail = 0;
for k = 1 : n_scripts
    r = results(k);
    if r.status
        status_str = 'OK';
        n_ok = n_ok + 1;
    else
        status_str = 'FAILED';
        n_fail = n_fail + 1;
    end
    fprintf('  %-28s  %-8s  %.1f\n', r.script, status_str, r.elapsed_s);
end

fprintf('  %s\n', repmat('─', 1, 58));
fprintf('  %d / %d scripts succeeded  |  Total: %.1f s\n', ...
        n_ok, n_scripts, total_elapsed);
fprintf('%s\n\n', repmat('═', 1, 62));

% Save run log
log_path = fullfile('..', 'data', 'matlab_run_log.mat');
save(log_path, 'results', 'total_elapsed', '-v7');
fprintf('  Run log saved: %s\n\n', log_path);

if n_fail > 0
    warning('RIS:matlab_runner', '%d script(s) failed — check error messages above.', n_fail);
end
