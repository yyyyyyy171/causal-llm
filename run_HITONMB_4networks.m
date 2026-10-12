clear;
clc;

% ============================================================
% HITON-MB baseline
% ASIA / CHILD / INSURANCE / HAILFINDER
% n = 100 / 500 / 1000 / 5000
% ============================================================

cd('E:\Causal-Learner-main\Causal-Learner-main\Code');
addpath(genpath(pwd));

% ------------------------------------------------------------
% 路径
% ------------------------------------------------------------

base_dir = 'E:\Causal-LLM\data';
result_dir = 'E:\Causal-LLM\results';

if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

% ------------------------------------------------------------
% 网络和参数
% ------------------------------------------------------------

networks = {'ASIA', 'CHILD', 'INSURANCE', 'HAILFINDER'};
sample_sizes = [100 500 1000 5000];

alpha = 0.05;
maxK = 3;

% ------------------------------------------------------------
% 结果容器
% ------------------------------------------------------------

result_network = {};
result_sample = [];
result_target = [];
result_pred = {};
result_truth = {};
result_precision = [];
result_recall = [];
result_f1 = [];
result_cits = [];
result_time = [];

% ============================================================
% 网络循环
% ============================================================

for ni = 1:length(networks)

    net = networks{ni};

    fprintf('\n\n');
    fprintf('============================================\n');
    fprintf('NETWORK: %s\n', net);
    fprintf('============================================\n');

    net_dir = fullfile(base_dir, net);

    % --------------------------------------------------------
    % 读取 Ground Truth MB 矩阵
    % --------------------------------------------------------

    truth_file = fullfile(net_dir, [net '_true_MB_matrix.txt']);
    true_MB = readmatrix(truth_file);

    [p_truth, p_truth2] = size(true_MB);

    if p_truth ~= p_truth2
        error('%s 的 Ground Truth MB matrix 不是方阵。', net);
    end

    p = p_truth;

    fprintf('变量数 = %d\n', p);

    % ========================================================
    % 样本量循环
    % ========================================================

    for si = 1:length(sample_sizes)

        n = sample_sizes(si);

        fprintf('\n');
        fprintf('--------------------------------------------\n');
        fprintf('%s | n = %d\n', net, n);
        fprintf('--------------------------------------------\n');

        % ----------------------------------------------------
        % 读取数据
        % ----------------------------------------------------

        data_file = fullfile( ...
            net_dir, ...
            sprintf('%s_n%d_num.txt', net, n) ...
        );

        data = readmatrix(data_file);

        [samples, p_data] = size(data);

        fprintf('samples = %d\n', samples);
        fprintf('variables = %d\n', p_data);

        if p_data ~= p
            error('%s n=%d: 数据变量数与 Ground Truth 不一致。', net, n);
        end

        % ----------------------------------------------------
        % 计算每个变量实际出现的状态数
        % ----------------------------------------------------

        ns_original = zeros(1, p);

        for j = 1:p
            ns_original(j) = numel(unique(data(:, j)));
        end

        % 至少两个实际状态
        valid_vars = ns_original >= 2;

        % 原始变量编号
        original_indices = find(valid_vars);

        % 算法实际使用的数据
        data_work = data(:, valid_vars);

        % 实际状态数
        ns = ns_original(valid_vars);

        % 实际变量数
        p_work = size(data_work, 2);

        fprintf('有效变量 = %d / %d\n', p_work, p);

        % ====================================================
        % target 循环
        % ====================================================

        for target_local = 1:p_work

            target_original = original_indices(target_local);

            fprintf('[%d/%d] target = %d\n', ...
                target_local, p_work, target_original);

            try

                % ------------------------------------------------
                % HITON-MB
                % ------------------------------------------------

                [MB_local, test, runtime] = HITONMB_G2( ...
                    data_work, ...
                    target_local, ...
                    alpha, ...
                    ns, ...
                    p_work, ...
                    maxK);

                MB_local = unique(MB_local);

                % ------------------------------------------------
                % 局部索引 -> 原始变量索引
                % ------------------------------------------------

                if isempty(MB_local)
                    MB_original = [];
                else
                    MB_original = original_indices(MB_local);
                end

                % ------------------------------------------------
                % Ground Truth
                % ------------------------------------------------

                truth_original = find(true_MB(target_original, :) > 0);

                % ------------------------------------------------
                % Precision / Recall / F1
                % ------------------------------------------------

                TP = length(intersect(MB_original, truth_original));
                FP = length(setdiff(MB_original, truth_original));
                FN = length(setdiff(truth_original, MB_original));

                if TP + FP == 0
                    Precision = 0;
                else
                    Precision = TP / (TP + FP);
                end

                if TP + FN == 0
                    Recall = 0;
                else
                    Recall = TP / (TP + FN);
                end

                if Precision + Recall == 0
                    F1 = 0;
                else
                    F1 = 2 * Precision * Recall / (Precision + Recall);
                end

            catch ME

                fprintf('HITON-MB失败: target=%d\n', target_original);
                fprintf('%s\n', ME.message);

                MB_original = [];
                truth_original = find(true_MB(target_original, :) > 0);

                Precision = NaN;
                Recall = NaN;
                F1 = NaN;
                test = NaN;
                runtime = NaN;

            end

            % ----------------------------------------------------
            % 保存结果
            % ----------------------------------------------------

            result_network{end+1, 1} = net;
            result_sample(end+1, 1) = n;
            result_target(end+1, 1) = target_original;
            result_pred{end+1, 1} = mat2str(MB_original);
            result_truth{end+1, 1} = mat2str(truth_original);
            result_precision(end+1, 1) = Precision;
            result_recall(end+1, 1) = Recall;
            result_f1(end+1, 1) = F1;
            result_cits(end+1, 1) = test;
            result_time(end+1, 1) = runtime;

        end

    end

end

% ============================================================
% 构造详细结果表
% ============================================================

all_results = table( ...
    result_network, ...
    result_sample, ...
    result_target, ...
    result_pred, ...
    result_truth, ...
    result_precision, ...
    result_recall, ...
    result_f1, ...
    result_cits, ...
    result_time, ...
    'VariableNames', { ...
    'network', ...
    'sample_size', ...
    'target', ...
    'predicted_MB', ...
    'true_MB', ...
    'Precision', ...
    'Recall', ...
    'F1', ...
    'CITs', ...
    'Time'});

% ============================================================
% 保存详细结果
% ============================================================

detail_file = fullfile(result_dir, 'HITONMB_4networks_results.csv');

writetable(all_results, detail_file);

% ============================================================
% 汇总
% ============================================================

summary_network = {};
summary_sample = [];
summary_target_count = [];
summary_precision = [];
summary_recall = [];
summary_f1 = [];
summary_cits = [];
summary_time = [];

for ni = 1:length(networks)

    net = networks{ni};

    for si = 1:length(sample_sizes)

        n = sample_sizes(si);

        idx = strcmp(all_results.network, net) & ...
              all_results.sample_size == n;

        precision_values = all_results.Precision(idx);
        recall_values = all_results.Recall(idx);
        f1_values = all_results.F1(idx);
        cits_values = all_results.CITs(idx);
        time_values = all_results.Time(idx);

        summary_network{end+1, 1} = net;
        summary_sample(end+1, 1) = n;

        summary_target_count(end+1, 1) = ...
            sum(~isnan(f1_values));

        summary_precision(end+1, 1) = ...
            mean(precision_values, 'omitnan');

        summary_recall(end+1, 1) = ...
            mean(recall_values, 'omitnan');

        summary_f1(end+1, 1) = ...
            mean(f1_values, 'omitnan');

        summary_cits(end+1, 1) = ...
            mean(cits_values, 'omitnan');

        summary_time(end+1, 1) = ...
            mean(time_values, 'omitnan');

    end

end

% ============================================================
% 最终汇总表
% ============================================================

summary_results = table( ...
    summary_network, ...
    summary_sample, ...
    summary_target_count, ...
    summary_precision, ...
    summary_recall, ...
    summary_f1, ...
    summary_cits, ...
    summary_time, ...
    'VariableNames', { ...
    'network', ...
    'sample_size', ...
    'target_count', ...
    'Precision', ...
    'Recall', ...
    'F1', ...
    'Mean_CITs', ...
    'Mean_Time'});

% ============================================================
% 保存
% ============================================================

summary_file = fullfile( ...
    result_dir, ...
    'HITONMB_4networks_summary.csv');

writetable(summary_results, summary_file);

% ============================================================
% 输出
% ============================================================

fprintf('\n');
fprintf('============================================\n');
fprintf('4 个网络 HITON-MB 全部完成\n');
fprintf('============================================\n');

disp(summary_results);

fprintf('\n详细结果:\n');
fprintf('%s\n', detail_file);

fprintf('\n汇总结果:\n');
fprintf('%s\n', summary_file);