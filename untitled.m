cd('E:\Causal-Learner-main\Causal-Learner-main\Code'); addpath(genpath(pwd));
data = readmatrix('E:\Causal-LLM\data\ASIA\ASIA_n500_num.txt');
ns = zeros(1, size(data,2));
for j = 1:size(data,2), ns(j) = numel(unique(data(:,j))); end
% target = 4 是 lung；打印它的 PC 与分离集
[pc, ntest, t, sepset] = HITONPC_G2(data, 4, 0.05, ns, size(data,2), 3);
disp('PC ='); disp(pc)
for i = 1:numel(sepset)
    if ~isempty(sepset{i}), fprintf('sepset{%d} = %s\n', i, mat2str(sepset{i})); end
end
% 再打印 6（either）的 PC，看它是否包含 2(tub) 和 4(lung)
[pc6, ~, ~, ~] = HITONPC_G2(data, 6, 0.05, ns, size(data,2), 3);
disp('PC(either) ='); disp(pc6)