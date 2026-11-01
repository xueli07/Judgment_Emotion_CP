function ICCs = compute_icc_matrix(Y)
% COMPUTE_ICC_MATRIX  计算 ICC(1,1), ICC(2,1), ICC(3,1)
% 输入 Y: nSubjects x nItems （被试 × 像素）
% 输出 ICCs: 结构体，字段 .ICC1, .ICC2, .ICC3

% 预设返回（保证总会被赋值）
ICCs = struct('ICC1', NaN, 'ICC2', NaN, 'ICC3', NaN);

% 基本输入检查
if isempty(Y)
    warning('compute_icc_matrix: input Y is empty. Returning NaNs.');
    return;
end

[n, k] = size(Y); % n = 被试, k = 像素数

if n < 2
    warning('compute_icc_matrix: 需要至少 2 个被试 (n=%d). 返回 NaN.', n);
    return;
end
if k < 2
    warning('compute_icc_matrix: 需要至少 2 个测量项/像素 (k=%d). 返回 NaN.', k);
    return;
end

% 去掉含 NaN 的像素列（若整列 NaN，则去掉）
colNan = all(isnan(Y),1);
if any(colNan)
    Y(:, colNan) = [];
    k = size(Y,2);
    if k < 2
        warning('compute_icc_matrix: 去掉含 NaN 的像素后不足 2 个像素。返回 NaN.');
        return;
    end
end

% 重新计算均值与方差（按定义）
grandMean = mean(Y(:));
meanSubj = mean(Y, 2);    % 每个被试的均值
meanItem = mean(Y, 1);    % 每个像素的均值

% 方差分解
SS_total = sum((Y(:) - grandMean).^2);

SS_subject = k * sum((meanSubj - grandMean).^2); % between subjects
SS_item = n * sum((meanItem - grandMean).^2);    % between items
SS_residual = SS_total - SS_subject - SS_item;

% 自由度
df_subject = n - 1;
df_item = k - 1;
df_residual = (n - 1) * (k - 1);

% 防止除零（如果任何 df == 0 则返回 NaN）
if df_subject == 0 || df_item == 0 || df_residual == 0
    warning('compute_icc_matrix: 自由度为0 (df_subject=%d, df_item=%d, df_residual=%d). 返回 NaN.', ...
        df_subject, df_item, df_residual);
    return;
end

% 均方
MS_subject = SS_subject / df_subject;
MS_item = SS_item / df_item;
MS_residual = SS_residual / df_residual;

% 若 MS_residual 为 0，则不能计算 ICC，返回 NaN
if MS_residual == 0
    warning('compute_icc_matrix: MS_residual == 0，无法计算 ICC（返回 NaN）。');
    return;
end

% 计算 ICC（公式按 Shrout & Fleiss）
ICC1 = (MS_subject - MS_residual) / (MS_subject + (k - 1) * MS_residual);  % ICC(1,1)
ICC3 = (MS_subject - MS_residual) / (MS_subject + (k - 1) * MS_residual);  % ICC(3,1) 与 ICC1 形式相同（不同假设解释不同）
ICC2 = (MS_subject - MS_residual) / (MS_subject + (k - 1) * MS_residual + (k/n) * (MS_item - MS_residual)); % ICC(2,1)

% 安全处理 NaN/Inf
if ~isfinite(ICC1); ICC1 = NaN; end
if ~isfinite(ICC2); ICC2 = NaN; end
if ~isfinite(ICC3); ICC3 = NaN; end

% 返回结构
ICCs.ICC1 = ICC1;
ICCs.ICC2 = ICC2;
ICCs.ICC3 = ICC3;
end
