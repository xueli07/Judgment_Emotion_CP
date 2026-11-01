% Enhanced Classification Image (CI) Analysis Script - Task Version (One-Sample Only)
% Features: 
% 1) Import mask boundaries via B&W image 
% 2) Batch import PNG grayscale images and convert to CI files 
% 3) Support for 512x512 image dimensions
%--------------------------------------------------------------------------
% User Configuration Parameters
%--------------------------------------------------------------------------
addpath(genpath('subfunction'));

% Set basic parameters
xySize = 512;            % Image size (pixels)
nbTask = 2;              % Number of tasks

% Set mask image path (B&W image, Black = non-face region, White = face region)
maskImagePath = 'input_pic/average_face_mask.png';  

% Set paths for PNG image folders for both tasks
continuousImagesFolder = 'continuous';  % Continuous task PNG folder
categoricalImagesFolder = 'category';   % Categorical task PNG folder

% Set image file extension
imageExtension = '*.png';  

% Set effective number of subjects for each task (0 = read automatically)
nContinuous = 0;  
nCategorical = 0; 

% Statistical analysis parameters
pValue = 0.025;  % Significance level (0.05 for two-tailed test)
tCritical = 2.7; % Critical t-value (approx. corresponds to p=0.05 two-tailed)
nPeriod = 3;     % Period parameter for Gaussian smoothing kernel

%--------------------------------------------------------------------------
% Step 1: Import Mask Image and Base Background Image
%--------------------------------------------------------------------------
visage = double(imread('input_pic/average_face.jpg')) / 255;
visage = imresize(visage, [xySize, xySize]);  % Ensure dimension matches xySize

% Input background images (kept original filenames to avoid read errors)
task_bg = ["input_pic/thinkstyle1_autoscaled.png", "input_pic/thinkstyle2_autoscaled.png"];
factor_visage = .25; % Base image ratio
factor_noise = .75;  % Noise ratio

try
    % Import mask image
    maskImage = imread(maskImagePath);
    maskImage = imresize(maskImage, [xySize, xySize]);  % Unify mask dimension
    
    % Convert to grayscale if it is a color image
    if size(maskImage, 3) > 1
        maskImage = rgb2gray(maskImage);
    end
    
    % Binarize mask image (Threshold = 0.5)
    mask = imbinarize(maskImage, 0.5);   % 1 = face, 0 = non-face
    valid_mask = mask;                   % "Face region" for subsequent display/stats
    bg_mask = ~mask;                     % 1 = background/non-face, used for Z-transform null hypothesis
    
    % Get pixel indices for mask regions
    vectMask = find(bg_mask == 1);       % Z-standardization on non-face background region
    faceMask = find(bg_mask == 0);
    fprintf('Successfully imported and processed mask image. Non-face pixels: %d\n', length(vectMask));
    
    % Display mask image for verification
    figure('Name', 'Imported Mask Image');
    subplot(1, 2, 1);
    imagesc(valid_mask);
    colormap(gray);
    title('Original Mask Image');
    axis image;
    
    subplot(1, 2, 2);
    imagesc(bg_mask);
    colormap(gray);
    title('Processed Mask (Black = Face Region)');
    axis image;
    pause(1);  % Brief display for user confirmation
catch ME
    warning('PublicRepository:MaskImport', 'Failed to import mask image. Using default boundary mask: %s', ME.message);
    % Create default boundary mask
    vectMask = find(bg_mask == 1);  
    faceMask = find(bg_mask == 0);
end

%--------------------------------------------------------------------------
% Step 2: Batch Import PNG Grayscale Images and Convert to CI
%--------------------------------------------------------------------------
% Batch import continuous task PNG images
try
    continuousImageFiles = dir(fullfile(continuousImagesFolder, imageExtension));
    if nContinuous > 0 && nContinuous < length(continuousImageFiles)
        continuousImageFiles = continuousImageFiles(1:nContinuous);
    end
    nContinuous = length(continuousImageFiles);
    
    % Initialize 3D array for continuous task CI data
    ICContinuous = zeros(xySize, xySize, nContinuous);
    
    fprintf('Importing continuous task PNG images (Total: %d)...\n', nContinuous);
    for i = 1:nContinuous
        filePath = fullfile(continuousImagesFolder, continuousImageFiles(i).name);
        img = imread(filePath);
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        if size(img, 1) ~= xySize || size(img, 2) ~= xySize
            img = imresize(img, [xySize, xySize]);
        end
        ICContinuous(:, :, i) = double(img);
        
        if mod(i, 10) == 0 || i == nContinuous
            fprintf('Imported continuous task images: %d/%d\n', i, nContinuous);
        end
    end
    fprintf('Successfully imported continuous task CI data: %d x %d x %d\n', size(ICContinuous));
catch ME
    error('Failed to import continuous task PNG images: %s', ME.message);
end

% Batch import categorical task PNG images
try
    categoricalImageFiles = dir(fullfile(categoricalImagesFolder, imageExtension));
    if nCategorical > 0 && nCategorical < length(categoricalImageFiles)
        categoricalImageFiles = categoricalImageFiles(1:nCategorical);
    end
    nCategorical = length(categoricalImageFiles);
    
    % Initialize 3D array for categorical task CI data
    ICCategorical = zeros(xySize, xySize, nCategorical);
    
    fprintf('Importing categorical task PNG images (Total: %d)...\n', nCategorical);
    for i = 1:nCategorical
        filePath = fullfile(categoricalImagesFolder, categoricalImageFiles(i).name);
        img = imread(filePath);
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        if size(img, 1) ~= xySize || size(img, 2) ~= xySize
            img = imresize(img, [xySize, xySize]);
        end
        ICCategorical(:, :, i) = double(img);
        
        if mod(i, 10) == 0 || i == nCategorical
            fprintf('Imported categorical task images: %d/%d\n', i, nCategorical);
        end
    end
    fprintf('Successfully imported categorical task CI data: %d x %d x %d\n', size(ICCategorical));
catch ME
    error('Failed to import categorical task PNG images: %s', ME.message);
end

% Optional: Save imported CI data as MAT files for future direct use
shouldSaveCIMat = true;  
if shouldSaveCIMat
    if ~exist('result', 'dir'), mkdir('result'); end
    save('result/Continuous_Task_CI.mat', 'ICContinuous');
    save('result/Categorical_Task_CI.mat', 'ICCategorical');
    fprintf('CI data saved as MAT files.\n');
end

KeptSujContinuous = 1:nContinuous;
KeptSujCategorical = 1:nCategorical;

%--------------------------------------------------------------------------
% Step 3: Create Gaussian Smoothing Kernel
%--------------------------------------------------------------------------
ySize = xySize;
xSize = xySize;
nBands = 6;
probBubbles = 4.^(nBands-2:-1:0) / sum(4.^(nBands-2:-1:0));
imSubSample = 0;
nZero = 2.18;

% Set smoothing directly via FWHM
smoothFWHM_px = 7;              
stdev = smoothFWHM_px / sqrt(8*log(2));  
maxHalfSize1 = round(stdev * 2.18);
[y, x] = meshgrid(-maxHalfSize1:maxHalfSize1, -maxHalfSize1:maxHalfSize1);
gauss1 = exp(-(x.^2 + y.^2) / (stdev^2));

%--------------------------------------------------------------------------
% Step 4: Z-score Standardization
%--------------------------------------------------------------------------
ZCiAllContinuous = zeros(ySize, xSize, nContinuous);
ZCiAllCategorical = zeros(ySize, xSize, nCategorical);

fprintf('Z-score standardization for continuous task CI data...\n');
for sujet = 1:nContinuous
    temp = ICContinuous(:, :, sujet);
    smTemp = convn(temp, gauss1, 'same');   
    ZtheMean = mean(smTemp(faceMask));      
    ZtheStd = std(smTemp(faceMask));        
    ZCiAllContinuous(:, :, sujet) = (smTemp - ZtheMean) ./ ZtheStd;  
    if mod(sujet, 10) == 0 || sujet == nContinuous
        fprintf('Processed continuous task subjects: %d/%d\n', sujet, nContinuous);
    end
end

fprintf('Z-score standardization for categorical task CI data...\n');
for sujet = 1:nCategorical
    temp = ICCategorical(:, :, sujet);
    smTemp = convn(temp, gauss1, 'same');   
    ZtheMean = mean(smTemp(faceMask));      
    ZtheStd = std(smTemp(faceMask));        
    ZCiAllCategorical(:, :, sujet) = (smTemp - ZtheMean) ./ ZtheStd;  
    if mod(sujet, 10) == 0 || sujet == nCategorical
        fprintf('Processed categorical task subjects: %d/%d\n', sujet, nCategorical);
    end
end

%--------------------------------------------------------------------------
% Extra Step: Calculate ICC and Alpha Coefficients
%--------------------------------------------------------------------------
validIdx = find(valid_mask == 1); % Face pixels
if isempty(validIdx)
    error('No valid pixels found within the mask. Please check the mask image.');
end

% Construct matrix for Continuous task
matCont = zeros(nContinuous, numel(validIdx));
for s = 1:nContinuous
    temp = ZCiAllContinuous(:,:,s);
    matCont(s,:) = temp(validIdx);
end
fprintf('matCont size: %d subjects x %d pixels\n', size(matCont,1), size(matCont,2));
try
    ICCs_Cont = compute_icc_matrix(matCont);
    fprintf('Continuous task ICC(2,1)=%.3f (ICC1=%.3f, ICC3=%.3f)\n', ...
        ICCs_Cont.ICC2, ICCs_Cont.ICC1, ICCs_Cont.ICC3);
catch
    warning('compute_icc_matrix function not found or failed. Skipping ICC.');
end

% Construct matrix for Categorical task
matCat = zeros(nCategorical, numel(validIdx));
for s = 1:nCategorical
    temp = ZCiAllCategorical(:,:,s);
    matCat(s,:) = temp(validIdx);
end
fprintf('matCat size: %d subjects x %d pixels\n', size(matCat,1), size(matCat,2));
try
    ICCs_Cat = compute_icc_matrix(matCat);
    fprintf('Categorical task ICC(2,1)=%.3f (ICC1=%.3f, ICC3=%.3f)\n', ...
        ICCs_Cat.ICC2, ICCs_Cat.ICC1, ICCs_Cat.ICC3);
catch
    warning('compute_icc_matrix function not found or failed. Skipping ICC.');
end

%--------------------------------------------------------------------------
% Step 5: Statistical Analysis (One-sample t-test)
%--------------------------------------------------------------------------
t_rev = zeros(xySize, xySize, nbTask);
effectSize = zeros(xySize, xySize, nbTask);
confIntervals = zeros(xySize, xySize, nbTask, 2);
StatsDescr = zeros(xySize, xySize, nbTask, 2);

% Process continuous task (Task 1)
n = nContinuous;
temp = squeeze(ZCiAllContinuous);
theMean = squeeze(mean(temp, 3));
theStd = squeeze(std(temp, [], 3));
std_err = (theStd / sqrt(n));  

t_rev(:, :, 1) = theMean ./ std_err;  
effectSize(:, :, 1) = theMean ./ theStd;  
confIntervals(:, :, 1, 1) = theMean - tCritical .* std_err;  
confIntervals(:, :, 1, 2) = theMean + tCritical .* std_err;  
StatsDescr(:, :, 1, 1) = theMean;
StatsDescr(:, :, 1, 2) = theStd;

% Process categorical task (Task 2)
n = nCategorical;
temp = squeeze(ZCiAllCategorical);
theMean = squeeze(mean(temp, 3));
theStd = squeeze(std(temp, [], 3));
std_err = (theStd / sqrt(n));  

t_rev(:, :, 2) = theMean ./ std_err;  
effectSize(:, :, 2) = theMean ./ theStd;  
confIntervals(:, :, 2, 1) = theMean - tCritical .* std_err;  
confIntervals(:, :, 2, 2) = theMean + tCritical .* std_err;  
StatsDescr(:, :, 2, 1) = theMean;
StatsDescr(:, :, 2, 2) = theStd;

%--------------------------------------------------------------------------
% Step 6: Determine Statistical Thresholds (Multiple Comparison Correction)
%--------------------------------------------------------------------------
p = pValue / nbTask;  % Adjusted significance level
FWHM = smoothFWHM_px;  
D = 2;
[volumes, N] = CiVol(valid_mask, D);  

% Initialize result variables
SeuilsCiAll = zeros(ySize, xSize, 3);
pos_ci = zeros(ySize, xSize, 3);
neg_ci = zeros(ySize, xSize, 3);
maxVal = 5; 
minVal = -5; 
meanEffectSize = zeros(nbTask, 1);
meanConfIntervals = zeros(nbTask, 2);
StatsDescrMean = zeros(nbTask, 2);

fprintf('Determining statistical thresholds and extracting results...\n');
for task = 1:nbTask
    if task == 1
        df = nContinuous - 1;  
    else
        df = nCategorical - 1; 
    end
    
    [tP, k] = stat_threshold(volumes, N, FWHM, df, p, tCritical);
    Res.ZSCi = t_rev(:, :, task);
    Res.ZSCi = Res.ZSCi .* valid_mask;
    Res.tP = tP;
    Res.p = p;
    Res.tC = tCritical;
    Res.k = k;
    Res.FWHM = FWHM;
    
    file_bg_path = task_bg(task);
    background = double(imread(file_bg_path));
    
    % Force resize to prevent "Index exceeds array bounds" error 
    background = imresize(background, [xySize, xySize]); 
    
    out = DiplayResBi(Res, background);  
    
    % Store significant regions
    SeuilsCiAll(:, :, task) = out{1} + out{2};
    pos_ci(:, :, task) = out{1};
    neg_ci(:, :, task) = out{2};
    
    % Calculate mean effect size and CIs for significant regions
    sigVect = find(out{1}(:) + out{2}(:));
    tempEffectS = effectSize(:, :, task);
    tempCI_1 = confIntervals(:, :, task, 1);
    tempCI_2 = confIntervals(:, :, task, 2);
    tempDescr_1 = StatsDescr(:, :, task, 1);
    tempDescr_2 = StatsDescr(:, :, task, 2);
    
    meanEffectSize(task) = mean(abs(tempEffectS(sigVect)));
    meanConfIntervals(task, 1) = mean(abs(tempCI_1(sigVect)));
    meanConfIntervals(task, 2) = mean(abs(tempCI_2(sigVect)));
    StatsDescrMean(task, 1) = mean(abs(tempDescr_1(sigVect)));
    StatsDescrMean(task, 2) = mean(abs(tempDescr_2(sigVect)));
    fprintf('Completed statistical threshold determination for Task %d\n', task);
end

if ~exist('outputfig', 'dir'), mkdir('outputfig'); end

for task = 1:nbTask
    file_bg_path = task_bg(task);
    fprintf('%s\n', file_bg_path);
    visage_tk = double(imread(file_bg_path));
    
    % Force resize to prevent dimension mismatch error during assignment
    visage_tk = imresize(visage_tk, [xySize, xySize]); 
    
    visage3D = zeros(xySize, xySize, 3);  
    for ii = 1:3  
        visage3D(:, :, ii) = visage_tk;
    end
    
    figure, imagesc(t_rev(:, :, task).*valid_mask, [minVal maxVal]), colorbar; 
    axis image;
    axis ij; 
    
    name = 'temp.jpg';  
    exportfig(gcf, name, 'Format', 'jpeg', 'FontMode', 'fixed', 'FontSize', 10, 'color', 'cmyk'); 
    
    tempFig = double(imread(name))/255;  
    tempFig = tempFig(49:561, 100:680, :);  
    tempFig = imresize(tempFig, [ySize xSize]);  
    
    % Highlight significant region boundaries
    sig = zeros(xySize, xySize, 3);  
    for ii = 1:3
        sig(:, :, ii) = edge(SeuilsCiAll(:,:,task));  
    end
    
    posSig = find(sig(:));  
    
    % Blend t-value map and face image
    backFig = tempFig .* .25 + visage3D .* .75;  
    backFig(posSig) = 1;  
   
    imwrite(uint8(255*backFig), sprintf('outputfig/Exp4_Task%d.png', task)); 
end

%--------------------------------------------------------------------------
% Step 7: Save Data
%--------------------------------------------------------------------------
results = struct(...
    'tValues_Continuous', t_rev(:, :, 1), ...
    'tValues_Categorical', t_rev(:, :, 2), ...
    'effectSize_Continuous', effectSize(:, :, 1), ...
    'effectSize_Categorical', effectSize(:, :, 2), ...
    'confIntervals_Continuous', confIntervals(:, :, 1, :), ...
    'confIntervals_Categorical', confIntervals(:, :, 2, :), ...
    'meanEffectSize_Continuous', meanEffectSize(1), ...
    'meanEffectSize_Categorical', meanEffectSize(2), ...
    'significantRegions_Continuous', SeuilsCiAll(:, :, 1), ...
    'significantRegions_Categorical', SeuilsCiAll(:, :, 2), ...
    'maskUsed', mask, ...
    'nContinuous', nContinuous, ...
    'nCategorical', nCategorical ...
    );

save('result/Enhanced_CI_Task_Results.mat', 'results');
fprintf('\nAnalysis complete! Results saved to result/Enhanced_CI_Task_Results.mat\n');
fprintf('Number of continuous task subjects: %d\n', nContinuous);
fprintf('Number of categorical task subjects: %d\n', nCategorical);
