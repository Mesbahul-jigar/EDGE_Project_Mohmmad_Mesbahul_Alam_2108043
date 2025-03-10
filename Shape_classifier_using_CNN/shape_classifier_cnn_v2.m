%% Load and Prepare Dataset
datasetPath = 'C:\Users\mmaji\ETE\EDGE_COURSE_MATLAB\machine learning\CNN\shape_classifier\shape_dataset';

imds = imageDatastore(datasetPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

if numel(unique(imds.Labels)) ~= 4
    error('Dataset should contain exactly 4 classes: triangle, circle, star, square');
end

[imdsTrain, imdsVal] = splitEachLabel(imds, 0.7, 'randomized');

%% Enhanced Data Augmentation
targetSize = [224 224 3];

augmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandRotation', [-30 30], ...
    'RandScale', [0.8 1.2], ...
    'RandXTranslation', [-20 20], ...
    'RandYTranslation', [-20 20]);

augimdsTrain = augmentedImageDatastore(targetSize, imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsVal = augmentedImageDatastore(targetSize, imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

%% Simplified CNN Architecture
numClasses = numel(categories(imds.Labels));

layers = [
    imageInputLayer(targetSize)
    
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')

    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')

    globalAveragePooling2dLayer('Name', 'gap')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc_final') 
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

%% Training Configuration
numTraining = numel(imdsTrain.Files);
validationFrequency = floor(numTraining / 32);

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.0001, ...        % Lowered for stability
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augimdsVal, ...
    'ValidationFrequency', validationFrequency, ...
    'ValidationPatience', 5, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10, ...
    'L2Regularization', 0.001, ...         % Increased to reduce overfitting
    'OutputNetwork', 'best-validation-loss'); % Save best model

%% Train and Save Model
[net, trainInfo] = trainNetwork(augimdsTrain, layers, options);

%% Enhanced Evaluation and Result Saving
resultsDir = fullfile(pwd, 'results');
if ~exist(resultsDir, 'dir')
    mkdir(resultsDir);
end

saveas(gcf, fullfile(resultsDir, 'training_progress.png'));

[YPred, scores] = classify(net, augimdsVal);
YVal = imdsVal.Labels;

accuracy = mean(YPred == YVal);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

fid = fopen(fullfile(resultsDir, 'metrics.txt'), 'w');
fprintf(fid, 'Validation Accuracy: %.2f%%\n', accuracy * 100);
fclose(fid);

figure('Units', 'normalized', 'Position', [0.1 0.1 0.6 0.6])
cm = confusionchart(YVal, YPred, ...
    'Title', 'Shape Classification Confusion Matrix', ...
    'RowSummary', 'row-normalized', ...
    'ColumnSummary', 'column-normalized');
sortClasses(cm, categories(imds.Labels))
exportgraphics(gcf, fullfile(resultsDir, 'confusion_matrix.png'), 'Resolution', 300);

% Save final model and training info
save(fullfile(resultsDir, 'shape_classifier_model.mat'), 'net', 'trainInfo');