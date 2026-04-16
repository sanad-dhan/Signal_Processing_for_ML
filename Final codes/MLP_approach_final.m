%% MLP Embedding + Distance Matching
clc;
clear;
close all;

%% Initialization 
SR=16000;
N_MELS=64;
N_FFT=512;
HOP_LENGTH=256;
NUM_STUDENTS=100;
NUM_TRAIN_FILES=20;
MAX_EPOCHS=200;
MINI_BATCH=32;
LEARN_RATE=5e-4;
EMBED_DIM=128;
NUM_ITERATIONS=1000;
RNG_SEED=42;
rng(RNG_SEED);

%% LOAD DATASET 
ads=audioDatastore('E:\Sanad sem 8\SP4ML\Project\speech_data', 'IncludeSubfolders', true);
filepaths=ads.Files;
[~, names, ~]=cellfun(@fileparts, filepaths, 'UniformOutput', false);
speakerIDs=cellfun(@(x) strtok(x, '_'), names, 'UniformOutput', false);
uniqueStudents=unique(speakerIDs);
uniqueStudents=uniqueStudents(1:min(NUM_STUDENTS, numel(uniqueStudents)));

filepaths=filepaths(keep);
speakerIDs=speakerIDs(keep);
[~, ~, labelIdx]=unique(speakerIDs);

% Remove speakers with fewer than 3 files
labelCounts=histcounts(labelIdx, 1:max(labelIdx)+1);
validLabels=find(labelCounts >= 3);
keep2=ismember(labelIdx, validLabels);
filepaths=filepaths(keep2);
speakerIDs=speakerIDs(keep2);
[~, ~, labelIdx]=unique(speakerIDs);
nSpeakers=numel(unique(labelIdx));

%% EXTRACT FEATURE VECTORS 
N=numel(filepaths);
features=[];

for i=1:N
    vec=extractFlatFeatures(filepaths{i}, SR, N_MELS, N_FFT, HOP_LENGTH);
    if isempty(features)
        features=zeros(N, numel(vec));
    end
    features(i,:)=vec;
    if mod(i,400)==0
        fprintf(' %d / %d\n', i, N); 
    end
end

% Normalize features (zero mean, unit variance per dimension)
[features, mu, sg]=zscore(features);

%% TRAIN/TEST SPLIT 
trainIdx=false(N, 1);
for c=1:nSpeakers
    idx=find(labelIdx==c);
    n_train=min(NUM_TRAIN_FILES,numel(idx));
    trainIdx(idx(1:n_train))=true;
end
testIdx=~trainIdx;

Xtrain=features(trainIdx, :);
Xtest=features(testIdx,  :);
ytrain_int=labelIdx(trainIdx);
ytest_int=labelIdx(testIdx);

ytrain=categorical(cellstr(num2str(ytrain_int)));
ytest=categorical(cellstr(num2str(ytest_int)), categories(ytrain));

nClasses=numel(categories(ytrain));
featDim=size(Xtrain, 2);

%% MLP ARCHITECTURE 
% Flat feature vector → several FC layers → embedding → classifier
layers = [
    featureInputLayer(featDim, 'Normalization', 'none')

    % Hidden layer 1
    fullyConnectedLayer(512)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)

    % Hidden layer 2
    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.3)

    % Embedding layer
    fullyConnectedLayer(EMBED_DIM, 'Name', 'embedding')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.4)

    % Classifier head
    fullyConnectedLayer(nClasses, 'Name', 'classifier')
    softmaxLayer
    classificationLayer
];

%% TRAINING OPTIONS
options = trainingOptions('adam', ...
    'MaxEpochs',MAX_EPOCHS, ...
    'MiniBatchSize', MINI_BATCH, ...
    'InitialLearnRate',LEARN_RATE, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',15, ...
    'Shuffle','every-epoch', ...
    'ValidationData', {Xtest, ytest}, ...
    'ValidationFrequency', 30, ...
    'L2Regularization', 1e-4, ...
    'Plots','training-progress', ...
    'Verbose', true);

%% TRAIN MLP 
net = trainNetwork(Xtrain, ytrain, layers, options);

%% EXTRACT EMBEDDINGS
embed_train=activations(net, Xtrain, 'embedding', 'OutputAs', 'rows');
embed_test=activations(net, Xtest,  'embedding', 'OutputAs', 'rows');

% L2 normalize
embed_train=embed_train./(vecnorm(embed_train, 2, 2) + eps);
embed_test=embed_test./(vecnorm(embed_test,  2, 2) + eps);

%% BUILD DICTIONARY
D_mlp=zeros(EMBED_DIM, nSpeakers);
for c=1:nSpeakers
    idx=find(ytrain_int==c);
    if ~isempty(idx)
        D_mlp(:,c)=mean(embed_train(idx,:), 1)';
    end
end
D_mlp=D_mlp./(vecnorm(D_mlp) + eps);

%% DISTANCE MATCHING—1000 TRIALS 
fprintf('\nRunning %d test trials...\n', NUM_ITERATIONS);
metrics={'euclidean','cosine','cityblock'};
metric_names={'Euclidean','Cosine','Manhattan'};
correct=zeros(1,3);

testEmbed_byClass=cell(nSpeakers,1);
for c=1:nSpeakers
    testEmbed_byClass{c}=find(ytest_int == c);
end

for i = 1:NUM_ITERATIONS
    trueIdx=randi(nSpeakers);
    pool=testEmbed_byClass{trueIdx};
    if isempty(pool)
        continue;
    end
    fileIdx=pool(randi(numel(pool)));
    y_embed=embed_test(fileIdx,:);
    for d=1:3
        dists=pdist2(y_embed, D_mlp', metrics{d});
        [~, pred]=min(dists);
        if pred==trueIdx
            correct(d)=correct(d)+1; 
        end
    end
end

%% FULL TEST EVALUATION 
metrics_eval={'euclidean','cosine','cityblock'};
metric_names_eval={'Euclidean','Cosine','Manhattan'};
nTestFiles=numel(ytest_int);
acc_full_results=zeros(1,3); 

% Calculate Full Test results for all metrics
for d = 1:3
    % Vectorized distance calculation for the entire test set
    all_dists=pdist2(embed_test, D_mlp',metrics_eval{d});
    [~, pred_all]=min(all_dists, [], 2);
    acc_full_results(d)=mean(pred_all==ytest_int)*100;
end

%% FULL TEST EVALUATION 
metrics_eval={'euclidean','cosine','cityblock'};
metric_names_eval={'Euclidean','Cosine','Manhattan'};
nTestFiles=numel(ytest_int);
acc_full_results=zeros(1,3); 

% Calculate Full Test results for all metrics
for d = 1:3
    % Vectorized distance calculation for the entire test set
    all_dists = pdist2(embed_test, D_mlp', metrics_eval{d});
    [~, pred_all] = min(all_dists, [], 2);
    acc_full_results(d) = mean(pred_all == ytest_int) * 100;
end

%% FINAL CONSOLIDATED RESULTS 

fprintf('\n==============================\n');
fprintf('MLP + Distance Matching\n');
fprintf('Feature dim : %d\n',   featDim);
fprintf('Embed dim   : %d\n',   EMBED_DIM);
fprintf('Speakers    : %d\n',   nSpeakers);
fprintf('------------------------------\n');
fprintf('1000-TRIAL RANDOM TEST:\n');
for d = 1:3
    fprintf('%-12s : %.2f%%\n', metric_names_eval{d}, (correct(d)/NUM_ITERATIONS)*100);
end
fprintf('------------------------------\n');
fprintf('FULL TEST (ALL FILES):\n');
for d = 1:3
    fprintf('%-12s : %.2f%%\n', metric_names_eval{d}, acc_full_results(d));
end
fprintf('Total Test Files: %d\n', nTestFiles);
fprintf('==============================\n');

%% HELPER FUNCTION

function vec=extractFlatFeatures(filepath, sr, n_mels, n_fft, hop_length)
% Returns concatenated [MFCC mean | Mel mean | STFT mean] as a row vector
    [y, fs]=audioread(filepath);
    y=mean(y,2);
    if fs~=sr
        y=resample(y, sr, fs); 
    end
    if numel(y)<sr
        y(end+1:sr)=0;
    else 
        y = y(1:sr);
    end

    % MFCC — mean over time (1 x 13)
    coeffs=mfcc(y, sr, 'NumCoeffs', 13);
    mfcc_vec=mean(coeffs, 1);

    % Mel spectrogram — mean over time (1 x n_mels)
    [S, ~, ~]=melSpectrogram(y, sr, 'FFTLength', n_fft,'OverlapLength', n_fft-hop_length, 'NumBands', n_mels);
    mel_vec=mean(10*log10(S+1e-6),2)';

    % STFT — mean over time (1 x F)
    stft_mat=abs(stft(y, sr, 'FFTLength', 128, 'OverlapLength', 64));
    stft_vec=mean(abs(stft_mat), 2)';

    vec=[mfcc_vec, mel_vec, stft_vec];
end