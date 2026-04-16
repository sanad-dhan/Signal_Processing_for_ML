%% CNN Embedding + Hybrid Feature Images
clc;
clear; 
close all;

%% Initialization 
SR=16000;
N_MELS=64;
N_MFCC=20;     
N_FFT=512;
HOP_LENGTH=256;
NUM_STUDENTS=100;
NUM_TRAIN_FILES=20;     
MAX_EPOCHS=200;     
MINI_BATCH=32;
LEARN_RATE=5e-4;   
EMBED_DIM=128;
NUM_ITERATIONS=1000;
rng(42);

%% LOAD DATASET 
ads=audioDatastore('E:\Sanad sem 8\SP4ML\Project\speech_data', 'IncludeSubfolders', true);
filepaths=ads.Files;
[~, names, ~]=cellfun(@fileparts, filepaths, 'UniformOutput', false);
speakerIDs=cellfun(@(x) strtok(x, '_'), names, 'UniformOutput', false);
uniqueStudents=unique(speakerIDs);
uniqueStudents=uniqueStudents(1:min(NUM_STUDENTS, numel(uniqueStudents)));

keep=ismember(speakerIDs, uniqueStudents);
filepaths=filepaths(keep);
speakerIDs=speakerIDs(keep);
[~, ~, labelIdx]=unique(speakerIDs);
nSpeakers=numel(unique(labelIdx));

%% EXTRACT HYBRID FEATURE IMAGES
N = numel(filepaths);
sampleImg=extractHybridImage(filepaths{1}, SR, N_MELS, N_MFCC, N_FFT, HOP_LENGTH);
[H, W]=size(sampleImg);
allData=zeros(H, W, 1, N, 'single');

for i=1:N
    allData(:,:,1,i)=extractHybridImage(filepaths{i}, SR, N_MELS, N_MFCC, N_FFT, HOP_LENGTH);
    if mod(i,400)==0
        fprintf(' %d / %d\n', i, N); 
    end
end

%% TRAIN / TEST SPLIT 
trainIdx = false(N,1);
for c=1:nSpeakers
    idx=find(labelIdx == c);
    n_train=min(NUM_TRAIN_FILES, numel(idx));
    trainIdx(idx(1:n_train))=true;
end
testIdx=~trainIdx;

Xtrain=allData(:,:,:, trainIdx);
Xtest=allData(:,:,:, testIdx);
ytrain_int=labelIdx(trainIdx);
ytest_int=labelIdx(testIdx);
ytrain=categorical(cellstr(num2str(ytrain_int)));
ytest=categorical(cellstr(num2str(ytest_int)), categories(ytrain));

%% CNN ARCHITECTURE 
layers=[
    imageInputLayer([H W 1], 'Normalization', 'zerocenter')
    
    convolution2dLayer(3, 32, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 128, 'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(256) 
    reluLayer
    dropoutLayer(0.4)
    
    fullyConnectedLayer(EMBED_DIM, 'Name', 'embedding')
    batchNormalizationLayer
    
    fullyConnectedLayer(nSpeakers, 'Name', 'classifier')
    softmaxLayer
    classificationLayer
];

%% TRAINING 
options=trainingOptions('adam', ...
    'MaxEpochs', MAX_EPOCHS, ...
    'MiniBatchSize', MINI_BATCH, ...
    'InitialLearnRate', LEARN_RATE, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {Xtest, ytest}, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net=trainNetwork(Xtrain, ytrain, layers, options);

%% EMBEDDING EXTRACTION 
embed_train=activations(net, Xtrain, 'embedding', 'OutputAs', 'rows');
embed_test=activations(net, Xtest,  'embedding', 'OutputAs', 'rows');

% L2 Normalize
embed_train=embed_train./(vecnorm(embed_train, 2, 2)+eps);
embed_test=embed_test./(vecnorm(embed_test,  2, 2)+eps);

% Build Dictionary
D_cnn=zeros(EMBED_DIM, nSpeakers);
for c=1:nSpeakers
    idx=find(ytrain_int == c);
    if ~isempty(idx)
        D_cnn(:,c)=mean(embed_train(idx,:), 1)';
    end
end
D_cnn=D_cnn./(vecnorm(D_cnn)+eps);

%% EVALUATION (1000 TRIALS + FULL TEST)
metrics={'euclidean', 'cosine', 'cityblock'};
m_names={'Euclidean', 'Cosine', 'Manhattan'};
correct=zeros(1,3);

% 1000 Random Trials
for i=1:NUM_ITERATIONS
    trueIdx=randi(nSpeakers);
    pool=find(ytest_int == trueIdx);
    if isempty(pool)
        continue; 
    end
    y_vec=embed_test(pool(randi(numel(pool))), :);
    for d=1:3
        dists=pdist2(y_vec, D_cnn', metrics{d});
        [~, pred]=min(dists);
        if pred==trueIdx
            correct(d)=correct(d)+1;
        end
    end
end

% Full Test Evaluation
acc_full=zeros(1,3);
for d=1:3
    all_dists=pdist2(embed_test, D_cnn', metrics{d});
    [~, preds]=min(all_dists, [], 2);
    acc_full(d)=mean(preds == ytest_int)*100;
end

%% FINAL RESULTS 
fprintf('\n==============================\n');
fprintf('CNN HYBRID EMBEDDING RESULTS\n');
fprintf('Speakers    : %d\n', nSpeakers);
fprintf('------------------------------\n');
fprintf('1000-TRIAL RANDOM TEST:\n');
for d = 1:3
    fprintf('%-12s : %.2f%%\n', m_names{d}, (correct(d)/NUM_ITERATIONS)*100);
end
fprintf('------------------------------\n');
fprintf('FULL TEST (ALL FILES):\n');
for d = 1:3
    fprintf('%-12s : %.2f%%\n', m_names{d}, acc_full(d));
end
fprintf('==============================\n');

%% HELPER FUNCTION
function img=extractHybridImage(filepath, sr, n_mels, n_mfcc, n_fft, hop_length)
    [y, fs]=audioread(filepath);
    y=mean(y,2);
    if fs~=sr
        y=resample(y, sr, fs);
    end
    y=y(1:min(end, sr)); % 1 second
    if numel(y)<sr
        y(end+1:sr)=0;
    end
    
    % Mel Spectrogram
    S=melSpectrogram(y, sr, 'FFTLength', n_fft, 'OverlapLength', n_fft-hop_length, 'NumBands', n_mels);
    mel=10*log10(S + 1e-6);
    
    % MFCC
    coeffs=mfcc(y, sr, 'NumCoeffs', n_mfcc-1); % returns n_mfcc coefficients
    mfcc_img=resample(coeffs, size(mel, 2), size(coeffs, 1))'; % Match time width
    
    % Stack and Normalize
    combined=[mfcc_img; mel];
    img=(combined-min(combined(:)))/(max(combined(:))-min(combined(:))+eps);
end