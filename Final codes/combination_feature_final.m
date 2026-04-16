%% Hybrid Feature Speaker Recognition (MFCC + Mel + STFT) 
close all;
clc;
clear all;
%% Loading Data
ads=audioDatastore('E:\Sanad sem 8\SP4ML\Project\speech_data', 'IncludeSubfolders', true);
filepaths=ads.Files;
[~, names, ~]=cellfun(@fileparts, filepaths, 'UniformOutput', false); 
speakerIDs=cellfun(@(x) strtok(x, '_'), names, 'UniformOutput', false); 
uniqueStudents=unique(speakerIDs);
%% Dictionary Creation
numStudents=100; 
numFilesperstudent=10;
testStartIdx=numFilesperstudent + 1;
featureSize=14 + 200 + 200; % 14 MFCC, 200 Mel, 200 STFT
D=zeros(featureSize, numStudents);
fprintf('Building Hybrid Dictionary (%d features)...\n', featureSize);
for i=1:numStudents
    currentID=uniqueStudents{i};
    studentFiles=filepaths(strcmp(speakerIDs, currentID));
    n=min(numFilesperstudent, length(studentFiles));
    student_feature=zeros(featureSize, n);
    
    for j=1:n
        [audio, fs]=audioread(studentFiles{j});
        
        % Feature 1: MFCC
        coeffs=mfcc(audio, fs);
        feat_mfcc=mean(coeffs, 1)';
        
        % Feature 2: Mel Spectrogram
        S_mel=melSpectrogram(audio, fs, 'NumBands', 200, 'ApplyLog', true);
        feat_mel=mean(S_mel, 2);
        
        % Feature 3: STFT
        [S_stft, ~]=stft(audio, fs, 'Window', hamming(256), 'OverlapLength', 200, 'FFTLength', 398, 'FrequencyRange', 'onesided');
        feat_stft=mean(log10(abs(S_stft) + 1e-6), 2);
        
        % Concatenate
        student_feature(:, j)=[feat_mfcc; feat_mel; feat_stft];
    end
    D(:, i)=mean(student_feature, 2); 
end
% Normalization
meanD=mean(D, 2);
stdD=std(D, 0, 2);
D_norm=(D - meanD) ./ (stdD + 1e-6);
fprintf('Dictionary successfully created.\n');
%% 1000-TRIAL TEST 
numIterations=1000;
correctEuc=0; correctCos=0; correctMan=0;
fprintf('\nRunning %d Random Trials...\n', numIterations);
for i=1:numIterations
    trueIdx=randi(numStudents);
    testID=uniqueStudents{trueIdx};
    testFiles=filepaths(strcmp(speakerIDs, testID));
    
    if length(testFiles) >= testStartIdx
        fileIdx=randi([testStartIdx, length(testFiles)]);
    else
        fileIdx=length(testFiles);
    end
    
    [testAudio, testFs]=audioread(testFiles{fileIdx});
    
    % Extract Hybrid Features
    c_test=mfcc(testAudio, testFs);
    f_mfcc=mean(c_test, 1)';
    
    S_mel_test=melSpectrogram(testAudio, testFs, 'NumBands', 200, 'ApplyLog', true);
    f_mel=mean(S_mel_test, 2);
    
    [S_stft_test, ~]=stft(testAudio, testFs, "Window", hamming(256), "OverlapLength", 200, 'FFTLength', 398, 'FrequencyRange', 'onesided');
    f_stft=mean(log10(abs(S_stft_test) + 1e-6), 2);
    
    y=[f_mfcc; f_mel; f_stft];
    y_norm=(y - meanD) ./ (stdD + 1e-6); 
    % Metrics
    distsEuc=zeros(1, numStudents);
    distsCos=zeros(1, numStudents);
    distsMan=zeros(1, numStudents);
    
    for k=1:numStudents
        distsEuc(k)=norm(y_norm - D_norm(:, k), 2);
        distsCos(k)=1 - dot(y_norm, D_norm(:, k)) / (norm(y_norm) * norm(D_norm(:, k)) + eps);
        distsMan(k)=norm(y_norm - D_norm(:, k), 1);
    end
    
    [~, predEuc]=min(distsEuc);
    [~, predCos]=min(distsCos);
    [~, predMan]=min(distsMan);
    
    if predEuc == trueIdx, correctEuc=correctEuc + 1; end
    if predCos == trueIdx, correctCos=correctCos + 1; end
    if predMan == trueIdx, correctMan=correctMan + 1; end
end
fprintf('\n==============================\n');
fprintf('1000-Trial Results (Hybrid)\n');
fprintf('------------------------------\n');
fprintf('Euclidean  : %.2f%%\n', (correctEuc/numIterations)*100);
fprintf('Cosine     : %.2f%%\n', (correctCos/numIterations)*100);
fprintf('Manhattan  : %.2f%%\n', (correctMan/numIterations)*100);
fprintf('==============================\n');
%% FULL TEST EVALUATION 
fprintf('\nBuilding full test set...\n');
fullTestFiles={};
fullTestLabels=[];
for i=1:numStudents
    currentID=uniqueStudents{i};
    studentFiles=filepaths(strcmp(speakerIDs, currentID));
    if length(studentFiles) >= testStartIdx
        extraFiles=studentFiles(testStartIdx:end);
        fullTestFiles=[fullTestFiles; extraFiles];
        fullTestLabels=[fullTestLabels; repmat(i, numel(extraFiles), 1)];
    end
end
nTest=numel(fullTestFiles);
Y_test=zeros(featureSize, nTest);
for i=1:nTest
    [audio, fs]=audioread(fullTestFiles{i});
    
    % MFCC
    c=mfcc(audio, fs);
    % Mel
    Sm=melSpectrogram(audio, fs, 'NumBands', 200, 'ApplyLog', true);
    % STFT
    [Ss, ~]=stft(audio, fs, 'Window', hamming(256), 'OverlapLength', 200, 'FFTLength', 398, 'FrequencyRange', 'onesided');
    
    y_raw=[mean(c,1)'; mean(Sm,2); mean(log10(abs(Ss)+1e-6),2)];
    Y_test(:, i)=(y_raw - meanD) ./ (stdD + 1e-6);
end
metric_names={'Euclidean', 'Cosine', 'Manhattan'};
fprintf('\n==============================\n');
fprintf('Full Test Set Results (Hybrid)\n');
fprintf('Total Test Files: %d\n', nTest);
fprintf('------------------------------\n');
for m=1:3
    preds=zeros(nTest, 1);
    for i=1:nTest
        y_vec=Y_test(:, i);
        switch m
            case 1 % Euclidean
                dists=vecnorm(D_norm - y_vec, 2, 1);
            case 2 % Cosine
                dists=1 - (D_norm' * y_vec) ./ (vecnorm(D_norm, 2, 1)' .* norm(y_vec) + eps);
            case 3 % Manhattan
                dists=sum(abs(D_norm - y_vec), 1);
        end
        [~, preds(i)]=min(dists);
    end
    acc=mean(preds == fullTestLabels) * 100;
    fprintf('%-12s : %.2f%%\n', metric_names{m}, acc);
end
fprintf('==============================\n');