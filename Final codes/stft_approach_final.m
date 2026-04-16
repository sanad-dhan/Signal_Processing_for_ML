%% %% STFT Speaker Recognition with Multi-Metric Evaluation %% %%
close all
clc
clear all
%% Loading Data
ads=audioDatastore('E:\Sanad sem 8\SP4ML\Project\speech_data', 'IncludeSubfolders', true);
filepaths=ads.Files;
[~, names, ~]=cellfun(@fileparts, filepaths, 'UniformOutput', false); 
speakerIDs=cellfun(@(x) strtok(x, '_'), names, 'UniformOutput', false); 
uniqueStudents=unique(speakerIDs);

%% Dictionary Creation
numStudents=100; 
numFilesperstudent=10;
featureSize=200;
testStartIdx=numFilesperstudent+1; 
D=zeros(featureSize, numStudents);
fprintf('Building Dictionary...\n');
for i=1:numStudents
    currentID=uniqueStudents{i};
    studentFiles=filepaths(strcmp(speakerIDs, currentID));
    n=min(numFilesperstudent, length(studentFiles));
    tempmat=zeros(featureSize, n);
    
    for j=1:n
        [audio, fs]=audioread(studentFiles{j});
        [S, ~]=stft(audio, fs, "Window", hamming(256), "OverlapLength", 200);
        % Use magnitude then log
        logS=log10(abs(S) + 1e-6);
        tempmat(:, j)=mean(logS(1:featureSize, :), 2);
    end
    D(:, i)=mean(tempmat, 2); 
end
% Normalization parameters based on Dictionary
meanD=mean(D, 2);
stdD=std(D, 0, 2);
D_norm=(D - meanD) ./ (stdD + 1e-6);
fprintf('Dictionary D successfully created with size %d x %d\n', size(D,1), size(D,2));

%% 1000-TRIAL TEST 
numIterations=1000;
correctEuc=0; correctCos=0; correctMan=0;
fprintf('\nRunning %d Random Trials...\n', numIterations);
for i=1:numIterations
    trueIdx=randi(numStudents);
    testID=uniqueStudents{trueIdx};
    testFiles=filepaths(strcmp(speakerIDs, testID));
    
    % Pick a random file from the test set (files after the 10th)
    if length(testFiles) >= testStartIdx
        fileIdx=randi([testStartIdx, length(testFiles)]);
    else
        fileIdx=length(testFiles);
    end
    
    [testAudio, testFs]=audioread(testFiles{fileIdx});
    [S_test, ~]=stft(testAudio, testFs, "Window", hamming(256), "OverlapLength", 200);
    
    % Feature extraction matching Dictionary
    y=mean(log10(abs(S_test(1:featureSize, :)) + 1e-6), 2);
    y_norm=(y - meanD) ./ (stdD + 1e-6); 
    % Calculate Distances
    distsEuc=zeros(1, numStudents);
    distsCos=zeros(1, numStudents);
    distsMan=zeros(1, numStudents);
    
    for k=1:numStudents
        % Euclidean (L2)
        distsEuc(k)=norm(y_norm - D_norm(:, k), 2);
        % Cosine
        distsCos(k)=1 - dot(y_norm, D_norm(:, k)) / (norm(y_norm) * norm(D_norm(:, k)) + eps);
        % Manhattan (L1)
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
fprintf('1000-Trial Results\n');
fprintf('Dictionary : %d students\n', numStudents);
fprintf('------------------------------\n');
fprintf('Euclidean  : %.2f%%\n', (correctEuc/numIterations)*100);
fprintf('Cosine     : %.2f%%\n', (correctCos/numIterations)*100);
fprintf('Manhattan  : %.2f%%\n', (correctMan/numIterations)*100);
fprintf('==============================\n');

%% FULL TEST EVALUATION
fprintf('\nBuilding full test set (all files after index %d)...\n', testStartIdx-1);
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
    [S, ~]=stft(audio, fs, "Window", hamming(256), "OverlapLength", 200);
    y=mean(log10(abs(S(1:featureSize, :)) + 1e-6), 2);
    Y_test(:, i)=(y - meanD) ./ (stdD + 1e-6);
end
% Evaluate across the entire test matrix
metric_names={'Euclidean', 'Cosine', 'Manhattan'};
fprintf('\n==============================\n');
fprintf('Full Test Set Results\n');
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