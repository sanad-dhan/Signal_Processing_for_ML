%% MEL SPECTROGRAM Dictionary 
close all;
clc; 
clear all;
%% LOAD DATA 
ads=audioDatastore('E:\Sanad sem 8\SP4ML\Project\speech_data','IncludeSubfolders', true);
filepaths=ads.Files;
[~, names, ~]=cellfun(@fileparts, filepaths, 'UniformOutput', false);
speakerIDs=cellfun(@(x) strtok(x, '_'), names, 'UniformOutput', false);
uniqueStudents=unique(speakerIDs);

%% CONFIGURATION
numStudents=100;
numFilesPerStudent=10;
testStartIdx=11;
featureSize=200;
numIterations=1000;

%% BUILD DICTIONARY 
D=zeros(featureSize, numStudents);
for i=1:numStudents
    currentID=uniqueStudents{i};
    studentFiles=filepaths(strcmp(speakerIDs, currentID));
    n=min(numFilesPerStudent, length(studentFiles));
    tempmat=zeros(featureSize, n);
    for j=1:n
        [audio, fs]=audioread(studentFiles{j});
        S=melSpectrogram(audio, fs, 'NumBands', featureSize, 'ApplyLog', true);
        logS=log10(S + 1e-6);
        tempmat(:, j)=mean(logS, 2);
    end
    D(:, i)=mean(tempmat, 2);
end
% Normalize dictionary
meanD=mean(D, 2);
stdD=std(D, 0, 2);
D=(D - meanD) ./ (stdD + 1e-6);
fprintf('Dictionary built: %d x %d\n', size(D));

%% 1000-TRIAL TEST
correctEuc=0;
correctCos=0;
correctMan=0;
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
    S_test=melSpectrogram(testAudio, testFs, 'NumBands', featureSize, 'ApplyLog', true);
    y=mean(log10(S_test + 1e-6), 2);
    y=(y - meanD) ./ (stdD + 1e-6);   % same normalization as D
    distsEuc=zeros(1, numStudents);
    distsCos=zeros(1, numStudents);
    distsMan=zeros(1, numStudents);
    for k=1:numStudents
        distsEuc(k)=norm(y - D(:,k), 2);
        distsCos(k)=1 - dot(y, D(:,k)) / (norm(y) * norm(D(:,k)) + eps);
        distsMan(k)=norm(y - D(:,k), 1);
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
fprintf('Euclidean  : %.2f%%\n', correctEuc/numIterations*100);
fprintf('Cosine     : %.2f%%\n', correctCos/numIterations*100);
fprintf('Manhattan  : %.2f%%\n', correctMan/numIterations*100);
fprintf('==============================\n');

%% FULL TEST EVALUATION
fprintf('\nBuilding full test set...\n');
testFiles={};
testLabels=[];
for i=1:numStudents
    currentID=uniqueStudents{i};
    studentFiles=filepaths(strcmp(speakerIDs, currentID));
    if length(studentFiles) >= testStartIdx
        extraFiles=studentFiles(testStartIdx:end);
        testFiles=[testFiles; extraFiles];
        testLabels=[testLabels; repmat(i, numel(extraFiles), 1)];
    end
end
nTest=numel(testFiles);
fprintf('Test files: %d\n', nTest);
% Extract and normalize features for all test files
Y_test=zeros(featureSize, nTest);
for i=1:nTest
    [audio, fs]=audioread(testFiles{i});
    S=melSpectrogram(audio, fs, 'NumBands', featureSize, 'ApplyLog', true);
    y=mean(log10(S + 1e-6), 2);
    Y_test(:, i)=(y - meanD) ./ (stdD + 1e-6);
end
% Evaluate all three metrics
metrics={'euclidean', 'cosine', 'cityblock'};
metric_names={'Euclidean', 'Cosine', 'Manhattan'};
pred_store=zeros(nTest, 3);
fprintf('\n==============================\n');
fprintf('Full Test Set Results\n');
fprintf('Test files  : %d\n', nTest);
fprintf('Speakers    : %d\n', numStudents);
fprintf('------------------------------\n');
for d=1:3
    for i=1:nTest
        y=Y_test(:, i);
        switch metrics{d}
            case 'euclidean'
                dists=vecnorm(D - y, 2, 1);
            case 'cosine'
                dists=1 - (D' * y) ./ (vecnorm(D,2,1)' .* norm(y) + eps);
            case 'cityblock'
                dists=sum(abs(D - y), 1);
        end
        [~, pred_store(i,d)]=min(dists);
    end
    acc=mean(pred_store(:,d) == testLabels) * 100;
    fprintf('%-12s : %.2f%%\n', metric_names{d}, acc);
end
fprintf('==============================\n');