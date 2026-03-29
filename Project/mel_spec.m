%% %% MEL SPECTROGRAM multiple %% %%

close all
clc
clear all

%% loading data
ads=audioDatastore('E:\Sanad sem 8\SP4ML\Project\speech_data', 'IncludeSubfolders', true);

%% get speaker ids
filepaths=ads.Files;
[~, names, ~]=cellfun(@fileparts, filepaths, 'UniformOutput', false);  % extracts filepath name and extension
speakerIDs=cellfun(@(x) strtok(x, '_'), names, 'UniformOutput', false); % gives student id, all before _ underscore
uniqueStudents=unique(speakerIDs);


%% create dictionary
numStudents=100; % can be modified as per requirements
numFilesperstudent=10;
featureSize=200; % 
D=zeros(featureSize, numStudents);

for i=1:numStudents
    currentID=uniqueStudents{i}; % for the student in the dictionary
    studentFiles=filepaths(strcmp(speakerIDs,currentID)); % finds files belonging to a particular student
    n=min(numFilesperstudent,length(studentFiles));
    tempmat=zeros(featureSize,n);
    
    for j=1:n
        [audio, fs]=audioread(studentFiles{j});
        S=melSpectrogram(audio,fs,'NumBands',featureSize,'ApplyLog',true);
        logS=log10(S+1e-6);
        tempmat(:,j)=mean(logS,2);
    end

    
    D(:,i)=mean(tempmat,2)'; % mean coz we have to create a fixed length vector
end

meanD=mean(D,2);
stdD=std(D,0,2);
D=(D-meanD)./(stdD+1e-6);

fprintf('Dictionary D successfully created with size %d x %d\n', size(D,1), size(D,2));
% every column of D corresponds to a sudent

%% testing for 1000 comparisons

numIterations=1000; 
correctCount=0;

for i=1:numIterations
    trueidx=randi(numStudents);
    testID=uniqueStudents{trueidx};
    testFiles=filepaths(strcmp(speakerIDs, testID));

    if length(testFiles)>10  % not used in training D
        fileIdx=randi([11,length(testFiles)]);
    else
        % If a student only has 1 file total, we skip this iteration or use the same file (but accuracy will be artificially high)
        fileIdx = length(testFiles); 
    end

    % extract features
    [testAudio, testFs]=audioread(testFiles{fileIdx});
    S_test=melSpectrogram(testAudio,testFs,'NumBands',featureSize,'ApplyLog',true);
    y=mean(log10(S_test+1e-6),2);

    y=(y-meanD)./(stdD+1e-6);

    dists=zeros(1,numStudents);
    for k=1:numStudents
        dists(k)=norm(y-D(:,k),2);
    end

    [~,recognizedidx]=min(dists);
    if recognizedidx==trueidx
        correctCount=correctCount+1;
    end

end

% --- Final Stats ---
totalAccuracy = (correctCount / numIterations) * 100;
fprintf('==============================\n')
fprintf('Dictionary Size: %d Students\n', numStudents);
fprintf('Trials Run: %d \n', numIterations);
fprintf('Correct Matches: %d\n', correctCount);
fprintf('Estimated Accuracy: %.2f%%\n', totalAccuracy);
fprintf('==============================\n')