%% %% mfcc mel soectrogram stft %% %%

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
featureSize=414; % 14mfcc, 200 mel and 200 stft 
D=zeros(featureSize, numStudents);

for i=1:numStudents
    currentID=uniqueStudents{i}; % for the student in the dictionary
    studentFiles=filepaths(strcmp(speakerIDs,currentID)); % finds files belonging to a particular student
    n=min(numFilesperstudent,length(studentFiles));
    student_feature=zeros(414,n);
    for j=1:n
        [audio, fs]=audioread(studentFiles{j});

        coeffs=mfcc(audio,fs);
        feat_mfcc=mean(coeffs,1)';

        S_mel=melSpectrogram(audio,fs,'NumBands',200,'ApplyLog',true);
        feat_mel=mean(S_mel,2);

        [S_stft, ~]=stft(audio, fs,'Window', hamming(256),'OverlapLength', 200,'FFTLength', 398,'FrequencyRange', 'onesided'); % Ensures real-valued logic
        feat_stft=mean(log10(abs(S_stft)+1e-6),2);

        student_feature(:,j)=[feat_mfcc; feat_mel; feat_stft];
    end

    
    D(:,i)=mean(student_feature,2); % mean coz we have to create a fixed length vector
end
meanD=mean(D,2);
stdD=std(D,0,2);
D_norm=(D-meanD)./(stdD+1e-6);
%% testing

numIterations=1000; 
correctCount=0;

for i=1:numIterations
    trueidx=randi(numStudents);
    testID=uniqueStudents{trueidx};
    testFiles=filepaths(strcmp(speakerIDs, testID));

    if length(testFiles)>10  % not used in training D
        fileIdx=randi([11,length(testFiles)]);
    else
        fileIdx=length(testFiles); 
    end

    % extract features
    [testAudio, testFs]=audioread(testFiles{fileIdx});

    c_test=mfcc(testAudio,testFs);
    f_mfcc=mean(c_test,1)';

    S_mel_test=melSpectrogram(testAudio,testFs,'NumBands',200,'ApplyLog',true);
    f_mel=mean(S_mel_test,2);

    [S_stft_test, ~]=stft(testAudio,testFs,"Window",hamming(256),"OverlapLength",200,'FFTLength', 398,'FrequencyRange', 'onesided');
    f_stft=mean(log10(abs(S_stft_test)+1e-6),2);

    y=[f_mfcc;f_mel;f_stft];
    y_norm=(y-meanD)./(stdD+1e-6);


    dists=zeros(1,numStudents);
    for k=1:numStudents
        dists(k)=norm(y_norm-D_norm(:,k),2);
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