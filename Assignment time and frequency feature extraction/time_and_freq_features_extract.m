close all
clc
clear all

%% Readme
% Please Install the following toolboxes before running the script
% Signal Processing toolbox
% Wavelet toolbox
% Audio Toolbox

%% load files
[value1 fs_value1]=audioread("Valve1.wav");
[speech1 fs_speech1]=audioread("Speech1.mp3");
[speech2 fs_speech2]=audioread("Speech2.mp3");
[music1 fs_music1]=audioread("Music1.mp3");
[music2 fs_music2]=audioread("Music2.mp3");
[gear1 fs_gear1]=audioread("Gear1.wav");

% normalize

value1=value1/max(abs(value1));
speech1=speech1/max(abs(speech1));
speech2=speech2/max(abs(speech2));
music1=music1/max(abs(music1));
music2=music2/max(abs(music2));
gear1=gear1/max(abs(gear1));

%% get time domain features
td_value1=time_domain_fearures(value1,fs_value1);
td_speech1=time_domain_fearures(speech1,fs_speech1);
td_speech2=time_domain_fearures(speech2,fs_speech2);
td_music1=time_domain_fearures(music1,fs_music1);
td_music2=time_domain_fearures(music2,fs_music2);
td_gear1=time_domain_fearures(gear1,fs_gear1);

fd_value1=frequency_domain_features(value1,fs_value1);
fd_speech1=frequency_domain_features(speech1,fs_speech1);
fd_speech2=frequency_domain_features(speech2,fs_speech2);
fd_music1=frequency_domain_features(music1,fs_music1);
fd_music2=frequency_domain_features(music2,fs_music2);
fd_gear1=frequency_domain_features(gear1,fs_gear1);

tfd_value1=time_freq_features(value1,fs_value1);
tfd_speech1=time_freq_features(speech1,fs_speech1);
tfd_speech2=time_freq_features(speech2,fs_speech2);
tfd_music1=time_freq_features(music1,fs_music1);
tfd_music2=time_freq_features(music2,fs_music2);
tfd_gear1=time_freq_features(gear1,fs_gear1);

%% plots

signals     = {value1, speech1, speech2, music1, music2, gear1};
fs_list     = [fs_value1, fs_speech1, fs_speech2, fs_music1, fs_music2, fs_gear1];
names       = {'Valve1','Speech1','Speech2','Music1','Music2','Gear1'};
td_all      = {td_value1, td_speech1, td_speech2, td_music1, td_music2, td_gear1};
fd_all      = {fd_value1, fd_speech1, fd_speech2, fd_music1, fd_music2, fd_gear1};
tfd_all     = {tfd_value1, tfd_speech1, tfd_speech2, tfd_music1, tfd_music2, tfd_gear1};
colors      = {'#2166ac','#d6604d','#4dac26','#762a83','#e08214','#1a9850'};
n           = numel(signals);

%% --- Waveforms ---
figure('Name','Waveforms','Position',[50 50 1400 700]);
for i = 1:n
    subplot(2,3,i);
    fs  = fs_list(i);
    sig = signals{i};
    t   = (0:length(sig)-1) / fs;
    plot(t, sig, 'Color', colors{i}, 'LineWidth', 0.6);
    title(names{i}); xlabel('Time (s)'); ylabel('Amplitude');
    ylim([-1.1 1.1]); grid on; box on;
end
sgtitle('Waveforms — All Signals');

%% --- Amplitude Envelope (Time Domain) ---
FRAME_SIZE = round(0.025 * fs_list(1));
HOP_LENGTH = round(0.010 * fs_list(1));

figure('Name','Amplitude Envelope','Position',[50 50 1400 700]);
for i = 1:n
    subplot(2,3,i);
    fs  = fs_list(i);
    sig = signals{i};
    t   = (0:length(sig)-1) / fs;

    % Recompute envelope for plotting
    frame_len = round(0.025*fs);
    hop_len   = round(0.010*fs);
    frames_p  = buffer(sig, frame_len, frame_len-hop_len, 'nodelay');
    env       = max(abs(frames_p));
    n_fr      = length(env);
    t_fr      = ((0:n_fr-1) * hop_len) / fs;

    plot(t, sig, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5); hold on;
    plot(t_fr, env, 'Color', colors{i}, 'LineWidth', 1.8);
    title(names{i}); xlabel('Time (s)'); ylabel('Amplitude');
    ylim([-0.05 1.1]); legend('Waveform','Envelope','Location','northeast');
    grid on; box on;
end
sgtitle('Amplitude Envelope — All Signals');

%% --- Zero Crossing Rate (Time Domain) ---
figure('Name','Zero Crossing Rate','Position',[50 50 1400 700]);
for i = 1:n
    subplot(2,3,i);
    fs        = fs_list(i);
    sig       = signals{i};
    frame_len = round(0.025*fs);
    hop_len   = round(0.010*fs);
    frames_p  = buffer(sig, frame_len, frame_len-hop_len, 'nodelay');
    zcr       = sum(abs(diff(sign(frames_p)))) / (2*frame_len);
    n_fr      = length(zcr);
    t_fr      = ((0:n_fr-1) * hop_len) / fs;

    plot(t_fr, zcr, 'Color', colors{i}, 'LineWidth', 1.2);
    title(sprintf('%s  |  mean=%.3f', names{i}, mean(zcr)));
    xlabel('Time (s)'); ylabel('ZCR'); grid on; box on;
end
sgtitle('Zero Crossing Rate — All Signals');


%% --- FFT Magnitude Spectrum (Frequency Domain) ---
figure('Name','FFT Spectrum','Position',[50 50 1400 700]);
for i = 1:n
    subplot(2,3,i);
    plot(fd_all{i}.freq_axis, fd_all{i}.fft_magnitude, ...
         'Color', colors{i}, 'LineWidth', 0.7);
    xlabel('Frequency (Hz)'); ylabel('Magnitude');
    title(names{i}); xlim([0, fs_list(i)/2]);
    grid on; box on;
end
sgtitle('FFT Magnitude Spectrum — All Signals');

%% --- Spectrogram / STFT (Time-Frequency) ---
figure('Name','Spectrograms','Position',[50 50 1400 700]);
for i = 1:n
    subplot(2,3,i);
    S_dB = 20 * log10(tfd_all{i}.stft_mag + eps);
    imagesc(tfd_all{i}.stft_time, tfd_all{i}.stft_freq, S_dB);
    axis xy; colormap(gca, 'jet');
    cb = colorbar; cb.Label.String = 'dB';
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title(names{i});
    ylim([0, fs_list(i)/2]);
    clim([max(S_dB(:))-80, max(S_dB(:))]);   % 80 dB dynamic range
end
sgtitle('STFT Spectrogram — All Signals');

%% --- CWT Scalogram (Time-Frequency) ---
figure('Name','CWT Scalogram','Position',[50 50 1400 700]);
for i = 1:n
    subplot(2,3,i);
    fs  = fs_list(i);
    sig = signals{i};
    t   = (0:length(sig)-1) / fs;
    n_scales = length(tfd_all{i}.cwt_freq);

    imagesc(t, 1:n_scales, tfd_all{i}.cwt_mag);
    axis xy; colormap(gca, 'parula');
    colorbar;

    % Label y-axis with actual frequencies (sparse ticks)
    tick_idx = round(linspace(1, n_scales, 6));
    set(gca, 'YTick', tick_idx, ...
             'YTickLabel', arrayfun(@(x) sprintf('%.0f', x), ...
                           tfd_all{i}.cwt_freq(tick_idx), 'UniformOutput', false));
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title(names{i});
end
sgtitle('CWT Scalogram — All Signals');





%% time domain features

function features=time_domain_fearures(signal,fs)
    N=length(signal);

    % RMS energy
    features.rms=sqrt(mean(signal.^2));

    % zero crossing rate
    frame_len=round(0.025*fs);
    hop_len=round(0.010*fs);
    frames=buffer(signal,frame_len,frame_len-hop_len,"nodelay");
    zcr=sum(abs(diff(sign(frames))))/(2*frame_len);
    features.zcr_mean=mean(zcr);
    features.zcr_std=std(zcr);

    % amplitude envelope
    features.amp_env=max(abs(frames));

end

%% function for freq features

function features=frequency_domain_features(signal,fs)

    N=length(signal);
    Y=fft(signal);
    Y=Y(1:floor(N/2)+1);
    P=(1/(fs*N))*abs(Y).^2;
    P(2:end-1)=2*P(2:end-1);
    f=(0:floor(N/2))*(fs/N);

    % mfcc
    win_len=round(0.025*fs);
    coeffs=mfcc(signal, fs, "NumCoeffs", 13, "Window", hann(win_len,'periodic'), "OverlapLength", round(0.015*fs));
    features.mfcc_mean=mean(coeffs,1);
    features.mfcc_std=std(coeffs,0,1);

    features.fft_magnitude=abs(Y);
    features.freq_axis=f;
end

%% time freqency domain

function features=time_freq_features(signal,fs)
    win_len=round(0.025*fs);
    hop_len=round(0.010*fs);
    nfft=512;

    [S F T]=spectrogram(signal,hann(win_len),win_len-hop_len,nfft,fs);
    features.stft_mag=abs(S);
    features.stft_freq=F;
    features.stft_time=T;

    % CWT
    [wt, freq]=cwt(signal,'amor',fs);
    features.cwt_mag=abs(wt);
    features.cwt_freq=freq;

end






