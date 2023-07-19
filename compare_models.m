clc
clear all

addpath('./original_model')

audio_file = './audio/61-70968-0000.flac';

% target sampling frequency
fs          = 16000;

% model fiber parameters
numcfs      = 80;
CFs         = logspace(log10(125),log10(8e3),numcfs);  % CF in Hz;
species     = 2;                                       % Human cochlear tuning 

% spefiy SPL and total signal duration
stimdb      = 60;
stimT       = 3;

% segment duration in seconds (CPU recommendation: 0.5 or 1, CUDA recommendation: as high as GPU memory allows)
segdur      = 1; 

% read audio signal and resample if fsOrig differs from fs
[audio, fsOrig] = audioread(audio_file);
if fsOrig ~= fs
    audio = resample(audio,fs,fsOrig);
end

% enable MATLAB comparison for python code:
matlabComp  = 1;

% select processing unit ('cpu' or 'cuda')
procUnit    = 'cpu';

% extract audio segment
audio       = audio(1:fs*stimT);

% normalize audio segment
audio       = audio/rms(audio)*20e-6*10^(stimdb/20);

% processing of original model
disp('Processing original auditory model in MATLAB...')
disp('======================================================')
tic
vihc_mat    = generate_ihcgram_BEZ2018_parallelized(audio,fs,species,numcfs,CFs);
elapsedTime = toc;
disp(['Processing time for original auditory model: ' num2str(elapsedTime) ' s'])

% discard first 2047 samples for better comparion with WaveNet model
vihc_mat    = vihc_mat(:,2048:end);

% processing of WaveNet-based approximation
disp(' ')
disp('Processing WaveNet model in Python/PyTorch...')
disp('======================================================')
system(['python3 run_model.py --file ' audio_file ' --spl ' num2str(stimdb) ' --totdur ' num2str(stimT)...
    ' --segdur ' num2str(segdur) ' --matlabComp ' num2str(matlabComp) ' --proc ' procUnit]);
load('tmp.mat')

% approximation quality
disp(' ')
disp('Approximation quality in terms of SDR:')
SDR = mean(20*log10(rms(vihc_mat,2)./rms(vihc_mat-ihcogram,2)));
disp(['SDR = ' num2str(SDR) ' dB'])

% plotting
climlow = -0.0025;
climhigh = 0.06;
t = [2048:stimT*fs]/fs;
tickFreqs = [125,250,500,1000,2000,4000,8000];
tickPosVec = zeros(length(tickFreqs),1);
for m = 1:length(tickFreqs)
    [~,ind] = min(abs(CFs-tickFreqs(m)));
    tickPosVec(m) = ind;
end

figure()
tiledlayout(3,1)
nexttile
imagesc(t,[],vihc_mat)
yticks(tickPosVec)
yticklabels(tickFreqs/1000)    
ylabel('CF/kHz')
axis xy
caxis([climlow,climhigh])
title('Original IHCogram','FontWeight','normal')
% colormap(flipud(hot))
colorbar
nexttile
imagesc(t,[],ihcogram)
yticks(tickPosVec)
yticklabels(tickFreqs/1000)    
ylabel('CF/kHz')
axis xy
caxis([climlow,climhigh])
colorbar
title('Predicted IHCogram','FontWeight','normal')
nexttile
imagesc(t,[],vihc_mat-ihcogram)
yticks(tickPosVec)
yticklabels(tickFreqs/1000)    
ylabel('CF/kHz')
axis xy
caxis([climlow,climhigh])
colorbar
title('Prediction error','FontWeight','normal')
xlabel('time/s')

!rm tmp.mat
