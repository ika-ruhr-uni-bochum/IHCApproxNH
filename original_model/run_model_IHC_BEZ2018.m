clc
clear all

audio_file = '../audio/61-70968-0000.flac';

% model fiber parameters
numcfs      = 80;
CFs         = logspace(log10(125),log10(8e3),numcfs);  % CF in Hz;
species     = 2;                                        % Human cochlear tuning 

stimdb      = 60;
stimT       = 2;
[audio, fs] = audioread(audio_file);

% extract audio segment
audio       = audio(1:fs*stimT);
% audio       = audio(11000:22000);

% normalize audio segment
audio       = audio/rms(audio)*20e-6*10^(stimdb/20);

tic
vihc_mat    = generate_ihcgram_BEZ2018_parallelized(audio,fs,species,numcfs,CFs);
% convert to single precision
%     vihc_mat = cast(vihc_mat,'single'); 
toc
    

