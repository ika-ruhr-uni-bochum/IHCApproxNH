function vihc_mat = generate_ihcgram_BEZ2018_parallelized(stim,Fs_stim,species,numcfs,CFs)

% Normal hearing
ag_fs = [125 250 500 1e3 2e3 4e3 8e3];
ag_dbloss = [0 0 0 0 0 0 0];                        

% mixed loss
dbloss = interp1(ag_fs,ag_dbloss,CFs,'linear','extrap');
[cohcs,cihcs]=fitaudiogram2(CFs,dbloss,species);

% cohcs  = ones(1,numcfs);  % normal ohc function
% cihcs  = ones(1,numcfs);  % normal ihc function

% stimulus parameters
Fs = 100e3;  % sampling rate in Hz (must be 100, 200 or 500 kHz)
stim100k = resample(stim,Fs,Fs_stim).';
T  = length(stim100k)/Fs;  % stimulus duration in seconds

% PSTH parameters
nrep = 1;
psthbinwidth_mr = 100e-6; % mean-rate binwidth in seconds;

simdur = ceil(T*1.2/psthbinwidth_mr)*psthbinwidth_mr;

% Run model_IHC_BEZ2018 function to estimate the size of vihc and variable.  
vihc = model_IHC_BEZ2018(stim100k,CFs(1),nrep,1/Fs,simdur,cohcs(1),cihcs(1),species);
vihc_mat=zeros(numcfs,length(vihc));
clear vihc

% loop over all CFs
parfor CFind = 1:numcfs
    
    CF = CFs(CFind);
    cohc = cohcs(CFind);
    cihc = cihcs(CFind);
        
    vihc = model_IHC_BEZ2018(stim100k,CF,nrep,1/Fs,simdur,cohc,cihc,species);
    vihc_mat(CFind,:) = vihc;
    
end

vihc_mat = vihc_mat(:,1:round(T*Fs));

% reduce sampling rate
vihc_mat = resample(vihc_mat',Fs_stim,Fs)';

