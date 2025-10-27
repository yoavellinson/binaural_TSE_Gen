room.humidity               = 0.42;         % relative humidity (0,...,1)
room.temperature            = 20;           % room temperature (celsius)
room.surface.frequency      = [  125       250       500       1000      2000      4000];


room.surface.diffusion      = [  0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0   ; 
                                 0.0       0.0       0.0       0.0       0.0       0.0    ];

options.fs                  = 16000;                % sampling frequency in Hz
options.responseduration    = 1.8;                 % duration of impulse response
options.bandsperoctave      = 4;                    % simulation frequency accuracy (1, 2, 3, or 4 bands/octave)
options.referencefrequency  = 125;                  % reference frequency for frequency octaves
options.airabsorption       = true;                 % apply air absorption?
options.distanceattenuation = true;                 % apply distance attenuation?
options.subsampleaccuracy   = true;                % apply subsample accuracy?
options.highpasscutoff      = 1;                    % 3dB frequency of high-pass filter (0=none)
options.verbose             = false;                 % print status messages?

options.simulatespecular    = true;                 % simulate specular reflections?

options.simulatediffuse     = false;                 % simulate diffuse reflections?
options.numberofrays        = 2000;                 % number of rays in simulation (20*K^2)
options.diffusetimestep     = 0.010;                % time resolution in diffuse energy histogram (seconds)
options.rayenergyfloordB    = -80;                  % ray energy threshold (dB, with respect to initial energy)
options.uncorrelatednoise   = true;                 % use uncorrelated poisson arrivals for binaural impulse responses?

options.mex_saveaswav       = false;                % enable or disable saving the results of sofamyroom on disk
                                                    % when using MATLAB
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/axd/p0032.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.2 0.62 0.98 0.62 0.62 0.62;
                                    0.37 0.85 1.0 1.0 1.0 1.0;
                                    0.34 0.67 0.98 1.0 1.0 1.0;
                                    0.3 0.3 0.6 0.85 0.75 0.75;
                                    0.3 0.7 0.85 0.9 0.85 0.8;
                                    0.04 0.1 0.07 0.06 0.07 0.07];
receiver(1).orientation      = [ 13.830182727205852 0 0 ];
receiver(1).location         = [ 5.351066182349821 6.9544919307532815 1.7659277363412764 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 5.962807053574127 8.534121670050018 1.7659277363412764 ];
source(1).orientation        = [ -111.16981727279415 -0.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 4.970540118873347 8.500201014898549 1.4852399575450468 ];
source(2).orientation        = [ -76.16981727279415 10.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 10 10 10 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/axd_p0032_az155.0_elev10.0_az290.0_elev2-10.0/rt_60_0.23987449425654647/h_rt60';
