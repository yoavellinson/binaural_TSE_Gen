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
receiver(1).description      = 'SOFA /home/workspace/yoavellinson/binaural_TSE_Gen/sofas/ari_atl_and_full/hrtf_b_nh110.sofa interp=1 norm=1 resampling=1';
room.surface.absorption  = [0.3 0.3 0.6 0.85 0.75 0.75;
                                    0.18 0.42 0.55 0.77 0.87 0.94;
                                    0.53 0.92 1.0 1.0 1.0 1.0;
                                    0.53 0.92 1.0 1.0 1.0 1.0;
                                    0.3 0.7 0.85 0.9 0.85 0.8;
                                    0.1 0.25 0.3 0.3 0.3 0.3];
receiver(1).orientation      = [ 83.04907633381309 0 0 ];
receiver(1).location         = [ 5.660496673152085 4.481352909930547 1.5833923071200295 ]; 
room.dimension              = [ 10 10 2.8 ];

source(1).location           = [ 4.61342484436291 5.422517896472433 1.4602179953352161 ];
source(1).orientation        = [ -41.95092366618691 5.0 0 ];
source(1).description        = 'subcardioid';
source(2).location           = [ 6.323527697179702 5.939550325104682 1.3009414790707865 ];
source(2).orientation        = [ 245.5490763338131 10.0 0 ];
source(2).description        = 'subcardioid';
options.reflectionorder     = [ 10 10 10 ];
options.outputname			= '/home/workspace/yoavellinson/binaural_TSE_Gen/scripts/reverbration_simulator/hrtf/ari_atl_and_full_hrtf_b_nh110_az155.0_elev1-5.0_az2342.5_elev2-10.0/rt_60_0.21139898488970424/h_rt60';
