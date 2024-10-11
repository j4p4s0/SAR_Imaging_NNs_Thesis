function slc = RD_SAR_focus(raw, Vr, fc, PRF, fs, swst, ch_R, ch_T)
% slc = SAR_focus(raw, Vr, fc, PRF, fs, swst, ch_R, ch_T)
%
% This function implements a basic range-Doppler algorithm to focus a
% single-look-complex (SLC) image out of raw SAR data.
%
% Inputs:
%   raw: baseband RAW data (complex pixels, range along rows)
%   Vr:  sensor velocity [m/s]
%   fc:  SAR central frequency [Hz]
%   PRF: pulse repetition frequency [Hz]
%   fs:  range sampling frequency [Hz]
%   swst: sampling window start time [s] (fast-time to first sample)
%   ch_R: range chirp rate [Hz/s] (ch_R>0 for up-chirp)
%   ch_T: range chirp duration [s]
%
% Outputs:
%   slc: single-look-complex focused image
%
% Notes:
%   - Non-squinted SAR data is assumed
%   - Transients after focusing are not removed by this function
%
% Author: Mario Azcueta <mazcueta@gmail.com> (Feb 2014, updated Apr 2022)


% Constants
c0 = 299792458;          % Speed of light [m/s]
lambda = c0/fc;          % SAR wavelength [m]
[Nl, Ns] = size(raw);    % Number of lines and samples

% Slant range [m] for each range pixel
SR = (swst + (0:Ns-1)/fs)*c0/2; %colocar a 0

% Doppler frequency vector after FFT (assumes DC=0)
fdopp = (0:Nl-1)'/Nl*PRF;
idx_wrap = fdopp>=PRF/2;
fdopp(idx_wrap) = fdopp(idx_wrap) - PRF;


%% Range compression
disp 'Range compression'

% Reference range chirp
t_rg = 0:1/fs:ch_T; t_rg = t_rg - mean(t_rg);
chr_rg = exp(1i*pi*ch_R*t_rg.^2);

rgc = ifft(fft(raw, [], 2).*conj(fft(chr_rg, Ns)), [], 2);


%% Range cell migration correction
disp 'RCMC'

rgc_dopp = fft(rgc); % Range/Doppler domain
for k=1:Nl
    SR2 = SR + SR*(lambda*fdopp(k)).^2/(8*Vr^2);
    rgc_dopp(k,:) = interp1(SR, rgc_dopp(k,:), SR2, 'cubic', 0);
end


%% Azimuth compression
disp 'Azimuth compression'

dopp_R = -2*Vr^2/lambda./SR; % Doppler rate
slc = ifft(rgc_dopp.*exp(1i*pi*fdopp.^2./dopp_R));

disp 'END :)'

end
