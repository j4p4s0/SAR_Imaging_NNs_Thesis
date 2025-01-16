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


%% JAPS CODE

file = load('echo.mat', 'echo');
raw = file.echo;
clear file;

% Radar Params for RADARSAT: 
R0  = 988647.462; % Center Slant Range U: m
Vr  = 7062;       % Radar Velocity U: m/s
Tr  = 41.74e-6;   % Pulse Duration U: s
%Kr  = -0.72135e12;% Pulse Rate U: Hz/s -- ORIGINAL!!!!
Kr  = 0.72135e12;% Pulse Rate U: Hz/s
f0  = 5.3e9;      % Carrier (radar) Frequency U: Hz
Fr  = 32.317e6;   % Smapling Rate U: Hz
Fa  = 1256.98;    % Pulse Repetition Frequecny U: Hz
Naz = 4096;       % Range lines
Nrg = 4096;       % Smaples per Range
fc  = -596.271;   % Doppler centroid U: Hz


fc = f0;
PRF = Fa;
fs = Fr;
ch_R = Kr;
ch_T = Tr;
swst = 0;
    
%fc  = 0 ; % Doppler centroid U: Hz

%%

% Constants
c0 = 299792458;          % Speed of light [m/s]
lambda = c0/fc;          % SAR wavelength [m]
[Nl, Ns] = size(raw);    % Number of lines and samples

% Slant range [m] for each range pixel
%SR = (0 + (0:Ns-1)/fs)*c0/2;
SR = (swst + (0:Ns-1)/fs)*c0/2; %prof. Rui disse para por a 0 - apenas relevante para processamento em tempo real


% 59 | 631-646 | F16.7 | Slant range reference (for range spreading loss compensation) | 847.0 | km
%ir buscar ao ficheiro ".meta"

% Doppler frequency vector after FFT (assumes DC=0)
fdopp = (0:Nl-1)'/Nl*PRF;
idx_wrap = fdopp>=PRF/2;
fdopp(idx_wrap) = fdopp(idx_wrap) - PRF;


%% Range-Doppler Algorithm

%   1 - Range Compression - Range FFT is performed, a matched ﬁlter multiplication and, lastly, a range inverse fast Fourier transform"
%       S0(fτ, η) is the range FFT of sr and G(fτ) is the frequency domain matched ﬁlter
%       
%       src(τ, η) = IFFTτ[S0(fτ, η)G(fτ)]
%
%   2 - Azimuth FFT - Data transformed into the Range–Doppler domain with an azimuth FFT
%       
%       s1(τ, fη) = FFTη[Src(fτ, η)]
%
%   3 - Range Cell Migration Correction - azimuth compression along each parallel azimuth line
%
%       ∆R(fη) =(λ^2)Rt(fη^2) / 8(Vr^2)
%       
%       Rt - distance from the antenna to the target
%       Ka - azimuth FM rate of point target signal
%       fη = −Kaη
%       λ - wavelength of carrier frequency
%
%       s2(τ, fη) = A0 pr [τ − 2Rt/c] Wa(fη − fηc) × exp [−j 4πf0Rt(η)/c] exp [jπ (fη^2)/Ka]
%
%   4 - Azimuth Compression - matched ﬁlter is applied to the data after RCMC and, lastly, an IFFT is performed
%
%       s3(τ, fη) = S2(τ, η) Haz fη
%       Haz(fη) = exp [−jπ(fη^2)/Ka] - matched filter
%
%   5 - Azimuth IFFT - Transforms the data into the time domain
%
%       sac(τ, fη) = IFFTη S3(τ, fη)



%% Range compression
disp 'Range compression'

% Reference range chirp
t_rg = 0:1/fs:ch_T;
t_rg = t_rg - mean(t_rg);
chr_rg = exp(1i*pi*ch_R*t_rg.^2);

rgc = ifft(fft(raw, [], 2).*conj(fft(chr_rg, Ns)), [], 2);

% disp("Range Compression")
% Fr  = 60e6;       % Smapling Rate U: Hz
% Nrg = 320;        % Smaples per Range
% Kr  = 20e12;      % Pulse Rate U: Hz/s

% Frg = ((0:Nrg-1) - Nrg/2) / Nrg * Fr;
% GFilter = exp(1j * pi * ((Frg.^2) / Kr));
% data = ifty(fty(data) .* GFilter);
% disp("Range Compression done")
% data = ftx(data);
% disp("Azimuth FFT done")

%% Range cell migration correction
disp 'RCMC'

rgc_dopp = fft(rgc); % Range/Doppler domain
for k=1:Nl
    SR2 = SR + SR*(lambda*fdopp(k)).^2/(8*Vr^2);
    rgc_dopp(k,:) = interp1(SR, rgc_dopp(k,:), SR2, 'cubic', 0);
end

% disp("RCMC")
% Fr  = 60e6;       % Smapling Rate U: Hz
% Nrg = 320;        % Smaples per Range
% Frg = ((0:Nrg-1) - Nrg/2) / Nrg * Fr;

% Fa  = 100;        % Pulse Repetition Frequecny U: Hz
% Naz = 256;        % Range lines
% Faz = fc + ((0:Naz-1) - Naz/2) / Naz * Fa;
% lambda = C / f0; % Wavelength
% R0  = 20e3;       % Center Slant Range U: m
% dR = lambda^2 * R0 .* Faz.^2 / (8 * Vr^2);
% [Frg_2D, dR2D] = meshgrid(Frg, dR);
% G = exp(1j * 4 * pi * Frg_2D .* dR2D / C);
% data = data .* G;
% disp("RCMC done")

% disp("RCMC")
% Fr  = 60e6;       % Smapling Rate U: Hz
% Nrg = 320;        % Smaples per Range
% Frg = ((0:Nrg-1) - Nrg/2) / Nrg * Fr;

% PRF == Fa  = 100;        % Pulse Repetition Frequecny U: Hz
% Naz = 256;        % Range lines
% Faz = fc + ((0:Naz-1) - Naz/2) / Naz * Fa;
% lambda == lambda = C / f0; % Wavelength
% R0  = 20e3;       % Center Slant Range U: m
% dR = lambda^2 * R0 .* Faz.^2 / (8 * Vr^2);
% [Frg_2D, dR2D] = meshgrid(Frg, dR);
% G = exp(1j * 4 * pi * Frg_2D .* dR2D / C);
% data = data .* G;
% disp("RCMC done")


%% Azimuth compression
disp 'Azimuth compression'

dopp_R = -2*Vr^2/lambda./SR; % Doppler rate
slc = ifft(rgc_dopp.*exp(1i*pi*fdopp.^2./dopp_R));

% disp("Azimuth compression")
% Ka = 2 * Vr^2 / lambda / R0;
% H = exp(-1j * pi * Faz.^2 ./ Ka);
% H_2D = repmat(H.', [1, size(data, 2)]);  % Replicate H across the range dimension
% data = data .* H_2D;
% disp("Azimuth compression done")

disp 'END :)'


% Normalize the SAR image with the maximum pixel amplitude
SARImage = slc / max(slc(:));

% Display the final SAR image
figure; imshow(abs(SARImage), []); title('Final Raw (unedited) SAR Image');

% Image processing
% Constrast increase
disp("Image contrast increase")
SARImageEdit = imadjust(SARImage, [0.00015 0.06]);
disp("Image contrast increase done")

figure; imshow(abs(SARImageEdit), []); title('Constrast Increased');

% Circshift
disp("image shift")
SARImageShifted = circshift(SARImageEdit, -295, 1);
SARImageShifted = circshift(SARImageShifted, -580, 2);
disp("image shift done")
figure; imshow(abs(SARImageShifted), []); title('Image shifted');


% Despekle
disp("image despeckle")
SARImageSpek = specklefilt(SARImageShifted,DegreeOfSmoothing=0.2,NumIterations=50);
disp("image despeckle done")
figure; imshow(abs(SARImageSpek), []); title('Speckle Filter');

end
