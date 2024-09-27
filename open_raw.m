outputFolder = pwd;
dataURL = ['https://ssd.mathworks.com/supportfiles/radar/data/' ...
    'ERSData.zip'];
helperDownloadERSData(outputFolder,dataURL);

% Speed of light
c = physconst('LightSpeed');
    
% Extract ERS system parameters
[fs,fc,prf,tau,bw,v,ro,fd] = ERSParameterExtractor('E2_84699_STD_L0_F299.000.ldr');
    
% Extract raw data 
rawData = ERSDataExtractor('E2_84699_STD_L0_F299.000.raw',fs,fc,prf,tau,v,ro,fd).';

function helperDownloadERSData(outputFolder,DataURL)
% Download the data set from the given URL to the output folder.

    radarDataZipFile = fullfile(outputFolder,'ERSData.zip');
    
    if ~exist(radarDataZipFile,'file')
        
        disp('Downloading ERS data (134 MiB)...');
        websave(radarDataZipFile,DataURL);
        unzip(radarDataZipFile,outputFolder);
    end
end

function [fs,fc,prf,tau,bw,veff,ro,fdop] = ERSParameterExtractor(file)
    % Open the parameter file to extract required parameters
    fid = fopen(file,'r');
    
    % Radar wavelength (satellite specific)
    status = fseek(fid,720+500,'bof');
    lambda = str2double(fread(fid,[1 16],'*char'));         % Wavelength (m)
    
    % Pulse Repetition Frequency (satellite specific)
    status = fseek(fid,720+934,'bof')|status;
    prf = str2double(fread(fid,[1 16],'*char'));            % PRF (Hz)
    
    % Range sampling rate (satellite specific)
    status = fseek(fid,720+710,'bof')|status;
    fs =str2double(fread(fid,[1 16],'*char'))*1e+06;        % Sampling Rate (Hz)
    
    % Range Pulse length (satellite specific)
    status = fseek(fid,720+742,'bof')|status;
    tau = str2double(fread(fid,[1 16],'*char'))*1e-06;      % Pulse Width (sec)
    
    % Range Gate Delay to first range cell
    status = fseek(fid,720+1766,'bof')|status;
    rangeGateDelay = str2double(fread(fid,[1 16],'*char'))*1e-03;   % Range Gate Delay (sec)
    
    % Velocity X
    status = fseek(fid,720+1886+452,'bof')|status;
    xVelocity = str2double(fread(fid,[1 22],'*char'));    % xVelocity (m/sec)
    
    % Velocity Y
    status = fseek(fid,720+1886+474,'bof')|status;
    yVelocity = str2double(fread(fid,[1 22],'*char'));    % yVelocity (m/sec)
    
    % Velocity Z
    status = fseek(fid,720+1886+496,'bof')|status;
    zVelocity = str2double(fread(fid,[1 22],'*char'));    % zVelocity (m/sec)
    fclose(fid);
    
    % Checking for any file error
    if(status==1)
        fs = NaN;
        fc = NaN;
        prf = NaN;
        tau = NaN;
        bw = NaN;
        veff = NaN;
        ro = NaN;
        fdop = NaN;
        return;
    end
    
    % Values specific to ERS satellites
    slope = 4.19e+11;           % Slope of the transmitted chirp (Hz/s)
    h = 790000;                 % Platform altitude above ground (m)
    fdop = -1.349748e+02;       % Doppler frequency (Hz)
    
    % Additional Parameters
    Re = 6378144 ;              % Earth radius (m)
    
    % Min distance
    ro = time2range(rangeGateDelay);  % Min distance (m)  
    
    % Effective velocity
    v = sqrt(xVelocity^2+yVelocity^2+zVelocity^2);
    veff = v*sqrt(Re/(Re+h));   % Effective velocity (m/sec)
    
    % Chirp frequency
    fc = wavelen2freq(lambda);  % Chirp frequency (Hz)     
    
    % Chirp bandwidth
    bw = slope*tau;             % Chirp bandwidth (Hz)
end

function rawData = ERSDataExtractor(datafile,fs,fc,prf,tau,v,ro,doppler)
c = physconst('LightSpeed');                    % Speed of light

% Values specific to data file
totlines = 28652;                                % Total number of lines
numLines = 2048;                                 % Number of lines
numBytes = 11644;                                % Number of bytes of data
numHdr = 412;                                    % Header size
nValid = (numBytes-numHdr)/2 - round(tau*fs);    % Valid range samples

% Antenna length specific to ERS
L = 10;

% Calculate valid azimuth points
range = ro + (0:nValid-1) * (c/(2*fs));             % Computes range perpendicular to azimuth direction 
rdc = range/sqrt(1-(c*doppler/(fc*(2*v))^2));       % Squinted range 
azBeamwidth = rdc * (c/(fc*L)) * 0.8;               % Use only 80%  
azTau = azBeamwidth / v;                            % Azimuth pulse length 
nPtsAz = ceil(azTau(end) * prf);                    % Use the far range value
validAzPts = numLines - nPtsAz ;                    % Valid azimuth points  

% Start extracting
fid = fopen(datafile,'r');
status = fseek(fid,numBytes,'bof');     % Skipping first line
numPatch = floor(totlines/validAzPts);  % Number of patches           


if(status==-1)
   rawData = NaN;
   return;
end
rawData=zeros(numPatch*validAzPts,nValid);
% Patch data extraction starts
for patchi = 1:numPatch      
    fseek(fid,11644,'cof');
    data = fread(fid,[numBytes,numLines],'uint8')'; % Reading raw data file
    
    % Interpret as complex values and remove mean
    data = complex(data(:,numHdr+1:2:end),data(:,numHdr+2:2:end));
    data = data - mean(data(:));
    
    rawData((1:validAzPts)+((patchi-1)*validAzPts),:) = data(1:validAzPts,1:nValid);
    fseek(fid,numBytes + numBytes*validAzPts*patchi,'bof');
end
fclose(fid);
end