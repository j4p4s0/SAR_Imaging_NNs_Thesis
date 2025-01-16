
extraction_path = pwd; %Data folder dir
% data_name = "E2_84699_STD_L0_F299"; %Matlab Example
% data_file = extraction_path + "\" + data_name;

%% JAPS Code - Extraction of zip folders

%data_name = "E2_84661_STD_L0_F145";
%data_file = extraction_path + data_name + "\" + data_name;

% Speed of light
c = physconst('LightSpeed');

% Data folder dir
%data_dir_path = 'C:\Users\joaoa\Documents\[EU]Faculdade\Tese\ESA_ERS2\Extraidos'; 
data_dir_path = 'C:\Users\joaoa\Desktop\Tese'; 

%% Data extraction

data_name = "E2_84661_STD_L0_F138"; %Matlab Example
data_file = data_dir_path + "\" + data_name + "\" + data_name;

% Extract ERS system parameters
[fs,fc,prf,tau,bw,v,ro,fd] = ERSParameterExtractor(data_file + '.000.ldr');
    
% Extract raw data 
%rawData = ERSDataExtractor(data_file + '.000.raw', data_file + '.000.pi',fs,fc,prf,tau,v,ro,fd).';
rawData = ERSDataExtractor(data_file + '.000.raw', fs,fc,prf,tau,v,ro,fd).';

%rawData = fft(rawData.');


%rawData = rawData.';

% %% Range Migration
% % Create LFM waveform
% waveform = phased.LinearFMWaveform('SampleRate',fs,'PRF',prf,'PulseWidth',tau,'SweepBandwidth',bw,'SweepInterval','Symmetric');
% sqang = asind((c*fd)/(2*fc*v));        % Squint angle
% 
% % Range migration algorithm
% slcimg = rangeMigrationLFM(rawData,waveform,fc,v,ro,'SquintAngle',sqang);
% 
% % Display image
% figure(1)
% imagesc(log(abs(slcimg)))
% axis image
% colormap('gray')
% title('SLC Image - Range Migration')
% ylabel('Range bin')
% xlabel('Azimuth bin')
%
% 
% mlimg = multilookProcessing(abs(slcimg),4,20);
% 
% % Display Image
% figure(2)
% imagesc(log(abs(mlimg(1:end-500,:))))
% axis image
% colormap('gray')
% title('Multi-look Image')
% ylabel('Range bin')
% xlabel('Azimuth bin')
% 
% function image = multilookProcessing(slcimg,sx,sy)
% [nx,ny] = size(slcimg);
% nfx = floor(nx/sx);
% nfy = floor(ny/sy);
% image = (zeros(nfx,nfy));
% for i=1:nfx
%     for j = 1:nfy
%         fimg=0;
%         for ix = 1:sx
%             for jy = 1:sy
%                 fimg = fimg+slcimg(((i-1)*sx)+ix,((j-1)*sy)+jy);
%             end
%         end
%         image(i,j) = fimg/(sx*sy);
%     end
% end
% end


%% Range-Doppler Algorithm


%rawData_fft = ifft(rawData);

%rawData = rawData.';

img = RD_SAR_focus(rawData, v, fc, prf, fs, 843, bw, tau);

% Display image
figure(2)
imagesc(log(abs(img)))
axis image
colormap('gray')
title('SLC Image - RD')
ylabel('Range bin')
xlabel('Azimuth bin')

% % File List extraction to unzip all foledrs
% fid = fopen('C:\Users\joaoa\Documents\[EU]Faculdade\Tese\ESA_ERS2\Folders_List.txt');
% 
% text_line = fgetl(fid);
% 
% while ischar(text_line)
% 
%     data_name = text_line;
%     data_file = data_dir_path + "\" + data_name + "\" + data_name;
% 
%     disp(data_name);
%     disp(data_file);
% 
%     % ExtractERSData(extraction_path, data_name);
% 
%     % Extract ERS system parameters
%     [fs,fc,prf,tau,bw,v,ro,fd] = ERSParameterExtractor(data_file + '.000.ldr');
% 
%     % Extract raw data 
%     rawData = ERSDataExtractor(data_file + '.000.raw', data_file + '.000.pi',fs,fc,prf,tau,v,ro,fd).';
% 
%     text_line = fgetl(fid);
% 
% end


%% MATLAB CODE - Data and Parameters Extraction
% dataURL = ['https://ssd.mathworks.com/supportfiles/radar/data/' ...
%    'ERSData.zip'];
% helperDownloadERSData(extraction_path,dataURL);
% 
% % Speed of light
% c = physconst('LightSpeed');
% 
% % Extract ERS system parameters
% [fs,fc,prf,tau,bw,v,ro,fd] = ERSParameterExtractor(data_file + '.000.ldr');
% 
% % Extract raw data 
% rawData145 = ERSDataExtractor(data_file + '.000.raw',fs,fc,prf,tau,v,ro,fd).';

% function helperDownloadERSData(outputFolder,DataURL)
% % Download the data set from the given URL to the output folder.
% 
%     radarDataZipFile = fullfile(outputFolder,'ERSData.zip');
% 
%     if ~exist(radarDataZipFile,'file')
% 
%         disp('Downloading ERS data (134 MiB)...');
%         websave(radarDataZipFile,DataURL);
%         unzip(radarDataZipFile,outputFolder);
%     end
% end
%% JAPS CODE - Functions
% function ExtractERSData(outputFolder, folder_name)
% % Unzip the data set to the output folder.
% 
%     radarDataZipFile = fullfile("C:\Users\joaoa\Documents\[EU]Faculdade\Tese\ESA_ERS2",  folder_name+".zip");
% 
%     if ~exist(radarDataZipFile,'dir')
% 
%         disp('Extracting ERS data...');
%         unzip(radarDataZipFile, outputFolder);
%     end
% end

%% MATLAB CODE - Functions
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

    % Slant range reference
    status = fseek(fid,720+1886+650+630,'bof')|status;
    SR_ref = str2double(fread(fid,[1 16],'*char'));    % Slant range reference (km)
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

% function rawData = ERSDataExtractor(datafile,pi_file, fs,fc,prf,tau,v,ro,doppler)
% c = physconst('LightSpeed');                    % Speed of light
% 
% searchString = "number_lines:";
% 
% fid = fopen(pi_file, 'r');
% 
% lineNum = -1;
% colNum = -1;
% 
% % Read the file line by line
% lineCount = 0;
% while ~feof(fid)
%     lineCount = lineCount + 1;
%     line = fgetl(fid); % Read a line from the file
% 
%     % Search for the string in the current line
%     colNum = strfind(line, searchString);
% 
%     totlines = str2double(line(colNum+strlength(searchString):strlength(line)));
% 
%     if ~isempty(colNum)
%         break;
%     end
% end
% 
% % Close the file
% fclose(fid);
% 
% 
% % Values specific to data file
% %totlines = 28652;                                % Total number of lines
% numLines = 2048;                                 % Number of lines
% numBytes = 11644;                                % Number of bytes of data
% numHdr = 412;                                    % Header size
% nValid = (numBytes-numHdr)/2 - round(tau*fs);    % Valid range samples
% 
% % Antenna length specific to ERS
% L = 10;
% 
% % Calculate valid azimuth points
% range = ro + (0:nValid-1) * (c/(2*fs));             % Computes range perpendicular to azimuth direction 
% rdc = range/sqrt(1-(c*doppler/(fc*(2*v))^2));       % Squinted range 
% azBeamwidth = rdc * (c/(fc*L)) * 0.8;               % Use only 80%  
% azTau = azBeamwidth / v;                            % Azimuth pulse length 
% nPtsAz = ceil(azTau(end) * prf);                    % Use the far range value
% validAzPts = numLines - nPtsAz ;                    % Valid azimuth points  
% 
% % Start extracting
% fid = fopen(datafile,'r');
% status = fseek(fid,numBytes,'bof');     % Skipping first line
% numPatch = floor(totlines/validAzPts);  % Number of patches           
% 
% 
% if(status==-1)
%    rawData = NaN;
%    return;
% end
% rawData=zeros(numPatch*validAzPts,nValid);
% % Patch data extraction starts
% for patchi = 1:numPatch      
%     fseek(fid,11644,'cof');
%     data = fread(fid,[numBytes,numLines],'uint8')'; % Reading raw data file
% 
%     % Interpret as complex values and remove mean
%     data = complex(data(:,numHdr+1:2:end),data(:,numHdr+2:2:end));
%     data = data - mean(data(:));
% 
%     rawData((1:validAzPts)+((patchi-1)*validAzPts),:) = data(1:validAzPts,1:nValid);
%     fseek(fid,numBytes + numBytes*validAzPts*patchi,'bof');
% end
% fclose(fid);
% end

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