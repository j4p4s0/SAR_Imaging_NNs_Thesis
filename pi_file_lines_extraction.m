% Data folder dir
data_dir_path = 'C:\Users\joaoa\Documents\[EU]Faculdade\Tese\ESA_ERS2\Extraidos'; 

data_name = "E2_84699_STD_L0_F299"; %Matlab Example
data_file = data_dir_path + "\" + data_name + "\" + data_name;

data_file = data_file + ".000.pi";

searchString = "number_lines:";

fileID = fopen(data_file, 'r');

if fileID == -1
    error('Error opening the file.');
end

% Initialize line and column numbers
lineNum = -1;
colNum = -1;

% Read the file line by line
lineCount = 0;
while ~feof(fileID)
    lineCount = lineCount + 1;
    line = fgetl(fileID); % Read a line from the file
    
    % Search for the string in the current line
    colNum = strfind(line, searchString);

    num_lines = str2double(line(colNum+strlength(searchString):strlength(line)));
    
    if ~isempty(colNum)
        lineNum = lineCount;
        colNum = colNum(1); % If the string appears multiple times, take the first occurrence
        break;
    end
end

% Close the file
fclose(fileID);

% If the string wasn't found, display a message
if lineNum == -1
    disp('String not found in the file.');
else
    fprintf('String found at line %d, column %d.\n', lineNum, colNum);
end
