%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    
%           Function created by Bo-Jui Chang (bjo4), 2021/12/6 @ Dallas
%           Modified to support BigTIFF and dynamic data types
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [FinalImage, mImage, nImage, NumberImages] = readtiffstack(path)

    % Validate input path before trying to read metadata
    if ~isfile(path)
        error('TIFF file does not exist: %s', path);
    end

    InfoImage = imfinfo(path);
    NumberImages = numel(InfoImage);

    if NumberImages == 0
        error('No TIFF pages found: %s', path);
    end

    % Image size is taken from the first page
    mImage = InfoImage(1).Height;
    nImage = InfoImage(1).Width;

    % Determine the MATLAB class from TIFF metadata
    bitsPerSample = InfoImage(1).BitsPerSample;

    if isfield(InfoImage(1), 'SampleFormat')
        sampleFormat = InfoImage(1).SampleFormat; % 1=uint, 2=int, 3=float
    else
        sampleFormat = 1; % Default to unsigned integer
    end

    if sampleFormat == 3
        if bitsPerSample == 32
            dataType = 'single';
        else
            dataType = 'double';
        end
    elseif sampleFormat == 2
        if bitsPerSample == 8
            dataType = 'int8';
        elseif bitsPerSample == 16
            dataType = 'int16';
        elseif bitsPerSample == 32
            dataType = 'int32';
        else
            dataType = 'int64';
        end
    else
        if bitsPerSample == 8
            dataType = 'uint8';
        elseif bitsPerSample == 16
            dataType = 'uint16';
        elseif bitsPerSample == 32
            dataType = 'uint32';
        else
            dataType = 'uint64';
        end
    end

    imageJDepth = parseImageJStackDepth(InfoImage(1));
    if NumberImages == 1 && imageJDepth > 1
        disp(sprintf('Detected ImageJ single-IFD stack with %d slices: %s', imageJDepth, path));
        FinalImage = readImageJSingleIFDStack(path, InfoImage(1), imageJDepth, dataType);
        NumberImages = imageJDepth;
        return;
    end

    % Preallocate output stack using the TIFFs native numeric type
    FinalImage = zeros(mImage, nImage, NumberImages, dataType);

    % Suppress noisy TIFF warnings, but restore warning state even on error
    warnState = warning('off', 'all');
    c = onCleanup(@() warning(warnState));

    TifLink = Tiff(path, 'r');
    tCleanup = onCleanup(@() TifLink.close());

    for i = 1:NumberImages
        TifLink.setDirectory(i);
        page = TifLink.read();

        % Fail fast if one page does not match the expected stack size
        if ~isequal(size(page,1), mImage) || ~isequal(size(page,2), nImage)
            error('Page %d size mismatch in %s', i, path);
        end

        FinalImage(:,:,i) = page;
    end
end

function depth = parseImageJStackDepth(info)
    depth = 1;
    if ~isfield(info, 'ImageDescription') || isempty(info.ImageDescription)
        return;
    end

    description = info.ImageDescription;
    imagesMatch = regexp(description, '(?m)^images=(\d+)\s*$', 'tokens', 'once');
    slicesMatch = regexp(description, '(?m)^slices=(\d+)\s*$', 'tokens', 'once');

    if ~isempty(imagesMatch)
        depth = str2double(imagesMatch{1});
    elseif ~isempty(slicesMatch)
        depth = str2double(slicesMatch{1});
    end

    if ~isfinite(depth) || depth < 1
        depth = 1;
    else
        depth = floor(depth);
    end
end

function stack = readImageJSingleIFDStack(path, info, depth, dataType)
    if isfield(info, 'Compression') && ~strcmpi(char(info.Compression), 'Uncompressed')
        error('Single-IFD ImageJ stack reading requires uncompressed TIFF data: %s', path);
    end
    if isfield(info, 'SamplesPerPixel') && info.SamplesPerPixel ~= 1
        error('Single-IFD ImageJ stack reading supports one sample per pixel: %s', path);
    end

    bytesPerSample = info.BitsPerSample / 8;
    expectedPixels = double(info.Height) * double(info.Width) * double(depth);
    expectedBytes = expectedPixels * bytesPerSample;
    stripOffset = firstStripOffset(path);

    fileInfo = dir(path);
    availableBytes = double(fileInfo.bytes) - double(stripOffset);
    if availableBytes < expectedBytes
        error('ImageJ stack metadata expects %.0f bytes but only %.0f bytes are available after pixel offset in %s', ...
              expectedBytes, availableBytes, path);
    end

    machineFormat = tiffMachineFormat(path);
    fid = fopen(path, 'r', machineFormat);
    if fid < 0
        error('Could not open TIFF file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));

    status = fseek(fid, double(stripOffset), 'bof');
    if status ~= 0
        error('Could not seek to TIFF pixel offset %.0f in %s', double(stripOffset), path);
    end

    raw = fread(fid, expectedPixels, ['*' dataType]);
    if numel(raw) ~= expectedPixels
        error('Expected %.0f pixels but read %.0f pixels from %s', expectedPixels, numel(raw), path);
    end

    stack = permute(reshape(raw, [info.Width, info.Height, depth]), [2 1 3]);
end

function offset = firstStripOffset(path)
    t = Tiff(path, 'r');
    cleanup = onCleanup(@() t.close());
    offsets = t.getTag('StripOffsets');
    offset = offsets(1);
end

function machineFormat = tiffMachineFormat(path)
    fid = fopen(path, 'r');
    if fid < 0
        error('Could not open TIFF file: %s', path);
    end
    cleanup = onCleanup(@() fclose(fid));
    byteOrder = fread(fid, 2, '*char')';
    if strcmp(byteOrder, 'II')
        machineFormat = 'ieee-le';
    elseif strcmp(byteOrder, 'MM')
        machineFormat = 'ieee-be';
    else
        error('Unrecognized TIFF byte order in %s', path);
    end
end
