%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Deskew the data by simply shifting the images laterally 
%           Currently working on the images that move up and down (not left
%           and right). If you want to process the data left and right,
%           please rotate the data first.
%           Note: 
%           1. You can choose if you want to save the deskew(shear) images.
%           2. BigTIFF format is automatically used for files larger than 4GB,
%           preserving full resolution for all output images.
%           3. Fixed some bugs about artificial strips in the topview. It
%           was discovered by Tadamoto Isogai
%           Bo-Jui, 2025/5/9 @ Dallas
%           Updated 2025/12/3: Added BigTIFF support to preserve full resolution
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Configuration Parameters


% =========================================================================
% INJECTED BY NEXTFLOW / PYTHON WRAPPER
% The following variables are passed directly into the workspace:
% imagePath, CellName, CellIndex, dx, dz, angle, flip, output_dir
% DO NOT hardcode them here or it will break the pipeline
% =========================================================================

if ~exist('output_dir', 'var') || isempty(output_dir)
    error('output_dir was not provided by the wrapper.');
end


tic;

%% Processing Setup
numFolders = 1;

for c = 1:numFolders

    % Backward-compatible layout: when CellName is provided, read
    % <imagePath>/<CellName>. Otherwise, read imagePath directly.
    if exist('CellName', 'var') && ~isempty(CellName)
        inputDir = fullfile(imagePath, CellName);
    else
        inputDir = imagePath;
    end

    % Discover input files in the cell folder. File selection is handled by
    % the workflow before this script runs, so do not require CH##_###### names.
    tifFiles = [dir(fullfile(inputDir, '*.tif')); dir(fullfile(inputDir, '*.tiff'))];
    if isempty(tifFiles)
        error('No TIFF files found in %s', inputDir);
    end

    % Sort files for deterministic processing.
    [~, idx] = sort({tifFiles.name});
    tifFiles = tifFiles(idx);

    detectedChannels = [];
    for i = 1:numel(tifFiles)
        tokens = regexp(tifFiles(i).name, '^CH(\d+)_(\d+)(?:_registered_consistent)?\.tiff?$', 'tokens', 'once');
        if ~isempty(tokens)
            detectedChannels(end + 1) = str2double(tokens{1}); %#ok<SAGROW>
        end
    end

    if ~isempty(detectedChannels)
        disp(sprintf('Detected channels from filename metadata: %s', mat2str(unique(detectedChannels))));
    else
        disp('No CH##_###### filename metadata detected; processing selected TIFFs by filename.');
    end
    disp(sprintf('Processing %d TIFF file(s).', numel(tifFiles)));
    if numel(unique(detectedChannels)) > 1
        warning(['Multiple channels are selected for deskew. The downstream ', ...
                 'deconvolution step estimates one PSF from the first selected ', ...
                 'TIFF and applies it to all selected TIFFs. Process one channel ', ...
                 'at a time unless applying one PSF across wavelengths is intentional.']);
    end

    % Process each selected TIFF directly.
    for fileIdx = 1:numel(tifFiles)
            tic;

            filepath = fullfile(inputDir, tifFiles(fileIdx).name);
            [~, baseName, ~] = fileparts(tifFiles(fileIdx).name);
            disp(sprintf('Processing TIFF: %s', tifFiles(fileIdx).name));

            Image = readtiffstack(filepath);

            % Image dimensions
            [ysize, xsize, zsize] = size(Image);
            newdz = dz * cosd(angle);

            disp("Deskewing ... ");
            clear ShearImage
            cz = floor(zsize / 2) + 1;

            max_yoffset = abs(round(flip * (1 - cz) * (newdz / dx)));
            ShearImage = zeros(ysize + 2 * max_yoffset, xsize, zsize, 'uint16');


            tic
            for z = 1:zsize
                yoffset = round(flip * (z - cz) * (newdz / dx));
                ShearImage(yoffset + max_yoffset + 1 : ysize + yoffset + max_yoffset, :, z) = Image(:,:,z);
            end
            toc

            % Average adjacent slices to reduce artifact ringing in the top view.
            for z = 1:zsize-1
                ShearImage(:,:,z) = (ShearImage(:,:,z) + ShearImage(:,:,z+1)) / 2;
            end

            toc

            % Save the processed image stack
            disp("Saving the shear image");
            tic
            output_size = size(ShearImage,1) * size(ShearImage,2) * size(ShearImage,3) * 2 / (1024*1024*1024);

            outputFolder = fullfile(output_dir, 'shear');
            if ~isfolder(outputFolder)
                mkdir(outputFolder);
            end
            if output_size > 4
                disp(sprintf("File is larger than 4GB (%.2f GB), saving as BigTIFF format", output_size));
            end
            writetiffstack(ShearImage, fullfile(outputFolder, [baseName '.tif']));
            toc

            %% Rotation to top view
            disp("Rotating to top view ... ");
            clear mipzy scaled_ShearImage scaled_mipzy rot_scaled_mipzy mask cropped_mipzy zy_view rotTop_ShearImage;

            if zsize == 1
                warning('Single-slice TIFF detected; skipping 3D top-view rotation and writing the selected image as-is.');
                rotTop_ShearImage = ShearImage;
            else
                % Compute MIP on the 3rd dimension (correct for 2D display)
                mipzy = max(ShearImage, [], 3);

                figure(1)
                imagesc(mipzy); axis equal tight

                scale_x = dz * sind(angle) / dx;

                % Resize only in x-direction using single precision to save memory
                scaled_ShearImage = imresize3(single(ShearImage), ...
                    [size(ShearImage,1), size(ShearImage,2), round(size(ShearImage,3) * scale_x)], ...
                    'Method', 'linear');

                scaled_mipzy = max(scaled_ShearImage, [], 3);

                figure(2)
                imagesc(scaled_mipzy); axis equal tight

                rot_scaled_mipzy = imrotate(scaled_mipzy, -1 * flip * angle, 'bilinear', 'crop');
                figure(2)
                imagesc(rot_scaled_mipzy); axis equal tight

                % Rotate entire 3D volume at once using imrotate3 (much faster than slice-by-slice)
                tic
                rotTop_ShearImage = imrotate3(scaled_ShearImage, -1 * flip * angle, [0 0 1], 'nearest', 'crop');
                rotTop_ShearImage = uint16(rotTop_ShearImage);
                toc

                rotTop_ShearImage = permute(rotTop_ShearImage, [1 3 2]);
            end

            output_size_rotTop = size(rotTop_ShearImage,1) * size(rotTop_ShearImage,2) * size(rotTop_ShearImage,3) * 2 / (1024*1024*1024);

            outputFolder2 = fullfile(output_dir, 'Top_shear');
            if ~isfolder(outputFolder2)
                mkdir(outputFolder2);
            end

            disp("Saving the top-view image");
            if output_size_rotTop > 4
                disp("File is larger than 4GB, saving as BigTIFF format");
            end

            tic
            writetiffstack(rotTop_ShearImage, fullfile(outputFolder2, [baseName '.tif']));

            % Write note file
            msg = 'z pixel = x(y) pixel. Full resolution preserved.';
            if output_size_rotTop > 4
                msg = sprintf('%s BigTIFF format used for %.2f GB file.', msg, output_size_rotTop);
            end

            note_filename = fullfile(outputFolder2, 'note.txt');
            fileID = fopen(note_filename,'w');
            fprintf(fileID,'%s', msg);
            fclose(fileID);
            toc;
            toc;
    end
end

toc
disp("All processing completed successfully.");
