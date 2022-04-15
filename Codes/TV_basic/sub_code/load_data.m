function [input_field,output_field]=load_data(bg_file,sp_file)
    if nargin==2
        output_field=load_data(sp_file);
    end
    [~,~,ext]=fileparts(bg_file);
    if strcmpi(ext,'.avi')
        input_field=read_AVI(bg_file);
    elseif strcmpi(ext, '.tiff')||strcmpi(ext, '.tif')
        input_field=loadTIFF(bg_file);
    elseif sum(strfind(bg_file, 'phantom')) ~= 0
        cd(bg_file)
        spdir = dir('*.tif');
        for j1 = 1:length(spdir)
            clc, disp([num2str(j1) ' / ' num2str(length(spdir))])
            input_field(:,:,j1)=loadTIFF(spdir(j1).name);
        end
    elseif sum(strfind(bg_file, '\set0')) ~= 0
        input_field=load_tiff_MS_setup(bg_file);
    elseif sum(strfind(bg_file, '\0')) ~=  0
        input_field=load_tomocube_PNG(bg_file);
    else % Bead droplet data is itself field. Will be converted in the main script.
        error('Filetype is not compatible.')
    end
end

function Data=load_tiff_MS_setup(dirname, num_field)
    fileformat=fullfile(dirname,'*.tiff');
    spdir=dir(fileformat);
    if nargin == 1
        num_field = length(spdir);
    end
    for i1=1:num_field
        img=imread(fullfile(spdir(i1).folder,spdir(i1).name));
        if i1==1
            Data=zeros([size(img) num_field]);
        end
        Data(:,:,i1)=img;
    end
end

function Data=load_tomocube_PNG(dirname)
    fileformat=fullfile(dirname,'*.png');
    spdir=dir(fileformat);
    for i1=1:length(spdir)
        img=imread(fullfile(spdir(i1).folder,spdir(i1).name));
        if i1==1
            Data=zeros([size(img) length(spdir)]);
        end
        Data(:,:,i1)=img;
    end
end

function object=loadTIFF(fname,num_images)
    info = imfinfo(fname);
    if nargin == 1
        num_images = numel(info);
    end
    display(['Number of images (read) : ',num2str(num_images)]);
    object= zeros(info(1).Width,info(1).Height,num_images,'single');
    for k = 1:num_images
       object(:,:,k) = imread(fname, k, 'Info', info);
    end
    if num_images==1
       object=squeeze(object);
    end
end