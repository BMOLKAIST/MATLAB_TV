classdef FL_SOLVER < handle
    properties (SetAccess = private, Hidden = true)
        parameters;
        utility;
    end
    methods(Static)
        function params=get_default_parameters(init_params)
            %OPTICAL PARAMETERS
            params=BASIC_OPTICAL_PARAMETER();
            params.use_GPU = true;
            if nargin==1
                params=update_struct(params,init_params);
            end
        end
    end
    methods
        function h=FL_SOLVER(params)
            h.parameters=params;
        end
        function [databg,datasp,ROI]=get_fields(h,bg_file,sp_file, typestring, ROI)
            databg = load_tiff_MS_setup(bg_file, typestring,h.parameters.RGB_on);
            datasp = load_tiff_MS_setup(sp_file, typestring,h.parameters.RGB_on);    
            if sum(strfind(typestring,'R')) > 0
                h.parameters.lambda = 0.66;
            elseif sum(strfind(typestring,'G')) > 0
                h.parameters.lambda = 0.61;
            elseif sum(strfind(typestring,'B')) > 0
                h.parameters.lambda = 0.52;
            end
        end
        
        function datasp = decompose(h,databg,datasp)
            U = exp(1i.*(((0:4)*2*pi/5)'*(0:4))); 
            inv_U = inv(U);

            datasp2 = datasp - mean(mean(mean(databg,3),4),5);
            datasp = datasp(:,:,:,1:7) * 0;
            for j2 = 1:3
                SIM_SEP = datasp2(:,:,:,1:5);
                datasp2 = datasp2(:,:,:,6:end);
                SIM_SEP2 = SIM_SEP * 0;
                for n1 = 1:3
                    SIM_SEP2(:,:,:,n1) = sum((SIM_SEP).*reshape(inv_U(:,n1),[1 1 1 5]),4);
                    if n1 == 1
                        a = real(SIM_SEP2(:,:,:,n1));
                        a(a<0)=0;
                        SIM_SEP2(:,:,:,n1) = a;
                    end
                end
                datasp(:,:,:,[1, 2*j2, 2*j2+1]) = datasp(:,:,:,[1, 2*j2, 2*j2+1]) + SIM_SEP2(:,:,:,1:3);
            end % orthosliceViewer(log10(abs(fftshift(fftn(ifftshift(datasp(:,:,:,3)))))))
            datasp(:,:,:,1) = datasp(:,:,:,1) / 3;
        end
        
        function datasp = solve_WF(h,datasp)
        % 1. Make theoretical OTF
            ZPZ = size(datasp,3);
            ZPX = size(datasp,1);
            OTF_3D = zeros(ZPX,ZPX,ZPZ,'single');
            if h.parameters.use_GPU
                datasp = gpuArray(datasp);
                OTF_3D = gpuArray(OTF_3D);
            end
            
            kresX = 1 / (ZPX * h.parameters.resolution_image(1));
            kresZ = 1 / (ZPZ * h.parameters.resolution_image(3));
            
            xr=round(h.parameters.NA/h.parameters.lambda/kresX); % radius of NA in the padded k space [pixel]
            [kx, ky]=ndgrid(kresX*(-floor(size(OTF_3D,1)/2):size(OTF_3D,1)-floor(size(OTF_3D,1)/2)-1),kresX*(-floor(size(OTF_3D,1)/2):size(OTF_3D,1)-floor(size(OTF_3D,1)/2)-1));
            k0 = h.parameters.RI_bg / h.parameters.lambda;

            kz=real(sqrt((k0)^2-kx.^2-ky.^2)); % Generating coordinates on the surface of Ewald Sphere
            Kx=kx;Ky=ky;Kz=kz-k0;
            
            xind=find(~mk_ellipse_MS(xr,xr,ZPX,ZPX)...
            .*(Kx>(kresX*(-floor(ZPX/2))))...
            .*(Ky>(kresX*(-floor(ZPX/2))))...
            .*(Kz>(kresZ*(-floor(ZPZ/2))))...
            .*(Kx<(kresX*(floor(ZPX/2)-1)))...
            .*(Ky<(kresX*(floor(ZPX/2)-1)))...
            .*(Kz<(kresZ*(floor(ZPZ/2)-1))));
            Kx=Kx(xind); Ky=Ky(xind); Kz=Kz(xind); kz = Kz + k0;
            Kx=round(Kx/kresX+floor(ZPX/2)+1);
            Ky=round(Ky/kresX+floor(ZPX/2)+1);
            Kz=round(Kz/kresZ+floor(ZPZ/2)+1);
            Kzp=(Kz-1)*ZPX^2+(Kx-1)*ZPX+Ky;

            OTF_3D(Kzp)=1; 
            OTF_2D = ~mk_ellipse_MS(xr,xr,ZPX,ZPX); sum_2d =  sum(OTF_2D(:));
            OTF_3D = real(fftshift(fftn(ifftshift( abs(fftshift(ifftn(ifftshift(OTF_3D)))).^2)))); 
            OTF_3D = OTF_3D ./ max(OTF_3D(:)) * length(Kzp);
            OTF_2D = real(fftshift(fftn(ifftshift( abs(fftshift(ifftn(ifftshift(OTF_2D)))).^2)))); 
            OTF_2D = OTF_2D ./ max(OTF_2D(:)) * sum_2d;
            OTF_mask_3D = OTF_3D > 0.999;  OTF_2D = repmat(OTF_2D,[1 1 ZPZ]);
            OTF_mask_2D = OTF_2D > 0.999; OTF_2D = OTF_2D .* OTF_mask_2D; 
            OTF_3D = OTF_3D .* OTF_mask_3D;
            Apodize = ~mk_ellipse_MS(2*xr,2*xr,2*xr*ZPZ/ZPX,ZPX,ZPX,ZPZ);
            Apodize = real(fftshift(fftn(ifftshift( abs(fftshift(ifftn(ifftshift(Apodize)))).^2)))); 
            Apodize = Apodize ./ max(Apodize(:));
            OTF_3D = OTF_3D .* Apodize;
            if h.parameters.use_GPU
                OTF_mask_3D = gpuArray(OTF_mask_3D);  OTF_2D = gpuArray(OTF_2D); Apodize = gpuArray(Apodize); OTF_3D = gpuArray(OTF_3D);
            end
            PSF = real(fftshift(ifftn(ifftshift(OTF_3D))));
            PSF = PSF ./ max(PSF(:));
            
        % Richardson Lucy
            max_datasp = max(datasp(:));
            datasp = RichardsonLucy_MS(datasp,PSF,h.parameters.RL_iterations,h.parameters.use_GPU); % orthosliceViewer(log10(abs(fftshift(fftn(ifftshift(I_WF_deconv))))))
            datasp = gather(datasp ./max(datasp(:)) * max_datasp);
            datasp(datasp<0)=0;
        end

        function [mask, OTF1] = mask_theoretical_OTF(h, datasp, NA_obj, NA_cond)
        % 1. Make theoretical OTF
            ZPZ = size(datasp,3);
            ZPX = size(datasp,1);
            OTF1 = zeros(ZPX,ZPX,ZPZ,'single');
            if h.parameters.use_GPU
                datasp = gpuArray(datasp);
                OTF1 = gpuArray(OTF1);
            end
            
            kresX = 1 / (ZPX * h.parameters.resolution(1));
            kresZ = 1 / (ZPZ * h.parameters.resolution(3));
            
            xr=round(NA_obj/h.parameters.wavelength/kresX); % radius of NA in the padded k space [pixel]
            [kx, ky]=ndgrid(kresX*(-floor(size(OTF1,1)/2):size(OTF1,1)-floor(size(OTF1,1)/2)-1),kresX*(-floor(size(OTF1,1)/2):size(OTF1,1)-floor(size(OTF1,1)/2)-1));
            k0 = h.parameters.RI_bg / h.parameters.wavelength;

            kz=real(sqrt((k0)^2-kx.^2-ky.^2)); % Generating coordinates on the surface of Ewald Sphere
            Kx=kx;Ky=ky;Kz=kz-k0;
            
            xind=find(~mk_ellipse_MS(xr,xr,ZPX,ZPX)...
            .*(Kx>(kresX*(-floor(ZPX/2))))...
            .*(Ky>(kresX*(-floor(ZPX/2))))...
            .*(Kz>(kresZ*(-floor(ZPZ/2))))...
            .*(Kx<(kresX*(floor(ZPX/2)-1)))...
            .*(Ky<(kresX*(floor(ZPX/2)-1)))...
            .*(Kz<(kresZ*(floor(ZPZ/2)-1))));
            Kx=Kx(xind); Ky=Ky(xind); Kz=Kz(xind); kz = Kz + k0;
            Kx=round(Kx/kresX+floor(ZPX/2)+1);
            Ky=round(Ky/kresX+floor(ZPX/2)+1);
            Kz=round(Kz/kresZ+floor(ZPZ/2)+1);
            Kzp=(Kz-1)*ZPX^2+(Kx-1)*ZPX+Ky;

            OTF1(Kzp)=1; 
            OTF2 = zeros(ZPX,ZPX,ZPZ,'single');
            if h.parameters.use_GPU
                datasp = gpuArray(datasp);
                OTF2 = gpuArray(OTF2);
            end
            
            kresX = 1 / (ZPX * h.parameters.resolution(1));
            kresZ = 1 / (ZPZ * h.parameters.resolution(3));
            
            xr=round(NA_cond/h.parameters.wavelength/kresX); % radius of NA in the padded k space [pixel]
            [kx, ky]=ndgrid(kresX*(-floor(size(OTF2,1)/2):size(OTF2,1)-floor(size(OTF2,1)/2)-1),kresX*(-floor(size(OTF2,1)/2):size(OTF2,1)-floor(size(OTF2,1)/2)-1));
            k0 = h.parameters.RI_bg / h.parameters.wavelength;

            kz=real(sqrt((k0)^2-kx.^2-ky.^2)); % Generating coordinates on the surface of Ewald Sphere
            Kx=kx;Ky=ky;Kz=kz-k0;
            
            xind=find(~mk_ellipse_MS(xr,xr,ZPX,ZPX)...
            .*(Kx>(kresX*(-floor(ZPX/2))))...
            .*(Ky>(kresX*(-floor(ZPX/2))))...
            .*(Kz>(kresZ*(-floor(ZPZ/2))))...
            .*(Kx<(kresX*(floor(ZPX/2)-1)))...
            .*(Ky<(kresX*(floor(ZPX/2)-1)))...
            .*(Kz<(kresZ*(floor(ZPZ/2)-1))));
            Kx=Kx(xind); Ky=Ky(xind); Kz=Kz(xind); kz = Kz + k0;
            Kx=round(Kx/kresX+floor(ZPX/2)+1);
            Ky=round(Ky/kresX+floor(ZPX/2)+1);
            Kz=round(Kz/kresZ+floor(ZPZ/2)+1);
            Kzp=(Kz-1)*ZPX^2+(Kx-1)*ZPX+Ky;

            OTF2(Kzp)=1; 
            
            OTF1 = real(fftshift(fftn(ifftshift( (fftshift(ifftn(ifftshift(OTF1)))).*conj((fftshift(ifftn(ifftshift(OTF2))))))))); 
            OTF1 = OTF1 ./ max(OTF1(:)) * length(Kzp);
            mask = OTF1 > 0.999; 
            OTF1 = gather(single(OTF1 .* mask));
            mask = gather(mask);

        end
        function datasp = solve_WF_TV(h,datasp)
            datasp = solve_WF(h,datasp);
        % 1. Make theoretical OTF
        
            Imax = max(datasp(:));
            datasp = datasp ./ Imax*0.3;
            ZPZ = size(datasp,3);
            ZPX = size(datasp,1);
            mask = zeros(ZPX,ZPX,ZPZ,'single');
            if h.parameters.use_GPU
                datasp = gpuArray(datasp);
                mask = gpuArray(mask);
            end
            
            kresX = 1 / (ZPX * h.parameters.resolution(1));
            kresZ = 1 / (ZPZ * h.parameters.resolution(3));
            
            xr=round(h.parameters.NA/h.parameters.lambda/kresX); % radius of NA in the padded k space [pixel]
            [kx, ky]=ndgrid(kresX*(-floor(size(mask,1)/2):size(mask,1)-floor(size(mask,1)/2)-1),kresX*(-floor(size(mask,1)/2):size(mask,1)-floor(size(mask,1)/2)-1));
            k0 = h.parameters.RI_bg / h.parameters.lambda;

            kz=real(sqrt((k0)^2-kx.^2-ky.^2)); % Generating coordinates on the surface of Ewald Sphere
            Kx=kx;Ky=ky;Kz=kz-k0;
            
            xind=find(~mk_ellipse_MS(xr,xr,ZPX,ZPX)...
            .*(Kx>(kresX*(-floor(ZPX/2))))...
            .*(Ky>(kresX*(-floor(ZPX/2))))...
            .*(Kz>(kresZ*(-floor(ZPZ/2))))...
            .*(Kx<(kresX*(floor(ZPX/2)-1)))...
            .*(Ky<(kresX*(floor(ZPX/2)-1)))...
            .*(Kz<(kresZ*(floor(ZPZ/2)-1))));
            Kx=Kx(xind); Ky=Ky(xind); Kz=Kz(xind); kz = Kz + k0;
            Kx=round(Kx/kresX+floor(ZPX/2)+1);
            Ky=round(Ky/kresX+floor(ZPX/2)+1);
            Kz=round(Kz/kresZ+floor(ZPZ/2)+1);
            Kzp=(Kz-1)*ZPX^2+(Kx-1)*ZPX+Ky;

            
            mask(Kzp)=1; 
            mask = real(fftshift(fftn(ifftshift( abs(fftshift(ifftn(ifftshift(mask)))).^2)))); 
%             mask = mask ./ max(mask(:));
            mask = mask ./ max(mask(:)) * length(Kzp);
            mask = mask.*(mask > 0.999); 
%             mask = (mask > 0.999);
%             figure,orthosliceViewer(abs(gather(mask))), error
            mask = ifftshift(ifftshift(ifftshift(mask~=0)));
            
        % TV
            ORytov = fftn(datasp) .* mask;

            dirichlet_boundary=false;
            use_gpu=h.parameters.use_GPU;

            alpha=1/h.parameters.step;
            s_n=0;
            t_n=0;
            t_np=1;
            
            u_n=datasp;
            x_n=datasp;
            c_n=0;
            c_np=Inf;
            err_list=[];


            if h.parameters.verbose
                close all
                f1=figure(1);
                f2=figure(2);
                f3=figure(3);
                %{
                f4=figure(4);
                f5=figure(5);
                f6=figure(6);
                f7=figure(7);
                %}
            end
            for ii=1:h.parameters.itter_max
                
                display(['itter : ' num2str(ii)]);
                
                tic;
                t_n=t_np;
                c_n=c_np;
                
                gradient_RI = real(ifftn(fftn(datasp).*mask - ORytov));
                
%                 size(u_n)
%                 size(gradient_RI)
                s_n=TV_FISTA_inner_v2(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,...
                    0, inf, 0, h.parameters.use_non_negativity,dirichlet_boundary,h.parameters.inner_itt,use_gpu);
%                 s_n=TV_FISTA_inner(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,h.parameters.use_non_negativity,dirichlet_boundary,h.parameters.inner_itt,use_gpu);
                t_np=(1+sqrt(1+4*t_n^2))/2;
                u_n=s_n+(t_n-1)/t_np*(s_n-x_n);
                x_n=s_n;
                datasp=u_n;
                toc;
                
                if h.parameters.verbose
                    set(0, 'currentfigure', f1);
                    imagesc(real(squeeze(datasp(round(size(datasp,1)/2),:,:,1,1))));colorbar; axis image;title('RI (xz)')
                    set(0, 'currentfigure', f2);
                    plot(err_list);title('Cost function')
                    set(0, 'currentfigure', f3);
                    semilogy((err_list));title('Cost function (log)')
                    %{
                    set(0, 'currentfigure', f4);
                    imagesc([abs(squeeze(trans_source(:,:,1,[1]))) squeeze(abs(output_field(:,:,1,[1]))) squeeze(abs(trans_source(:,:,1,1)-output_field(:,:,1,1)))]); axis image;title('Abs (predicted / experimental / delta)'),colorbar
                    set(0, 'currentfigure', f5);
                    imagesc([abs(squeeze(trans_source(:,:,1,[end]))) squeeze(abs(output_field(:,:,1,[scan_list(end)]))) squeeze(abs(trans_source(:,:,1,[end])-output_field(:,:,1,[scan_list(end)])))]); axis image;title('Abs (predicted / experimental / delta)'),colorbar
                    set(0, 'currentfigure', f6);
                    imagesc([squeeze(angle(trans_source(:,:,1,[1]))) squeeze(angle(output_field(:,:,1,[scan_list(1)]))) angle(trans_source(:,:,1,1)./output_field(:,:,1,1))]);axis image;title('Angle (predicted / experimental)'),colorbar
                    set(0, 'currentfigure', f7);
                    imagesc([angle(trans_source(:,:,1,end)) angle(output_field(:,:,1,scan_list(end))) angle(trans_source(:,:,1,end)./output_field(:,:,1,scan_list(end)))]);axis image;title('Angle (predicted / experimental)'),colorbar
                    %}
                    drawnow;
                end
            end
            datasp=gather(datasp*Imax/0.3);
        end
        
        
        function FL_wf_deconv = match_size(h,FL_wf_deconv)
            % Lateral crop
            resize_factor = round(size(FL_wf_deconv) .* h.parameters.resolution_image ./ h.parameters.resolution);
            FL_wf_deconv = imresize3(FL_wf_deconv, resize_factor);
%             h.parameters.resolution_image = h.parameters.resolution;
        end
        
        function [RI_bead, FL_bead,new_params] = register_beads(h,RI_bead, FL_bead)
            RI_bead = max(RI_bead,[],3);
            RI_bead = RI_bead - min(RI_bead(:));
            RI_bead = RI_bead ./ max(RI_bead(:));
            
            FL_bead = FL_bead - min(FL_bead(:));
            FL_bead = FL_bead ./ max(FL_bead(:));
            tformEstimate = imregcorr(FL_bead,RI_bead);
            Rfixed = imref2d(size(RI_bead));
            h.parameters.tformEstimate = tformEstimate;
            h.parameters.Rfixed = Rfixed;
            new_params = h.parameters;
        end
        
    end
end

function Data=load_tiff_MS_setup(namee, typestring,RGB_on)
cd(namee)
spdir=dir([typestring '*.tiff']);
Data.R = []; Data.G = []; Data.B = [];
if sum(strfind(typestring,'WF')) >0
    num_Z = length(spdir) / sum(RGB_on);
    for j1 = 1:length(spdir)
        if j1 == 1
            Data0 = loadTIFF(spdir(j1).name);
            Data0 = zeros([size(Data0) num_Z length(spdir)/num_Z],'single');
        end
        zid = floor((j1-1) / size(Data0, 4)) + 1;
        simid = rem(j1-1, size(Data0,4)) + 1;
        Data0(:,:,zid, simid) = loadTIFF(spdir(j1).name);
    end
    if RGB_on(1)
        Data.R = Data0(:,:,:,1); Data0(:,:,:,1) =[];
    end
    if RGB_on(2)
        Data.G = Data0(:,:,:,1); Data0(:,:,:,1) =[];
    end
    if RGB_on(3)
        Data.B = Data0(:,:,:,1); Data0(:,:,:,1) =[];
    end
elseif sum(strfind(typestring,'SIM')) >0
    num_Z = length(spdir) / (3 * 5);
    for j1 = 1:length(spdir)
        if j1 == 1
            Data = loadTIFF(spdir(j1).name);
            Data = zeros([size(Data) num_Z length(spdir)/num_Z],'single');
        end
        zid = floor((j1-1) / size(Data, 4)) + 1;
        simid = rem(j1-1, size(Data,4)) + 1;
        Data(:,:,zid, simid) = loadTIFF(spdir(j1).name);
    end
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

function H = mk_ellipse_MS(varargin)
if length(varargin)==4
    XR=varargin{1};YR=varargin{2};
    X=varargin{3};Y=varargin{4};
    [XX, YY]=meshgrid(1:X,1:Y);
    H=((XX-floor(X/2)-1)./XR).^2+((YY-floor(Y/2)-1)./YR).^2>1.0;
elseif length(varargin)==6
    XR=varargin{1};YR=varargin{2};ZR=varargin{3};
    X=varargin{4};Y=varargin{5};Z=varargin{6};
    [XX, YY,ZZ]=meshgrid(1:X,1:Y,1:Z);
    H=((XX-floor(X/2)-1)./XR).^2+((YY-floor(Y/2)-1)./YR).^2+((ZZ-floor(Z/2)-1)./ZR).^2>1.0;
else
    error('즐')
end
return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end