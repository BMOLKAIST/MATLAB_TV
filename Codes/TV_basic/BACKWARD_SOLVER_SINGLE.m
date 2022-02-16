classdef BACKWARD_SOLVER_SINGLE < BACKWARD_SOLVER
    methods
        function h=BACKWARD_SOLVER_SINGLE(params)
            init_params=struct( ...
                'init_solver',BACKWARD_SOLVER(),...
                'step',0.01,...%0.01;0.01;%0.01;
                'tv_param',0.001,...%0.1;
                'use_non_negativity',false,...
                'nmin',1.336,...
                'nmax',1.6,...
                'kappamax',0,... % imaginary RI
                'inner_itt',100,... % imaginary RI
                'itter_max',100,... % imaginary RI
                'num_scan_per_iteration',0,... % 0 -> every scan is used
                'verbose',true ...
            );
            if nargin==1
                warning('off','all');
                init_params=update_struct(init_params, params);
                warning('on','all');
            end
            %do not set the init solver it is for porent class compatibility
            h@BACKWARD_SOLVER(init_params);
        end
        
        function [RI]=solve(h,input_field,output_field, RI, mask)
            if nargin == 3
                [RI, mask]=(h.parameters.init_solver.solve(input_field,output_field));
            end
            RI = single(RI);
            mask = ifftshift(ifftshift(ifftshift(mask~=0)));
           
            err_list=[];
            
            dirichlet_boundary=false;
            use_gpu=h.parameters.use_GPU;

            alpha=1/h.parameters.step;
            s_n=0;
            
            t_n=0;
            t_np=1;
            
            u_n=(RI.^2 / h.parameters.RI_bg^2 - 1);
            x_n=(RI.^2 / h.parameters.RI_bg^2 - 1);
            c_n=0;
            c_np=Inf;
            
%             if mod(size(RI),2)==0
%                 error('need to be odd size');
%             end
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
            
            Vmin = (h.parameters.nmin^2 - h.parameters.kappamax^2) / h.parameters.RI_bg^2 - 1;
            Vmax = h.parameters.nmax^2 / h.parameters.RI_bg^2 - 1;
            Vimag_max = 2 * h.parameters.kappamax *h.parameters.nmax / h.parameters.RI_bg^2;
            
            ORytov = fftn(RI.^2 / h.parameters.RI_bg^2 - 1) .* mask;
            
%             figure,orthosliceViewer(abs(gather(mask))), error
            
            for ii=1:h.parameters.itter_max
                
                display(['itter : ' num2str(ii)]);
                
                tic;
                t_n=t_np;
                c_n=c_np;
                
                gradient_RI = ifftn(fftn(RI.^2 / h.parameters.RI_bg^2 - 1).*mask - ORytov);
                
%                 size(u_n)
%                 size(gradient_RI)
                s_n=TV_FISTA_inner_v2(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,...
                    Vmin, Vmax, Vimag_max, h.parameters.use_non_negativity,dirichlet_boundary,h.parameters.inner_itt,use_gpu);
%                 s_n=TV_FISTA_inner(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,h.parameters.use_non_negativity,dirichlet_boundary,h.parameters.inner_itt,use_gpu);
                t_np=(1+sqrt(1+4*t_n^2))/2;
                u_n=s_n+(t_n-1)/t_np*(s_n-x_n);
                x_n=s_n;
                RI=u_n;
                RI=h.parameters.RI_bg *sqrt(RI+1);
                
                toc;
                
                if h.parameters.verbose
                    set(0, 'currentfigure', f1);
                    imagesc(real(max(RI,[],3)));colorbar; axis image;title('RI (xz)')
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
            
            RI=gather(RI);
            
            
            warning('add kz ? also because of the abbe sine is it sqrt of kz ??')
            warning('implement cuda inner itteration tv');
            warning('implement mfista');
            
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
        
    end
end