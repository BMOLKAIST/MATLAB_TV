classdef BACKWARD_SOLVER_RYTOV < BACKWARD_SOLVER
    properties (SetAccess = protected, Hidden = true)
        utility;
    end
    methods
        function h=BACKWARD_SOLVER_RYTOV(params)
            init_params=struct('use_non_negativity',false,'non_negativity_iteration', 100);
            if nargin==1
                warning('off','all');
                init_params=update_struct(init_params, params);
                warning('on','all');
            end
            h@BACKWARD_SOLVER(init_params);
        end
        function [RI, ORytov]=solve(h,input_field,output_field)
            warning('off','all');
            h.utility=DERIVE_OPTICAL_TOOL(h.parameters);
            warning('on','all');
            
            % check fields and parameters
            assert(ndims(input_field) == 4, 'You need to provide the field with 4 dimenssion : dim1 x dim2 x polarisation x illuminationnumber')
            assert(size(input_field,1) == size(input_field,2), 'Please input a square field')
            assert(isequal(size(input_field),size(output_field)), 'Please input field and bg of same size')
            assert(h.parameters.resolution(1) == h.parameters.resolution(2), 'x/y input resolution must be isotropic')
            assert(h.parameters.size(1) == h.parameters.size(2), 'x/y output size must be isotropic')
            assert(h.parameters.size(1) == size(input_field,1) && h.parameters.size(2) == size(input_field,2), 'declare size in the parameter must be the same as the field size')
            
            [bg,sp]=vector2scalarfield(input_field,output_field);
            retPhase=angle(sp./bg);
            retPhase=gather(unwrapp2_gpu(gpuArray(single(retPhase))));
            for jj = 1:size(retPhase,3)
                imagesc(retPhase(:,:,jj)),axis image, drawnow
            end
            retAmplitude=abs(sp./bg);
            
            thetaSize=size(retPhase,3);
            %find angle
            f_dx=zeros(thetaSize,1);
            f_dy=zeros(thetaSize,1);
            Fbg=fftshift(fft2(bg(:,:,1)));
            [mj_0,mi_0]=find(Fbg==max(Fbg(:)));
            for ii=2:size(bg,3)
                Fbg=fftshift(fft2(bg(:,:,ii)));
                [mj,mi]=find(Fbg==max(Fbg(:)));
                f_dy(ii)=mj-mj_0;
                f_dx(ii)=mi-mi_0;
            end
            

            kx_res = h.utility.fourier_space.res{2};
            ky_res = h.utility.fourier_space.res{1};
            kz_res = h.utility.fourier_space.res{3};
            z_res = h.parameters.resolution(3);
            xsize = h.parameters.size(2);
            ysize = h.parameters.size(1);
            zsize = h.parameters.size(3);
            k0_x=kx_res.*f_dx;
            k0_y=ky_res.*f_dy;
            k0_z=real(sqrt((h.utility.k0_nm)^2-(k0_x).^2-(k0_y).^2));
            Emask = (h.utility.fourier_space.coorxy)<(2*h.parameters.NA/h.parameters.wavelength);
            ORytov=gpuArray(single(zeros(ysize,xsize,zsize,'single')));
            Count=gpuArray(single(zeros(ysize,xsize,zsize,'single')));
            
            for kk= 1 :thetaSize
                FRytov=squeeze(log(retAmplitude(:,:,kk))+1i*retPhase(:,:,kk));
                UsRytov=fftshift(fft2(ifftshift(FRytov))); % unit: (um^2)
                UsRytov=gpuArray(circshift(UsRytov,[round(k0_y(kk)/ky_res) round(k0_x(kk)/kx_res)]));
                UsRytov=UsRytov.*h.utility.NA_circle;
                size_check=zeros(h.parameters.size(1:2),'single');
                kz=h.utility.k3+size_check;
                kx=h.utility.fourier_space.coor{2}+size_check;
                ky=h.utility.fourier_space.coor{1}+size_check;
                Kx=kx-k0_x(kk);Ky=ky-k0_y(kk);Kz=kz-k0_z(kk);
                Uprime= (kz/1i).*UsRytov; % unit: (um^1) % kz is spatial frequency, so 2pi is multiplied for wave vector
                xind=find((kz>0).*h.utility.NA_circle...
                    .*(Kx>(kx_res*(-floor(xsize/2))))...
                    .*(Ky>(ky_res*(-floor(ysize/2))))...
                    .*(Kz>(kz_res*(-floor(zsize/2))))...
                    .*(Kx<(kx_res*(floor(xsize/2)-1)))...
                    .*(Ky<(ky_res*(floor(ysize/2)-1)))...
                    .*(Kz<(kz_res*(floor(zsize/2)-1))));
                Uprime=Uprime(xind); Kx=Kx(xind); Ky=Ky(xind); Kz=Kz(xind);
                Kx=round(Kx/kx_res+floor(xsize/2)+1); 
                Ky=round(Ky/ky_res+floor(ysize/2)+1); 
                Kz=round(Kz/kz_res+floor(zsize/2)+1);
                Kzp=sub2ind(size(Count),Ky,Kx,Kz);
                temp=ORytov(Kzp);
                ORytov(Kzp)=temp+Uprime;
                Count(Kzp)=Count(Kzp)+(Uprime~=0);
                %disp([num2str(kk),' / ',num2str(thetaSize)])
            end
            ORytov(Count>0)=ORytov(Count>0)./Count(Count>0)/kz_res; % should be (um^-2)*(px*py*pz), so (px*py*pz/um^3) should be multiplied.
            Reconimg=gather(fftshift(ifftn(ifftshift(ORytov))));
            Reconimg = potential2RI(Reconimg*4*pi,h.parameters.wavelength,h.parameters.RI_bg);
            clear Count
            if h.parameters.use_non_negativity
                Reconimg = gpuArray(fftshift(ifftn(ifftshift(ORytov))));
                for mm = 1 : h.parameters.non_negativity_iteration
                    Reconimg(real(Reconimg)<0)= 0 + 1i*imag(Reconimg(real(Reconimg)<0));
                    ORytov_new=fftshift(fftn(ifftshift(Reconimg)));
                    ORytov_new=Emask.*ORytov_new.*(abs(ORytov)==0)+ORytov;
                    Reconimg=fftshift(ifftn(ifftshift(ORytov_new)));
                    %disp([num2str(mm),' / ',num2str(h.parameters.non_negativity_iteration)])
                end
                Reconimg(real(Reconimg)<0)= 0 + 1i*imag(Reconimg(real(Reconimg)<0));
                Reconimg = potential2RI(Reconimg*4*pi,h.parameters.wavelength,h.parameters.RI_bg);
                ORytov = gather(ORytov);
                ORytov_new = gather(ORytov_new);
                Reconimg = gather(Reconimg);
            end
            RI=Reconimg;
        end
        function [field_trans_f] = refocus(h, field_trans, z) % z is [um]
            field_trans_f = fftshift(ifft2(ifftshift(fftshift(fft2(ifftshift(field_trans))) .* exp(z.*h.utility.refocusing_kernel))));
        end
    end
end