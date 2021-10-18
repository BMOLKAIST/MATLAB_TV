
% code by Herve Hugonnet and Moosung lee
% from the paper : 'Fast Gradient-Based Algorithms for Constrained Total Variation Image Denoising and Deblurring Problems'
% from the paper : 'Hessian Schatten-Norm Regularization for Linear Inverse Problems'
% inspired by the matlab code from Amir Beck and Marc Teboulle (c.f. https://sites.google.com/site/amirbeck314/software)

classdef TV < handle
    properties %(SetAccess = private, Hidden = true)
        parameters;
    end
    methods(Static)
        function params=get_default_parameters()
            params=struct;
            %min and maximum
            params.min_real=-inf;
            params.max_real=inf;
            params.min_imag=0;
            params.max_imag=0;
            %boundary condition
            params.use_boundary_value=false;
            params.boundary_value=0;
            %Tv parameters
            params.TV_strength=0.01;
            %hyper parameter
            params.outer_itterations=100;
            params.inner_itterations=100;
            %execution parameters
            params.use_gpu=true;
            params.use_cuda=true;
            %reduce gpu memory usage 
            params.low_memory=false;
        end
    end
    methods
        %% Constructor - initialise data
        function delete(h)
            
        end
        function h = TV(parameters)
            h.parameters=parameters;
        end
        %% Constructor - reconstruct refractive index
        function out_mat = solve(h,in_mat,fft_weight)
            
            reg_name='tv';
            
            if ~h.parameters.use_gpu
                h.parameters.use_cuda=false;
                
                in_mat=single(in_mat);
                fft_weight=single(fft_weight);
            else
                in_mat=gpuArray(single(in_mat));
                fft_weight=gpuArray(single(fft_weight));
            end
            
            if h.parameters.min_imag==h.parameters.max_imag
                display('data constraint to real');
                if ~isreal(in_mat)
                    
                    %in_mat=real(in_mat);
                    
                    in_mat=fftshift(fftn(ifftshift(in_mat)));
                    in_mat_0=(in_mat);
                    in_mat_0(~fft_weight)=0;
                    in_mat_0=(in_mat_0+TV_helper_fft_flip(conj(in_mat_0),[1 1 1],1))/2;
                    in_mat=(in_mat+TV_helper_fft_flip(conj(in_mat),[1 1 1],1))/2;
                    
                    signifiant_cost=single(fft_weight>0);
                    signifiant_cost=(signifiant_cost+TV_helper_fft_flip(signifiant_cost,[1 1 1],1))/2; 
                    
                    in_mat(signifiant_cost>0)=in_mat_0(signifiant_cost>0)./signifiant_cost(signifiant_cost>0);
                    
                    in_mat=real(fftshift(ifftn(ifftshift(in_mat))));
                    
                    fft_weight=(fft_weight+TV_helper_fft_flip(fft_weight,[1 1 1],1))/2; 
                    clear in_mat_0;
                    clear signifiant_cost;
                end
                
            end
            
            real_input=isreal(in_mat);
            
            
            
            if real_input
                
                %figure; orthosliceViewer(real(fft_weight));
            end
            if ~real_input
                warning('The input is complex so optimisation might be slower ! ');
            end
            if ~strcmp(reg_name,'tv') && ~strcmp(reg_name,'hessian')
                error('The regularisation must be either ''tv'' or ''hessian'' ')
            end
            if ~isempty(find(fft_weight>1,1)) || ~isempty(find(fft_weight<0,1))
                warning('Better set the weight to value between 0 and 1')
            end
            max_val=max(abs(in_mat(:)));
            if max_val>1 || max_val<0.5
                warning('Better to normalise the input to -1/1');
            end
            if h.parameters.inner_itterations<=0 || h.parameters.outer_itterations<=0
                error('The number of itterations must be of at least 1 ');
            end
            if length(size(in_mat))>3 || length(size(in_mat))<2
                error('Only 2D and 3D matrix are supported');
            end
            
            
            
            
            sz1=size(in_mat,1);
            sz2=size(in_mat,2);
            sz3=1;
            if length(size(in_mat))==3
                sz3=size(in_mat,3);
            end
            
            fft_weight=ifftshift(fft_weight);
            
            A=@(X) 1/sqrt(sz1*sz2*sz3)*fftn(X).*fft_weight;
            if real_input
                A_trans=@(X) real(sqrt(sz1*sz2*sz3)*ifftn(X.*conj(fft_weight)));
            else
                A_trans=@(X) sqrt(sz1*sz2*sz3)*ifftn(X.*conj(fft_weight));
            end
            
            alpha=max(max(abs(fft_weight(:)).^2));
            y=A(in_mat);%the base data
            
            if strcmp(reg_name,'tv')
                cost=@(X) sum(abs(A(X)-y).^2,'all') + 2*h.parameters.TV_strength*TV_helper_TV_val(X);
            elseif strcmp(reg_name,'hessian')
                error('To do');
            else
                error('Unrecognised regularisation name');
            end
            
            s_n=0;
            
            t_n=0;
            t_np=1;
            u_n=in_mat;
            x_n=in_mat;
            clear in_mat;
            
            c_n=0;
            c_np=Inf;
            
            
            cost_history=[];
            figure;
            hax1=axes;
            figure;
            hax2=axes;
            
            %start the itterations
            fprintf(1,'Computation Progress: %3.0f',0);
            for mm=1:h.parameters.outer_itterations
                fprintf(1,'\b\b\b%3.0f',mm);  % Deleting 4 characters (The three digits and the % symbol)
                
                t_n=t_np;
                c_n=c_np;
                
                %denoising for different regularisation
                        
                if strcmp(reg_name,'tv')
                    %tic;
                    if h.parameters.low_memory
                        tmp=u_n-(1/alpha)*A_trans(A(u_n)-y);
                        clear u_n;
                        clear s_n;
                        y=gather(y);
                        x_n=gather(x_n);
                        s_n=TV_helper_TV_FISTA_inner(tmp,h.parameters.TV_strength/alpha,h.parameters.inner_itterations,h.parameters.use_gpu,h.parameters.use_cuda,real_input,h.parameters.min_real,h.parameters.max_real,h.parameters.min_imag,h.parameters.max_imag,h.parameters.use_boundary_value,h.parameters.boundary_value);
                        y=gpuArray(y);
                        x_n=gpuArray(x_n);
                    else
                        s_n=TV_helper_TV_FISTA_inner(u_n-(1/alpha)*A_trans(A(u_n)-y),h.parameters.TV_strength/alpha,h.parameters.inner_itterations,h.parameters.use_gpu,h.parameters.use_cuda,real_input,h.parameters.min_real,h.parameters.max_real,h.parameters.min_imag,h.parameters.max_imag,h.parameters.use_boundary_value,h.parameters.boundary_value);
                    end
                    %toc;
                elseif strcmp(reg_name,'hessian')
                    error('To do');
                    error('use the symetry of the hessian');
                else
                    error('Unrecognised regularisation name');
                end
                t_np=(1+sqrt(1+4*t_n^2))/2;
                c_np=cost(s_n);
                if c_np>c_n
                    c_np=c_n;
                    u_n=x_n+(t_n/t_np)*(s_n-x_n);
                else
                    u_n=s_n+(t_n-1)/t_np*(s_n-x_n);
                    x_n=s_n;
                end
                %display(['cost = ' num2str(c_np)]);
                cost_history(end+1)=gather((c_np(:)));
                plot(hax1,cost_history);drawnow;
                imagesc(hax2,real(squeeze(x_n(:,round(end/2),:))'));colormap gray;drawnow;
            end
            
            fprintf('\n');
            
            out_mat=x_n;
            
            if h.parameters.use_gpu
                out_mat=gather(out_mat);
            end
            
        end
    end
end

function out_mat=TV_helper_TV_FISTA_inner(in_mat,lambda,inner_itt,use_gpu,use_cuda,real_input,min_real,max_real,min_imag,max_imag,use_boundary,boundary_value)
% TV_FISTA_inner_v2(u_n-(1/alpha)*gradient_RI,h.parameters.tv_param/alpha,...
%                     params.nmin, params.nmax, params.kappamax, h.parameters.use_non_negativity,dirichlet_boundary,inner_itt,use_gpu);
if ~exist('use_cuda','var')
    use_cuda=false;
end
if use_gpu
    in_mat=single(gpuArray(in_mat));
else
    in_mat=single(in_mat);
end
if use_gpu && use_cuda
    out_mat=helper_fista_TV_inner_gpu(in_mat,gather(single(lambda)),gather(logical(real_input)),gather(logical(use_boundary)),gather(uint32(inner_itt)),gather(single(min_real)),gather(single(max_real)),gather(single(min_imag)),gather(single(max_imag)),gather(single(boundary_value)));
    %warning('shifted coordinate and shared memory ? split data ? use double in computations ?');
    %warning('partitionning and overlapping data transfer ?');
else    
    dim_num=length(size(in_mat));
    
    if dim_num>=3
        dividend=12;
    else
        dividend=8;
    end
    
    P_n=0;
    P_np{1}=0.*in_mat;
    P_np{2}=0.*in_mat;
    if dim_num==3
        P_np{3}=0.*in_mat;
    end
    R=P_np;
    
    t_n=1;
    t_np=1;
    
    %start the itterations
    for mm=1:inner_itt
        P_n=P_np;
        t_n=t_np;
        P_np=TV_helper_TV_L_trans(TV_helper_project_non_neg(in_mat-lambda*TV_helper_TV_L(R),real_input,min_real,max_real,min_imag,max_imag,use_boundary,boundary_value));
        for kk=1:dim_num
            P_np{kk}=R{kk}+1/(dividend*lambda)*P_np{kk};
        end
        P_np=TV_helper_project_TV(P_np);
        
        t_np=(1+sqrt(1+4*t_n^2))/2;
        for kk=1:dim_num
            R{kk} = P_np{kk} + ((t_n-1)/t_np)*(P_np{kk}-P_n{kk});
        end
        
    end
    
    out_mat=TV_helper_project_non_neg(in_mat-lambda*TV_helper_TV_L(P_np),real_input,min_real,max_real,min_imag,max_imag,use_boundary,boundary_value);
 
end
end

function in_mat=TV_helper_project_non_neg(in_mat,real_input,min_real,max_real,min_imag,max_imag,use_boundary,boundary_value)

in_mat(real(in_mat)<min_real)=min_real+1i*imag(in_mat(real(in_mat)<min_real));
in_mat(real(in_mat)>max_real)=max_real+1i*imag(in_mat(real(in_mat)>max_real));

if ~real_input
    in_mat(imag(in_mat)<min_imag)=1i*min_imag+real(in_mat(imag(in_mat)<min_imag));
    in_mat(imag(in_mat)>max_imag)=1i*max_imag+real(in_mat(imag(in_mat)>max_imag));
end

if use_boundary
    in_mat(1,:,:)=boundary_value;
    in_mat(end,:,:)=boundary_value;
    in_mat(:,1,:)=boundary_value;
    in_mat(:,end,:)=boundary_value;
    in_mat(:,:,1)=boundary_value;
    in_mat(:,:,end)=boundary_value;
end

end

function P=TV_helper_project_TV(P)
if length(size(P{1}))==3
    A=sqrt(max(abs(P{1}).^2+abs(P{2}).^2+abs(P{3}).^2,1));
    P{1}=P{1}./A;
    P{2}=P{2}./A;
    P{3}=P{3}./A;
else
    A=sqrt(max(abs(P{1}).^2+abs(P{2}).^2,1));
    P{1}=P{1}./A;
    P{2}=P{2}./A;
end
end

function out_mat=TV_helper_TV_L(P)
out_mat=P{1}+P{2};
if length(size(out_mat))==3
    out_mat=out_mat+P{3};
end

if length(size(out_mat))==3
    out_mat=out_mat-circshift(P{1},[1 0 0]);
    out_mat=out_mat-circshift(P{2},[0 1 0]);
    out_mat=out_mat-circshift(P{3},[0 0 1]);
else
    out_mat=out_mat-circshift(P{1},[1 0]);
    out_mat=out_mat-circshift(P{2},[0 1]);
end
end

function val=TV_helper_TV_val(in_mat)
P=TV_helper_TV_L_trans(in_mat);
if length(size(in_mat))==3
    val=sum(sqrt(abs(P{1}).^2+abs(P{2}).^2+abs(P{3}).^2),'all');%iso tv
else
    val=sum(sqrt(abs(P{1}).^2+abs(P{2}).^2),'all');%iso tv
end
end

function out_mat=TV_helper_TV_L_trans(in_mat)
if length(size(in_mat))==3
    out_mat{1}=in_mat-circshift(in_mat,[-1 0 0]);
    out_mat{2}=in_mat-circshift(in_mat,[0 -1 0]);
    out_mat{3}=in_mat-circshift(in_mat,[0 0 -1]);
else
    out_mat{1}=in_mat-circshift(in_mat,[-1 0]);
    out_mat{2}=in_mat-circshift(in_mat,[0 -1]);
end
end

function array=TV_helper_fft_flip(array,flip_bool, use_shift)
%use_shift is set to true if the array has the zero frequency centered
for ii = 1:length(flip_bool)
    if flip_bool(ii)
        array=flip(array,ii);
        if mod(size(array,ii),2)==0 || ~use_shift
            array=circshift(array,1,ii);
        end
    end
end
end


