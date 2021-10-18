function I_out = RichardsonLucy_MS(varargin)
    [J, psf, iteration, GPU_on]= parse_inputs(varargin{:}); % orthosliceViewer(gather(I_original))

% Initialize J
    if GPU_on
        J{1} = gpuArray(J{1}); % orthosliceViewer(gather(I))
        psf = gpuArray(psf); % orthosliceViewer(gather(psf))
    end
    J{2} = J{1};
    J{3} = 0;


% Initialize H

    sizeI = size(psf);
    H = fftshift(fftn(ifftshift(psf)));
    nElem = prod(sizeI);
    nOps  = 0;
    for k=1:ndims(psf)
      nffts = nElem/sizeI(k);
      nOps  = nOps + sizeI(k)*log2(sizeI(k))*nffts; 
    end
    if max(abs(imag(H(:))))/max(abs(H(:))) <= nOps*eps % Discard the imaginary part of the psf if it's within roundoff error.
      H = real(H);
    end % orthosliceViewer(log10(abs(gather(H))))

% Initialize other parameters
    WEIGHT = ones(size(psf));
    idx = repmat({':'},[1 length(sizeI)]);
    numNSdim = find(sizeI~=1);
    for k = numNSdim % index replicates for non-singleton PSF sizes only
        idx{k} = reshape(repmat(1:sizeI(k),[1 1]),[1*sizeI(k) 1]);
    end
    J{2} = J{2}(idx{:});
    scale = real(fftshift(ifftn(ifftshift(conj(H).*fftshift(fftn(ifftshift(WEIGHT(idx{:})))))))) + sqrt(eps);
    J{4}(prod(sizeI)*1^length(numNSdim),2) = 0;
    wI = max(WEIGHT.*J{1},0); clear WEIGHT
    % orthosliceViewer(gather(wI))
    
    lambda = 2*any(J{4}(:)~=0);
    for j1 =  lambda + (1:iteration)
        if j1 > 2
            lambda = abs((J{4}(:,1).'*J{4}(:,2))/(J{4}(:,2).'*J{4}(:,2) +eps));
            lambda = max(min(lambda,1),0);% stability enforcement
            if lambda == 0
                disp(j1)
                break;
            end
        end
        Y = max(J{2} + lambda*(J{2} - J{3}),0);% plus positivity constraint
        
        if sum(Y(:)) == 0
            disp(j1)
            break;
        end
        
        J{3} = J{2};
        J{2} = max(Y.*real(fftshift(ifftn(ifftshift(conj(H).*fftshift(fftn(ifftshift(wI ./ real(fftshift(ifftn(ifftshift(fftshift(fftn(ifftshift(Y))).*H)))))))))))./scale,0);
        J{4} = [J{2}(:)-Y(:) J{4}(:,1)];
        %  orthosliceViewer(gather(gather(Y)))

    end %  orthosliceViewer(gather(J{2}))
    clear wI H scale Y;
    
    I_out = gather(J{2});
    %  orthosliceViewer(gather(I_out))
end


function [J, psf, iteration, GPU_on]= parse_inputs(varargin)
iteration = [];iteration_d = 10;% Number of  iterations, usually produces good
GPU_on =[];GPU_on_d = true;% GPU on

narginchk(2,4);

J{1} = varargin{1};
psf = varargin{2};
switch nargin
    case 3 %                 RichardsonLucy_MS(I,PSF,iteration)
        iteration = varargin{3};
    case 4 %                 RichardsonLucy_MS(I,PSF,iteration,GPU_on)
        iteration = varargin{3};
        GPU_on = varargin{4};
end

if isempty(iteration)
    iteration = iteration_d;
end

if isempty(GPU_on)
    GPU_on = GPU_on_d;
end
end