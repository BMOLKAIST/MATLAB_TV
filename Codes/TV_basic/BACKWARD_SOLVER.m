classdef BACKWARD_SOLVER < handle
    properties (SetAccess = protected, Hidden = false)
        parameters = struct(...
            'size',[100 100 100], ...
            'wavelength', 0.532, ...
            'NA',1.2, ...
            'RI_bg',1.336, ...
            'resolution',[0.1 0.1 0.1], ...
            'vector_simulation', true, ...
            'use_abbe_sine', true ...
        );
    end
    methods
        function h=BACKWARD_SOLVER(params)
            if nargin==1
                warning ('off','all');
                h.parameters=update_struct(h.parameters, params);
                warning ('on','all');
            end
        end
        function [RI]=solve(h,input_field,output_field)
            error("You need to specify the backward solver to solve");
        end
    end
end