classdef MishLayer < nnet.layer.Layer & nnet.internal.cnn.layer.Traceable ...
        & nnet.internal.cnn.layer.CPUFusableLayer ...
        & nnet.internal.cnn.layer.BackwardOptional
    % MishLayer   Mish layer
    %
    %   To create a mish layer, use MishLayer.
    %
    %   A mish layer. This type of layer performs the mish operation
    %   on its input. The mish operation is given by X.*tanh(log(1+exp(X))), where
    %   X is the layer input.
    %
    %   MishLayer properties:
    %       Name                   - Layer name
    %       NumInputs              - Number of layer inputs
    %       InputNames             - Layer input names
    %       NumOutputs             - Number of layer ouputs
    %       OutputNames            - Layer output names
    %
    %   Example:
    %       Create a mish layer.
    %
    %       layer = MishLayer
    %
    %
    %#codegen   
	   

    methods
        function layer = MishLayer()
           
        end
        
        function Z = predict(~, X)
            % Forward input data through the layer at prediction time and
            % output the result
            Z = X.*tanh(log(1+exp(X)));
        end
        

        function dLdX = backward(~, X, ~, dLdZ, ~)
            % Backward propagate the derivative of the loss function through 
            % the layer
             w=(4.*(X+1))+(4.*(exp(2.*X)))+exp(3.*X)+(exp(X).*((4.*X)+6));
             q=(2.*exp(X))+exp(2.*X)+2;
             dZdX = (exp(X).*w).*(1./q.^2);
             dLdX = dLdZ.*dZdX;
        end
    end

    methods (Hidden)
        function layerArgs = getFusedArguments(~)
             % getFusedArguments  Returned the arguments needed to call the
            % layer in a fused network.
            layerArgs = { 'mish' };
        end

        function tf = isFusable(~, ~, ~)
            % isFusable  Indicates if the layer is fusable in a given network.
            tf = true;
        end
    end
    
     methods (Static = true, Access = public)       
         function name = matlabCodegenRedirect(~)
             name = 'nnet.internal.cnn.coder.layer.MishLayer';
         end
     end 
end

function messageString = iGetMessageString( messageID )
messageString = getString( message( messageID ) );
end
