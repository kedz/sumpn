require 'nn'

local LogSoftMaxMaskZero, Parent = torch.class('nn.LogSoftMaxMaskZero', 
    'nn.LogSoftMax')

function LogSoftMaxMaskZero:updateOutput(input)

    self.mask = torch.eq(input, 0)
    input:maskedFill(self.mask, -math.huge)
    self.output = Parent.updateOutput(self, input)
    self.output:maskedFill(self.mask, 0)
    input:maskedFill(self.mask, 0)
    
    return self.output
    
end

function LogSoftMaxMaskZero:updateGradInput(input, grad_output)
    self.gradInput = Parent.updateGradInput(self, input, grad_output)
    self.gradInput:maskedFill(self.mask, 0)
    return self.gradInput
end
