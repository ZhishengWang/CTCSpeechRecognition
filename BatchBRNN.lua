------------------------------------------------------------------------
--[[ BatchBRNN ]] --
-- Adds sequence-wise batch normalization to cudnn RNN modules.
-- For a simple RNN: ht = ReLU(B(Wixt) + Riht-1 + bRi) where B
-- is the batch normalization.
-- Expects size seqLength x minibatch x inputDim.
-- Returns seqLength x minibatch x outputDim.
-- Can specify an rnnModule such as cudnn.LSTM (defaults to RNNReLU).
------------------------------------------------------------------------
local BatchBRNN, parent = torch.class('cudnn.BatchBRNN', 'nn.Sequential')

function BatchBRNN:__init(inputDim, outputDim)
    parent.__init(self)

    self.view_in = nn.View(1, 1, -1):setNumInputDims(3) -- input feature map should be the last 3 dims
    self.view_out = nn.View(1, -1):setNumInputDims(2) ---- input feature map should be the last 2 dims
    --self.view_in = nn.View()
    
    self.rnn = cudnn.RNN(outputDim, outputDim, 1)
    local rnn = self.rnn
    rnn.inputMode = 'CUDNN_SKIP_INPUT'
    rnn.bidirectional = 'CUDNN_UNIDIRECTIONAL'
    --rnn.numDirections = 2
    rnn:reset()
    self:add(self.view_in)
    self:add(nn.Linear(inputDim, outputDim, false)) --module = nn.Linear(inputDimension, outputDimension, [bias = false])
    self:add(nn.BatchNormalization(outputDim)) --only accepts 2D inputs.
    self:add(self.view_out)
    self:add(rnn)
    --uncommit it to fit for unidirectional rnn.
    --self:add(nn.View(-1, 2, outputDim):setNumInputDims(2))  --from seqlen x minibatch x outputdim to seqlen x (minibatch/2) x 2 outputdim
    --self:add(nn.Sum(3))  --nn.sum:Applies a sum operation over dimension 3
end

function BatchBRNN:updateOutput(input) --enable valiable length of utterance
    local T, N = input:size(1), input:size(2)
    self.view_in:resetSize(T * N, -1)
    self.view_out:resetSize(T, N, -1)
    return parent.updateOutput(self, input)
end

function BatchBRNN:__tostring__()
    local tab = '  '
    local line = '\n'
    local next = ' -> '
    local str = 'BatchBRNN'
    str = str .. ' {' .. line .. tab .. '[input'
    for i=1,#self.modules do
        str = str .. next .. '(' .. i .. ')'
    end
    str = str .. next .. 'output]'
    for i=1,#self.modules do
        str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
    end
    str = str .. line .. '}'
    return str
end
