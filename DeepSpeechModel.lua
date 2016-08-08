require 'UtilsMultiGPU'
require 'SequenceWise'

-- Chooses RNN based on if GRU or backend GPU support.
local function getRNNModule(nIn, nHidden, GRU, is_cudnn)
    if (GRU) then
        if is_cudnn then
            require 'cudnn'
            return cudnn.GRU(nIn, nHidden, 1)
        else
            require 'rnn'
        end
        return nn.GRU(nIn, nHidden)
    end
    if is_cudnn then
        require 'BatchBRNNReLU'
        return cudnn.BatchBRNNReLU(nIn, nHidden, 1)
    else
        require 'rnn'
    end
    return nn.SeqLSTM(nIn, nHidden)
end

local function ReLU(isCUDNN)
    if (isCUDNN) then return cudnn.ClippedReLU(true, 20) else return nn.ReLU(true) end
end

-- Creates the covnet+rnn structure.
local function deepSpeech(nGPU, isCUDNN)
    local model = nn.Sequential()
    if (isCUDNN) then require 'cudnn' end
    local GRU = false
    local conv = nn.Sequential()
    -- (nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH]) conv layers.
    conv:add(nn.SpatialConvolution(1, 32, 5, 21, 2, 2):noBias())
    conv:add(nn.SpatialBatchNormalization(32)) -- only accepts 4D inputs.
    conv:add(ReLU(isCUDNN))
    conv:add(nn.SpatialConvolution(32, 32, 5, 11, 1, 2):noBias())
    conv:add(nn.SpatialBatchNormalization(32))
    conv:add(ReLU(isCUDNN))

    local rnnInputsize = 32 * 31 -- based on the above convolutions.
    local rnnHiddenSize = 1760 -- size of rnn hidden layers
    local nbOfHiddenLayers = 8
    --initial:1760*8
    conv:add(nn.View(rnnInputsize, -1):setNumInputDims(3)) -- batch x features x seqLength
    conv:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- seqLength x batch x features

    local rnn = nn.Sequential()
    local rnn_module = getRNNModule(rnnInputsize, rnnHiddenSize,
        GRU, isCUDNN)
    rnn:add(rnn_module:clone())
    rnn_module = getRNNModule(rnnHiddenSize,
        rnnHiddenSize, GRU, isCUDNN)

    for i = 1, nbOfHiddenLayers - 1 do
        rnn:add(rnn_module:clone())
    end

    local post_sequential = nn.Sequential()
    post_sequential:add(nn.BatchNormalization(rnnHiddenSize))
    post_sequential:add(nn.Linear(rnnHiddenSize, 29))

    model:add(conv)
    model:add(rnn)
    model:add(nn.SequenceWise(post_sequential))
    model:add(nn.Transpose({1, 2})) -- batch x seqLength x features
	print(model)
    model = makeDataParallel(model, nGPU, isCUDNN)
    return model
end

-- Based on convolution kernel and strides.
local function calculateInputSizes(sizes)
    --print("sizes before calculate:",sizes)
    sizes = torch.floor((sizes - 5) / 2 + 1) -- conv1
    sizes = torch.floor((sizes - 5) / 1 + 1) -- conv2
    --print("sizes after calculate:",sizes)
    return sizes
end

return { deepSpeech, calculateInputSizes }
