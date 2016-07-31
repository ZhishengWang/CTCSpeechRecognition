require 'rnn'
require 'nngraph'
--require 'cunn'

local ffi = require 'ffi'

local default_GPU = 1
function makeDataParallel(model, nGPU, is_cudnn) --model type can only be nn.DataParallelTable.(desided by fun loadDataParallel)
    if nGPU >= 1 then
        if is_cudnn then
            cudnn.fastest = true
            model = cudnn.convert(model, cudnn)
        end
        if nGPU > 1 then
            gpus = torch.range(1, nGPU):totable()
            --Split along first (batch) dimension,replicate model to GPU 1,...,nGPU, and use a seperate thread for each replica. 
            dpt = nn.DataParallelTable(1):add(model, gpus):threads(function()
                require 'nngraph'
                require 'SequenceWise'
                if is_cudnn then
                    local cudnn = require 'cudnn'
                    cudnn.fastest = true  --set to true to pick the fastest convolution algorithm, default is false.
                    cudnn.verbose = true -- this prints out some more verbose information useful for debugging
                    require 'BatchBRNNReLU'
                else
                    require 'rnn'
                end
            end)
            dpt.gradInput = nil  --??
            model = dpt
        end
        model:cuda()
    end
    return model
end

local function cleanDPT(module)
    -- This assumes this DPT was created by the function above: all the
    -- module.modules are clones of the same network on different GPUs
    -- hence we only need to keep one when saving the model to the disk.
    local newDPT = nn.DataParallelTable(1)
    cutorch.setDevice(default_GPU)
    newDPT:add(module:get(1), default_GPU)
    return newDPT
end

function saveDataParallel(fileName, model)
    if torch.type(model) == 'nn.DataParallelTable' then
        torch.save(fileName, cleanDPT(model))
    elseif torch.type(model) == 'nn.Sequential' then
        local temp_model = nn.Sequential()
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                temp_model:add(cleanDPT(module))
            else
                temp_model:add(module)
            end
        end
        torch.save(fileName, temp_model)
    elseif torch.type(model) == 'nn.gModule' then
        torch.save(fileName, model)
    else
        error('This saving function only works with Sequential or DataParallelTable modules.')
    end
end

function loadDataParallel(filename, nGPU, is_cudnn)
    require 'SequenceWise'
    if (is_cudnn) then
        require 'cudnn'
        require 'BatchBRNNReLU'
    end
    local model = torch.load(filename)
    if torch.type(model) == 'nn.DataParallelTable' then
        return makeDataParallel(model:get(1):float(), nGPU, is_cudnn)
    elseif torch.type(model) == 'nn.Sequential' then
        for i, module in ipairs(model.modules) do
            if torch.type(module) == 'nn.DataParallelTable' then
                model.modules[i] = makeDataParallel(module:get(1):float(), nGPU, is_cudnn)
                --module:get(1):return module[1]
            end
        end
        return model
    elseif torch.type(model) == 'nn.gModule' then --graph model
        model = makeDataParallel(model, nGPU, is_cudnn)
        return model
    else
        error('The loaded model is not a Sequential or DataParallelTable module.')
    end
end