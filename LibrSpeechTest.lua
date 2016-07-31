--[[Calulates the WER using the AN4 Audio database test set.
-- Uses model created by AN4CTCTrain.]]

local Network = require 'Network'
local root_path = '../../../../media/vlsi/vlsi-data/sharedata/speech-corpus/LibriSpeech/LibriSpeech'

-- Load the network from the saved model.
local networkParams = {
    loadModel = true,
    saveModel = false,
    fileName = arg[1] or "../CTC-logs/models/CTCNetwork.t7", -- Rename the evaluated model to CTCNetwork.t7 or pass the file path as an argument.
    modelName = 'DeepSpeechModel',
    backend = 'cudnn',
    nGPU = 2, -- Number of GPUs, set -1 to use CPU
    trainingSetLMDBPath = root_path .. '/lmdb-testclean-100train/train/', -- online loading path
    validationSetLMDBPath = root_path .. '/lmdb-testclean-100train/test/',
    logsTrainPath = '../CTC-logs/logs/TrainingLoss/',
    logsValidationPath = '../CTC-logs/logs/TestScores/',
    modelTrainingPath = '../CTC-logs/models/',
    dictionaryPath = './dictionary',
    batchSize = 1,
    validationBatchSize = 1,
    validationIterations = 2620 -- batch size 1, goes through 130 samples.
}

Network:init(networkParams)

print("Testing network...")
local wer = Network:testNetwork()
print(string.format('Number of iterations: %d average WER: %2.f%%', networkParams.validationIterations, 100 * wer))
print(string.format('More information written to log file at %s', networkParams.logsValidationPath))
