--[[Trains the CTC model using the AN4 audio database.]]

local Network = require 'Network'

local epochs = 200
local root_path = '../../../../media/vlsi/vlsi-data/sharedata/speech-corpus/LibriSpeech/LibriSpeech'

local networkParams = {
    loadModel = true,
    saveModel = true,
    modelName = 'DeepSpeechModel',
    backend = 'cudnn', -- switch to rnn to use CPU
    nGPU = 2, -- Number of GPUs, set -1 to use CPU
    trainingSetLMDBPath = root_path .. '/lmdb-testclean-100train/train/',-- online loading path data.
    validationSetLMDBPath = root_path .. '/lmdb-testclean-100train/test/',
    logsTrainPath = '../CTC-logs/logs/TrainingLoss/',
    logsValidationPath = '../CTC-logs/logs/ValidationScores/',
    modelTrainingPath = '../CTC-logs/models/',
    fileName = '../CTC-logs/CTCNetwork.t7',
    dictionaryPath = './dictionary',
    batchSize = 10,
    validationBatchSize = 1,
    validationIterations = 100,
    saveModelInTraining = true, -- saves model periodically through training
    saveModelIterations = 50
}
--Parameters for the stochastic gradient descent (using the optim library).
local sgdParams = {
    learningRate = 1e-6,
    learningRateDecay = 1e-9,
    weightDecay = 0,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

--Create and train the network based on the parameters and training data.
Network:init(networkParams)

Network:trainNetwork(epochs, sgdParams)

--Creates the loss plot.
Network:createLossGraph()

print("finished")
