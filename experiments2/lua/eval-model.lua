-- Options

local opt = lapp [[
Train lead model1.
Options:
  --train-data    (default '')     Training data path.
  --dev-data      (default '')     Development data path.
  --model         (string)     Path model directory.
                               Default is no write.
  --batch-size (default 15)    Batch size. 
  --start-epoch (default 1)     Starting epoch to evaluate.
  --stop-epoch  (default 50)    Last epoch to evaluate (inclusive).
  --vocab-size (number)           Number of words in target vocab.
  --seed       (default 1986)  Random seed.
  --gpu        (default 0)     Which gpu to use. Default is cpu.
  --progress   (default true)  Show progress bar.
]]

require("rnn")
local dtools = require('dtools')
require("Model2")

local useGPU = false
if opt.gpu > 0 then useGPU = true end

if useGPU then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpu)
    cutorch.manualSeed(opt.seed)
    torch.manualSeed(opt.seed)
    print("running on gpu-" .. opt.gpu)

else
    torch.manualSeed(opt.seed)
    print("running on cpu")
end

local train_dataset = nil
local dev_dataset = nil
local test_dataset = nil

if string.len(opt.train_data) > 0 then
    train_dataset = dtools.read_data(opt.train_data)
    train_num_ex = train_dataset[1]:size(1)
    train_max_steps = math.floor(train_num_ex / opt.batch_size)
    if train_num_ex % opt.batch_size == 0 then 
        train_max_steps = train_max_steps + 1 
    end
    train_target_nnz = train_dataset[5]:nonzero():size(1)
end


if string.len(opt.dev_data) > 0 then
    dev_dataset = dtools.read_data(opt.dev_data)
    dev_num_ex = dev_dataset[1]:size(1)
    dev_max_steps = math.floor(dev_num_ex / opt.batch_size)
    if dev_num_ex % opt.batch_size == 0 then 
        dev_max_steps = dev_max_steps + 1 
    end
    dev_target_nnz = dev_dataset[5]:nonzero():size(1)
end

print("Evaluating epochs " .. opt.start_epoch .. " ... " .. opt.stop_epoch)
for epoch=opt.start_epoch,opt.stop_epoch do
    print("Epoch " .. epoch)

    local modelFile = opt.model .. "/model-" .. epoch .. ".bin"
    local model = torch.load(modelFile)

    if useGPU then
        model = model:cuda()
    end
    
    --model:float()
    --useGPU = false


    local nll_train = 0
    if train_dataset ~= nil then
        local batch_iter = dtools:iterbatch(
            train_dataset, useGPU, opt.batch_size, 1000, opt.vocab_size)
        for batch in batch_iter do
            if opt.progress then
                xlua.progress(batch.iter, train_max_steps)
            end
            local nll_batch = model:loss(
                batch.backbone, batch.support, 
                batch.target_input, batch.target_output)
            nll_train = nll_train + nll_batch
        end
    end



    local nll_dev = 0
    if dev_dataset ~= nil then
        local batch_iter = dtools:iterbatch(
            dev_dataset, useGPU, opt.batch_size, 1000, opt.vocab_size)
        for batch in batch_iter do
            if opt.progress then
                xlua.progress(batch.iter, dev_max_steps)
            end
            local nll_batch = model:loss(
                batch.backbone, batch.support, 
                batch.target_input, batch.target_output)
            nll_dev = nll_dev + nll_batch
        end
    end

    if train_dataset ~= nil then
        perpl_train = torch.exp(nll_train / train_target_nnz)
        print(epoch, "   Training perplexity = " .. perpl_train)
        --resultString = resultString .. "\t" .. devPerpl .. "\t" .. devAcc
    end
    if dev_dataset ~= nil then
        perpl_dev = torch.exp(nll_dev / dev_target_nnz)
        print(epoch, "Development perplexity = " .. perpl_dev)
        --resultString = resultString .. "\t" .. devPerpl .. "\t" .. devAcc
    end

    



end 
