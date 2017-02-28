-- Options

local opt = lapp [[
Train lead model1.
Options:
  --input-vocab   (string)     Vocab path.
  --output-vocab  (string)     Vocab path.
  --data       (string)        Training data path.
  --save       (default '')    Directory to write models/vocab. 
                               Default is no write.
  --batch-size (default 15)    Batch size. 
  --dims       (default 32)    Embedding/lstm dimension.
  --layers     (default 1)     Number of stacked lstm layers.
  --lr         (default 1E-3)  Learning rate.
  --epochs     (default 50)    Max number of training epochs.
  --seed       (default 1986)  Random seed.
  --gpu        (default 0)     Which gpu to use. Default is cpu.
  --progress   (default true)  Show progress bar.
]]

require("rnn")
local dtools = require('dtools')
require("Model2")
require("lfs")

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

local saveModels = string.len(opt.save) > 0

if saveModels then
    local result, msg = lfs.mkdir(opt.save)
    if result == nil and msg ~= "File exists" then
        print(msg)
        os.exit()
    end
else
    print("WARNING: not saving models to file!")
end

-------------------------------------------------------------------------------
------------------------------   READING DATA    ------------------------------
-------------------------------------------------------------------------------

print("Reading input vocab from " .. opt.input_vocab .. " ...")
local id2vocab_input = dtools.read_vocab(opt.input_vocab)
local input_vsize = #id2vocab_input
print(input_vsize .. " tokens read.")

print("Reading input vocab from " .. opt.output_vocab .. " ...")
local id2vocab_output = dtools.read_vocab(opt.output_vocab)
local output_vsize = #id2vocab_output
print(output_vsize .. " tokens read.")

print("Reading data from " .. opt.data .. " ...")
local dataset = dtools.read_data(opt.data)
local num_ex = dataset[1]:size(1)
local max_steps = math.floor(num_ex / opt.batch_size)
if num_ex % opt.batch_size == 0 then max_steps = max_steps + 1 end

local target_nnz = dataset[5]:nonzero():size(1)
print({dataset[1], dataset[2], dataset[3], dataset[4], dataset[5]})

-------------------------------------------------------------------------------
------------------------------    SETUP MODEL    ------------------------------
-------------------------------------------------------------------------------

local model = nn.Model2(input_vsize, output_vsize, opt.dims, opt.lr, useGPU)
--if useGPU then model:cuda() end


-------------------------------------------------------------------------------
------------------------------     TRAINING      ------------------------------
-------------------------------------------------------------------------------

print("\nTraining model for " .. opt.epochs .. " epochs... \n")

for epoch=1,opt.epochs do
    print("Epoch " .. epoch .. " ...")

    local batch_iter = dtools:iterbatch(
        dataset, useGPU, opt.batch_size, 1000, output_vsize)
    local nll = 0
    local total_nz = 0

    for batch in batch_iter do
        if opt.progress then
            xlua.progress(batch.iter, max_steps)
        end
        local nll_batch = model:train_step(
            batch.backbone, batch.support, 
            batch.target_input, batch.target_output)
        nll = nll + nll_batch
        total_nz = total_nz + batch["target_nnz"]

        if batch.iter == max_steps and epoch == opt.epochs then
            print(model:greedy_predict(batch.backbone, batch.support,
                batch.target_input, batch.target_output))
        end
        --if batch.iter % 100 == 0 then
        --print("avg perpl. ", torch.exp(nll / total_nz), torch.max(model.params), torch.mean(model.params), torch.min(model.params), torch.max(model.grad_params), torch.min(model.grad_params))
        
        --end

        
    end


    print(epoch, "avg perplexity = " .. torch.exp(nll / target_nnz))
    if saveModels then
        local modelFile = opt.save .. "/model-" .. epoch .. ".bin"
        print("Writing model to " .. modelFile .. " ...")
        torch.save(modelFile, model:float())
    end
end
