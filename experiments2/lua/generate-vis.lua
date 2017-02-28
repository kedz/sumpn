-- Options

local opt = lapp [[
Generate visualization meta data for Model2.
Options:
  --data     (string)        Path to data file.
  --output   (string)        Path to write meta data.
  --model    (string)        Path to model binary.
  --gpu      (default 0)     Which gpu to use. Default is cpu.
  --vocab    (string)        Path to target vocab.
  --samples  (default 25)    Number of examples to extract meta data for.
  --seed     (default 1986)  Random seed.
  --progress (default true)  Show progress bar.
]]

require("rnn")
local dtools = require('dtools')
require("Model2")
require("lfs")
local lyaml = require("lyaml")

local use_gpu = false
if opt.gpu > 0 then use_gpu = true end

if use_gpu then
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

print("Reading target vocab from " .. opt.vocab .. " ...")
local id2vocab = dtools.read_vocab(opt.vocab)
local vsize = #id2vocab
print(vsize .. " tokens read.")

print("Reading data from " .. opt.data .. " ...")
local dataset = dtools.read_data(opt.data)
print(dataset[1]:size(1) .. " examples read.")

print("Loading model from " .. opt.model .. " ...")
local model = torch.load(opt.model)
if use_gpu then
    print("Moving model to gpu ...")
    model = model:cuda()
end
    
print("REMOVING __UNK__ __PER__ __ORG__ __LOC__ tokens.")
model.tgt_emb_out[5]:fill(0)
model.tgt_emb_out[6]:fill(0)
model.tgt_emb_out[7]:fill(0)
model.tgt_emb_out[8]:fill(0)

local backbone_tokens = dataset[8]
local support_tokens = dataset[9]
local num_samples = math.min(#backbone_tokens, opt.samples)
local max_iters = math.ceil(num_samples / 15)

local batches = dtools:iterbatch(dataset, use_gpu, 15, 1, vsize)
local results = {}

for batch in batches do
   if opt.progress then xlua.progress(batch.iter, max_iters) end
   local pred_tgt, metadata = model:greedy_predict_debug(batch.backbone, 
                                                         batch.support,
                                                         batch.target_input,
                                                         batch.target_output)
   for i=1,math.min(num_samples - #results, #metadata) do
      local offset = batch.backbone:size(2) - batch.backbone_sizes[i] + 1 
      metadata[i]["target"] = {}
      metadata[i]["target"]["predicted"] = dtools.extract_tokens(
         backbone_tokens[batch.indices[i]], 
         support_tokens[batch.indices[i]],
         offset, 
         id2vocab, 
         pred_tgt[i])
      metadata[i]["target"]["gold"] = dtools.extract_tokens(
         backbone_tokens[batch.indices[i]], 
         support_tokens[batch.indices[i]],
         offset, 
         id2vocab,
         batch.target_output[i])

      table.insert(results, metadata[i]) 
   end

   if #results == num_samples then break end
end

print("Writing to " .. opt.output .. " ...")
f = io.open(opt.output, "w")
f:write(lyaml.dump({results}))
f:close()
