-- Options

local opt = lapp [[
Train lead model1.
Options:
  --data    (default '')       Data path.
  --output    (default '')     Directory to write summaries.
  --model         (string)     Path model directory.
                               Default is no write.
  --gpu        (default 0)     Which gpu to use. Default is cpu.
  --vocab  (string)           Path to target vocab.
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
    print("running on gpu-" .. opt.gpu)
else
    print("running on cpu")
end

local result, msg = lfs.mkdir(opt.output)
if result == nil and msg ~= "File exists" then
    print(msg)
    os.exit()
end

print("Reading vocab from " .. opt.vocab .. " ...")
local id2vocab = dtools.read_vocab(opt.vocab)
local vsize = #id2vocab
print(vsize .. " tokens read.")



print("Reading data.")
local dataset = dtools.read_data(opt.data)
local num_ex = dataset[1]:size(1)
local target_nnz = dataset[5]:nonzero():size(1)

print("Loading model.")
local model = torch.load(opt.model)
if useGPU then
    model = model:cuda()
end
    
print("REMOVING __UNK__ __PER__ __ORG__ __LOC__ tokens.")
model.tgt_emb_out[5]:fill(0)
model.tgt_emb_out[6]:fill(0)
model.tgt_emb_out[7]:fill(0)
model.tgt_emb_out[8]:fill(0)

local doc_iter = dtools:iterdoc(dataset, useGPU, vsize)

for doc in doc_iter do
   xlua.progress(doc.iter, doc.max_iter)
   local backbone_tokens = doc.backbone_tokens
   local support_tokens = doc.support_tokens

   local pred_tgt = model:greedy_predict(doc.backbone, doc.support) 
  
   local lines = {}
   for i=1,pred_tgt:size(1) do
      local offset = doc.backbone:size(2) - doc.backbone_sizes[i] + 1 
      local tokens = dtools.extract_tokens(
          backbone_tokens[i], support_tokens[i], offset, 
                id2vocab, pred_tgt[i])
      table.insert(lines, table.concat(tokens, " "))
   end
   local filename = opt.output .. "/" .. doc.id
   local f = io.open(filename, "w")
   f:write(table.concat(lines, "\n"))
   f:close()

end
