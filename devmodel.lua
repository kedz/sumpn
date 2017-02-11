dtools = {}

function dtools.read_vocab(path)
    local id2vocab = {}
    local special_toks = {"<S>", "<D>", "<E>", "<B>", "__ENTITY__", "__UNK__"}

    for i, w in ipairs(special_toks) do
        table.insert(id2vocab, w)
    end

    for line in io.lines(path) do
        table.insert(id2vocab, line)
    end

    return id2vocab
end

function dtools.read_data(path)

    local backbones = {}
    local supports = {}
    local targets_in = {}
    local targets_out = {}

    for line in io.lines(path) do

        local backbone, support, target = table.unpack(
            stringx.split(line, " | "))
        local backbone_tensor = torch.LongTensor(stringx.split(backbone, " "))
        local targets_tensor = torch.LongTensor(stringx.split(target, " "))
        local support_tensor = torch.LongTensor(stringx.split(support, " "))
        
        -- data is originally 0 indexed. Oh torch/lua and your choices.
        
        backbone_tensor:add(1) 
        targets_tensor:add(1)
        support_tensor:add(1)

        table.insert(
            targets_in,
            targets_tensor:narrow(1, 1, targets_tensor:size(1) - 1))
        
        table.insert(
            targets_out,
            targets_tensor:narrow(1, 2, targets_tensor:size(1) - 1))

        table.insert(backbones, backbone_tensor)
        table.insert(supports, support_tensor)

    end

    local bb_data, bb_sizes = dtools.pad(backbones, 0, "left", true)
    local tgt_in = dtools.pad(targets_in, 0, "right")    
    local tgt_out = dtools.pad(targets_out, 0, "right")    
    local sp = dtools.pad(supports, 0, "right")

    return bb_data, bb_sizes, sp, tgt_in, tgt_out

end

function dtools.pad(data, pad_value, direction, return_sizes)
     
    local is_left

    if direction == "left" then
        is_left = true
    elseif direction == "right" then
        is_left = false
    else
        error("Argument 3 must be either 'left' or 'right'.")
    end

    if return_sizes == nil then
        return_sizes = false
    end

    local sizes = torch.LongTensor(#data):zero()
    for i=1,#data do
        sizes[i] = data[i]:size(1)
    end

    local max_size = torch.max(sizes)

    padded_data = torch.Tensor():typeAs(data[1]):resize(#data, max_size)
    padded_data:fill(pad_value)

    if is_left then
        for i=1,#data do
            local size_i = sizes[i]
            local start = max_size - size_i + 1
            local slice = padded_data[i]:narrow(1, start, size_i)
            slice:copy(data[i])
        end
    else
        for i=1,#data do
            local size_i = sizes[i]
            local slice = padded_data[i]:narrow(1, 1, size_i)
            slice:copy(data[i])
        end
    end

    if return_sizes then
        return padded_data, sizes
    else
        return padded_data
    end

end

function dtools:correct_offset(tgt_out, bb, bb_sizes, tgt, vsize, pad_val)
    -- Adjust tgt copy indices to account for changes in the left padding
    -- of batch bb matrix. If its confusing, it is.

    local n_data = bb_sizes:size(1)
    local input_size = bb:size(2)
    local output_size = tgt:size(2)
    local mask = torch.le(tgt, vsize):cmul(torch.ne(tgt, pad_val))
    local pad_mask = torch.le(tgt, vsize)
    tgt_out:resizeAs(tgt)
    tgt_out:copy(bb_sizes:view(n_data, 1):expand(n_data, output_size))
    tgt_out:mul(-1)
    tgt_out:add(input_size)
    tgt_out:maskedFill(pad_mask, 0)
    tgt_out:add(tgt)
    return tgt_out
end

local json = require('json')
local id2vocab_in = dtools.read_vocab("data/input-vocab25k.txt")
local id2vocab_out = dtools.read_vocab("data/output-vocab1k.txt")
local bb, bb_sizes, sp, tgt_in, tgt_out = dtools.read_data(
    "data/lead-data.txt")

tgt_in_c = torch.LongTensor()
tgt_out_c = torch.LongTensor()
tgt_in = dtools:correct_offset(tgt_in_c, bb, bb_sizes, tgt_in, 1006, 0)
tgt_out = dtools:correct_offset(tgt_out_c, bb, bb_sizes, tgt_out, 1006, 0)

bb = bb:float()
sp = sp:float()
tgt_in = tgt_in:float()
tgt_out = tgt_out:float()

require 'LeadModel'

torch.manualSeed(1986)

local model = nn.LeadModel(25006, 1006, 128, .01)
print("ready")
for i=1,75 do
local nll = model:train_step(bb, sp, tgt_in, tgt_out)
print(i, nll)
end

print(tgt_out)
local debug_data = model:greedy_predict_debug(bb, sp, tgt_in, tgt_out)
json.save("dev2.json", debug_data)

