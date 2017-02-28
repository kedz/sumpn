require 'nn'
require 'rnn'
require 'optim'
require 'LogSoftMaxMaskZero'

local LeadModel = torch.class('nn.LeadModel')

function LeadModel:__init(in_vsize, out_vsize, dim_size, lr)

    self.in_vsize = in_vsize
    self.out_vsize = out_vsize
    self.dim_size = dim_size
    self.lr = lr
    self.optim_state = {learningRate=lr}

    self:build_network()
    self:allocate_memory()

end

function LeadModel:cuda()
    self.tgt_vocab_in = self.tgt_vocab_in:cuda()  
    self.tgt_vocab_out = self.tgt_vocab_out:cuda()  
    self.tgt_vocab_in_grad = self.tgt_vocab_in_grad:cuda()  
    self.tgt_vocab_out_grad = self.tgt_vocab_out_grad:cuda()  
    self.backbone_encoder = self.backbone_encoder:cuda()
    self.support_encoder = self.backbone_encoder:cuda()
    self.target_input_embeddings = self.target_input_embeddings:cuda()
    self.target_output_embeddings = self.target_output_embeddings:cuda()
    self.target_input_selector = self.target_input_selector:cuda()
    self.target_selector_bridge = self.target_selector_bridge:cuda()
    self.predict = self.predict:cuda()
    self.lsm = self.lsm:cuda()
    self.criterion = self.criterion:cuda()
    self.target_out_splitter = self.target_out_splitter:cuda()

    self:allocate_memory()
end

function LeadModel:float()
    self.tgt_vocab_in = self.tgt_vocab_in:float()  
    self.tgt_vocab_out = self.tgt_vocab_out:float()  
    self.tgt_vocab_in_grad = self.tgt_vocab_in_grad:float()  
    self.tgt_vocab_out_grad = self.tgt_vocab_out_grad:float()  
    self.backbone_encoder = self.backbone_encoder:float()
    self.support_encoder = self.backbone_encoder:float()
    self.target_input_embeddings = self.target_input_embeddings:float()
    self.target_output_embeddings = self.target_output_embeddings:float()
    self.target_input_selector = self.target_input_selector:float()
    self.target_selector_bridge = self.target_selector_bridge:float()
    self.predict = self.predict:float()
    self.lsm = self.lsm:float()
    self.criterion = self.criterion:float()
    self.target_out_splitter = self.target_out_splitter:float()

    self:allocate_memory()
end




function LeadModel:allocate_memory()

    local function appendTable(baseTable, otherTable)
        for i=1,#otherTable do table.insert(baseTable, otherTable[i]) end
    end

    local params = {}
    local grad_params = {}
    local bb_params, bb_grad_params = self.backbone_encoder:parameters()
    local sp_params, sp_grad_params = self.support_encoder:parameters()
    local pred_params, pred_grad_params = self.predict:parameters()

    appendTable(params, bb_params)
    appendTable(params, sp_params)
    appendTable(grad_params, bb_grad_params)
    appendTable(grad_params, sp_grad_params)
    table.insert(params, self.tgt_vocab_in)
    table.insert(params, self.tgt_vocab_out)
    table.insert(grad_params, self.tgt_vocab_in_grad)
    table.insert(grad_params, self.tgt_vocab_out_grad)
    appendTable(params, pred_params)
    appendTable(grad_params, pred_grad_params)

    self.params = nn.Module.flatten(params)
    self.grad_params = nn.Module.flatten(grad_params)

end

function LeadModel:zero_grad_parameters()
    self.grad_params:zero()
    self.target_input_embeddings:zeroGradParameters()
    self.target_output_embeddings:zeroGradParameters()
    self.target_input_selector:zeroGradParameters()
    self.target_selector_bridge:zeroGradParameters()
    self.lsm:zeroGradParameters()
    self.target_out_splitter:zeroGradParameters()
end

function LeadModel:forget()
    self.target_input_embeddings:forget()
    self.target_output_embeddings:forget()
    self.target_input_selector:forget()
    self.target_selector_bridge:forget()
    self.lsm:forget()
    self.target_out_splitter:forget()
    self.backbone_encoder:forget()
    self.support_encoder:forget()
    self.predict:forget()
end

function LeadModel:build_network()

    self.tgt_vocab_in = torch.rand(self.out_vsize, self.dim_size):float()
    self.tgt_vocab_out = torch.rand(self.out_vsize, self.dim_size):float()
    self.tgt_vocab_in_grad = torch.FloatTensor():resizeAs(self.tgt_vocab_in)
    self.tgt_vocab_out_grad = torch.FloatTensor():resizeAs(self.tgt_vocab_out)

    -- Input lstm encoders. Lookup table is shared for both the backbone and 
    -- supporting sentences.

    local lu1 = nn.LookupTableMaskZero(self.in_vsize, self.dim_size)
    local lu2 = lu1:clone("weight", "gradWeight")

    local bb_encoder = nn.Sequential()
    bb_encoder:add(nn.Transpose({2,1}))
    bb_encoder:add(lu1)
    local lstm = nn.SeqBRNN(self.dim_size, self.dim_size)
    lstm.forwardModule:maskZero(1)
    lstm.backwardModule:maskZero(1)
    bb_encoder:add(lstm)
    self.backbone_encoder = bb_encoder
    self.backbone_encoder:float()

    local sp_encoder = nn.Sequential()
    sp_encoder:add(nn.Transpose({2,1}))
    sp_encoder:add(lu2)
    local lstm = nn.SeqBRNN(self.dim_size, self.dim_size)
    lstm.forwardModule:maskZero(1)
    lstm.backwardModule:maskZero(1)
    sp_encoder:add(lstm)
    self.support_encoder = sp_encoder
    self.support_encoder:float()

    -- Setup up target input vocab, and join with encoded backbone and support.
    
    local tgt_in_emb = nn.Sequential()
    local tgt_in_emb_conv = nn.ParallelTable()
    tgt_in_emb_conv:add(nn.Identity())
    tgt_in_emb_conv:add(nn.Replicate(1))
    tgt_in_emb_conv:add(nn.Transpose({2,1}))
    tgt_in_emb_conv:add(nn.Transpose({2,1}))
    tgt_in_emb:add(tgt_in_emb_conv)
    tgt_in_emb:add(nn.JoinTable(2))
    tgt_in_emb:add(nn.SplitTable(1))

    self.target_input_embeddings = tgt_in_emb
    self.target_input_embeddings:float()

    -- Setup target output vocab and join with encoded backbone and support.
    
    local tgt_out_emb = nn.Sequential()
    local tgt_out_emb_conv = nn.ParallelTable()
    tgt_out_emb_conv:add(
        nn.Sequential():add(nn.Replicate(1)):add(nn.Transpose({3,2})))
    tgt_out_emb_conv:add(nn.Transpose({2,1}, {3,2}))
    tgt_out_emb_conv:add(nn.Transpose({2,1}, {3,2}))
    tgt_out_emb:add(tgt_out_emb_conv)
    tgt_out_emb:add(nn.JoinTable(3))

    self.target_output_embeddings = tgt_out_emb
    self.target_output_embeddings:float()

    -- Set up target input lookup table layer
    self.target_lookup_tables = {} 
    local tgt_in_selector = nn.Sequential()
    tgt_in_selector:add(nn.SplitTable(1))
    tgt_in_selector:add(nn.ParallelTable())
    --tgt_in_selector:add(nn.MapTable(nn.Unsqueeze(1)))
    --tgt_in_selector:add(nn.MapTable(nn.Transpose({2,1})))
    --tgt_in_selector:add(nn.JoinTable(1))
    --tgt_in_selector:add(nn.JoinTable(1))
    
    self.target_input_selector = nn.Recursor(tgt_in_selector)
    self.target_input_selector:float()

    self.target_selector_bridge = nn.Recursor(nn.JoinTable(1))
    self.target_selector_bridge:float()

    -- Setup target lstm and logits layer
    local dec_lstm = nn.FastLSTM(self.dim_size, self.dim_size)
    dec_lstm:maskZero(1)
    local dec_lstm3D = nn.Sequential():add(dec_lstm):add(nn.Unsqueeze(2))
    self.predict = nn.Recursor(
        nn.MaskZero(
            nn.Sequential():add(
                nn.ParallelTable():add(dec_lstm3D):add(nn.Identity())
            ):add(nn.MM()):add(nn.Squeeze(2))
            , 1))
    self.predict:float()

    self.lsm = nn.Recursor(nn.LogSoftMaxMaskZero())
    self.lsm:float()

    local nll = nn.ClassNLLCriterion()
    nll.sizeAverage = false
    self.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nll, 1))
    self.criterion:float()
 
    self.target_out_splitter = nn.SplitTable(1)
    self.target_out_splitter:float()
end

function LeadModel:set_replicator_batch_size(batch_size)
    self.target_input_embeddings:get(1):get(2).nfeatures = batch_size
    self.target_output_embeddings:get(1):get(1):get(1).nfeatures = batch_size
end

function LeadModel:set_target_lookups(lookup_embeddings)

    
    --local module = self.target_input_selector.recurrentModule
    --local par_table = module:get(2)
    
    local module = nn.ParallelTable()
    module:type(lookup_embeddings[1]:type())
    
--    print(module)
--    print(par_table)
--    os.exit()
--    self.target_input_selector.modules = {par_table}
--    self.target_input_selector.sharedClones = {}
--    self.target_input_selector.sharedClones[1] = par_table

    --par_table.modules = {}
    self.target_lookups_grads = {}

    local batch_size = #lookup_embeddings

    for batch=1,batch_size do
        lt = self.target_lookup_tables[batch]
        if lt == nil then
            lt = nn.LookupTableMaskZero(1, self.dim_size)
            self.target_lookup_tables[batch] = lt
        end
        emb_batch = lookup_embeddings[batch]
        lt:type(emb_batch:type())
        lt.weight = emb_batch
        lt.gradWeight:resizeAs(emb_batch):zero()
        module:add(lt)
        --table.insert(par_table.modules, lt)    
        table.insert(self.target_lookups_grads, lt.gradWeight)
    end

    self.target_input_selector = nn.Recursor(module)
    self.target_input_selector:type(lookup_embeddings[1]:type())
end

function LeadModel:loss(bb, sp, tgt_in, tgt_out)

    self:zero_grad_parameters()
    self:forget()

    local pred_ll = self:forward(bb, sp, tgt_in)
    local gold_out = self.target_out_splitter:forward(tgt_out:t())
    
    local nll = self.criterion:forward(pred_ll, gold_out)
    return nll
end

function LeadModel:train_step(bb, sp, tgt_in, tgt_out)
    
    local function feval(params)

        self:zero_grad_parameters()
        self:forget()

        local pred_ll = self:forward(bb, sp, tgt_in)
        local gold_out = self.target_out_splitter:forward(tgt_out:t())
        
        local nll = self.criterion:forward(pred_ll, gold_out)
        local grad_loss = self.criterion:backward(pred_ll, gold_out)
        local nnz = tgt_in:nonzero():size(1)
        for b=1,bb:size(1) do
            grad_loss[b]:mul(1/nnz)
        end
        self:backward(bb, sp, tgt_in, grad_loss)

        self.grad_params:clamp(-5,5)

        return nll, self.grad_params
    end

    local _, nll = optim.adam(feval, self.params, self.optim_state)
    return nll[1]

end

function LeadModel:forward(bb, sp, tgt_in)

    local batch_size = tgt_in:size(1)
    local target_size = tgt_in:size(2)
    assert(bb:size(1) == batch_size)
    assert(sp:size(1) == batch_size)

    self:set_replicator_batch_size(batch_size)

    local backbone_layer = self.backbone_encoder:forward(bb)
    local support_layer = self.support_encoder:forward(sp)

    self.zero = self.zero or torch.Tensor()
    self.zero = self.zero:typeAs(self.tgt_vocab_in)
    self.zero = self.zero:resize(batch_size, 1, self.dim_size):zero()

    local tgt_in_emb = self.target_input_embeddings:forward{
        self.zero, self.tgt_vocab_in, backbone_layer, support_layer}

    local tgt_out_emb = self.target_output_embeddings:forward{
        self.tgt_vocab_out, backbone_layer, support_layer}

    self:set_target_lookups(tgt_in_emb)
    
    local max_steps = tgt_in:size(2)

    self.target_input_embedding_table_steps = {}
    self.target_input_embedding_steps = {}
    self.logits = {}
    self.log_softmax = {}
    for step=1,max_steps do

        local tgt_in_step = torch.split(tgt_in:select(2, step), 1, 1)

        local tgt_in_emb_table_step = self.target_input_selector:forward(
            tgt_in_step)
        local tgt_in_emb_step = self.target_selector_bridge:forward(
            tgt_in_emb_table_step) 
        local logits_step = self.predict:forward{tgt_in_emb_step, tgt_out_emb}
        local lsm_step = self.lsm:forward(logits_step)
        
        self.target_input_embedding_table_steps[step] = tgt_in_emb_table_step
        self.target_input_embedding_steps[step] = tgt_in_emb_step
        self.logits[step] = logits_step
        self.log_softmax[step] = lsm_step
    end

    return self.log_softmax
end

function LeadModel:backward(bb, sp, tgt_in, grad_loss)

    local batch_size = tgt_in:size(1)

    local tgt_out_emb = self.target_output_embeddings.output
    local tgt_in_layer = self.target_input_selector.output

    local backbone_layer = self.backbone_encoder.output
    local support_layer = self.support_encoder.output

    local max_steps = tgt_in:size(2)

    self.grad_tgt_output = self.grad_tgt_output or torch.Tensor()
    self.grad_tgt_output = self.grad_tgt_output:typeAs(tgt_out_emb)
    self.grad_tgt_output = self.grad_tgt_output:resizeAs(tgt_out_emb):zero()

    local grad_tgt_selector = {}
    for step=max_steps,1,-1 do

        local tgt_in_step = torch.split(tgt_in:select(2, step), 1, 1)
        local logits_step = self.logits[step]
        local tgt_in_emb_step = self.target_input_embedding_steps[step]
        local tgt_in_emb_table_step = 
            self.target_input_embedding_table_steps[step]
        
        local grad_logits_step = self.lsm:backward(
            logits_step, grad_loss[step])
        local grad_predict_step = self.predict:backward(
            {tgt_in_emb_step, tgt_out_emb}, grad_logits_step)
        local grad_bridge_step = self.target_selector_bridge:backward(
            tgt_in_emb_table_step, grad_predict_step[1]) 
            
        self.target_input_selector:backward(
            tgt_in_step, grad_bridge_step)
        self.grad_tgt_output:add(grad_predict_step[2])
    end
    
    local grad_output_embeddings = self.target_output_embeddings:backward(
        {self.tgt_vocab_out, backbone_layer, support_layer}, 
        self.grad_tgt_output)

    self.tgt_vocab_out_grad:add(grad_output_embeddings[1])
    local grad_bb = grad_output_embeddings[2]
    local grad_sp = grad_output_embeddings[3]

    local grad_input_embeddings = self.target_input_embeddings:backward(
        {self.zero, self.tgt_vocab_in, backbone_layer, support_layer},
        self.target_lookups_grads)

    self.tgt_vocab_in_grad:add(grad_input_embeddings[2])
    grad_bb:add(grad_input_embeddings[3])
    grad_sp:add(grad_input_embeddings[4])

    self.backbone_encoder:backward(bb, grad_bb)
    self.support_encoder:forward(sp, grad_sp)

 
end

function LeadModel:greedy_predict(bb, sp)

    self:zero_grad_parameters()
    self:forget()
    self.log_softmax = {}

    local batch_size = bb:size(1)
    assert(sp:size(1) == batch_size)

    self:set_replicator_batch_size(batch_size)

    local backbone_layer = self.backbone_encoder:forward(bb)
    local support_layer = self.support_encoder:forward(sp)

    self.zero = self.zero or torch.Tensor()
    self.zero = self.zero:typeAs(self.tgt_vocab_in)
    self.zero = self.zero:resize(batch_size, 1, self.dim_size):zero()

    local tgt_in_emb = self.target_input_embeddings:forward{
        self.zero, self.tgt_vocab_in, backbone_layer, support_layer}

    local tgt_out_emb = self.target_output_embeddings:forward{
        self.tgt_vocab_out, backbone_layer, support_layer}

    self:set_target_lookups(tgt_in_emb)

    self.target_input_embedding_table_steps = {}
    self.target_input_embedding_steps = {}
    self.logits = {}
    self.log_softmax = {}

    local max_steps = 75 --tgt_in:size(2)

    self.tgt_in_step = self.tgt_in_step or torch.Tensor()
    self.tgt_in_step = self.tgt_in_step:typeAs(bb):resize(batch_size):fill(2)
    self.tgt_out_step = self.tgt_out_step or torch.LongTensor()
    if bb:type() == "torch.CudaTensor" then 
        self.tgt_out_step = self.tgt_out_step:type("torch.CudaLongTensor")
    end
    self.tgt_out_step = self.tgt_out_step:resize(max_steps, batch_size)


    self.output_lp = self.output_lp or torch.Tensor()
    self.output_lp = self.output_lp:typeAs(bb):resize(batch_size, 1)

    local output_mask = torch.eq(tgt_out_emb:select(2,1), 0)
    local is_finished = torch.eq(self.tgt_in_step, 3)

    local last_step = 0
    for step=1,max_steps do
        last_step = step

        local tgt_in_step = torch.split(self.tgt_in_step, 1, 1)

        local tgt_in_emb_table_step = self.target_input_selector:forward(
            tgt_in_step)
        local tgt_in_emb_step = self.target_selector_bridge:forward(
            tgt_in_emb_table_step) 
        local logits_step = self.predict:forward{tgt_in_emb_step, tgt_out_emb}
        local lsm_step = self.lsm:forward(logits_step)
        
        self.target_input_embedding_table_steps[step] = tgt_in_emb_table_step
        self.target_input_embedding_steps[step] = tgt_in_emb_step
        self.logits[step] = logits_step
        self.log_softmax[step] = lsm_step
        
        lsm_step:maskedFill(output_mask, -math.huge)
        local max_val, max_ind = torch.max(
            self.output_lp, self.tgt_out_step[step], lsm_step, 2)
        
        max_ind:maskedFill(is_finished, 0)    
        is_finished:maskedFill(torch.eq(max_ind, 3), 1)
        self.tgt_in_step:copy(max_ind)

        if torch.all(is_finished) then break end

    end

    return self.tgt_out_step:t():narrow(2, 1, last_step)
--    return self.log_softmax
end

function LeadModel:greedy_predict_debug(bb, sp, tgt_in, tgt_out)
    local batch_size = bb:size(1)
    local pred_tgt = self:greedy_predict(bb, sp)

    local data = {}
    
    for batch=1,batch_size do
       
        local bb_start = bb[batch]:nonzero()[1][1] + self.out_vsize
        local bb_end = bb:size(2) + self.out_vsize
        local sp_start = bb_end + 1
        local sp_end = sp[batch]:nonzero()[-1][1] + bb_end

        local datum = {plates={}, output=nil}
        local backbone_pred_plate = {
            name="backbone-prediction-layer",
            vocab="input",
            steps={}
        }
        local support_pred_plate = {
            name="support-prediction-layer",
            vocab="input",
            steps={}
        }
        local vocab_pred_plate = {
            name="vocab-prediction-layer",
            vocab="output",
            steps={}
        }

        local outputs_info = {
            predicted_steps={},
            gold_steps={}
        }
        datum["output"] = outputs_info

        for step=1,tgt_out:size(2) do

            local gold_index = tgt_out[batch][step]
            
            if gold_index == 0 then 
                break 
            end
            
            local gold_ll = -math.huge;
            if step <= #self.log_softmax then
                gold_ll = self.log_softmax[step][batch][gold_index]
            end

            if gold_index <= self.out_vsize then
                t = {
                    id=gold_index, 
                    source="vocab",
                    ll=gold_ll
                }
                table.insert(outputs_info.gold_steps, t)
            elseif gold_index <= bb_end then
                t = {
                    id=gold_index - bb_start + 1, 
                    source="backbone-prediction-layer",
                    ll=gold_ll
                }
                table.insert(outputs_info.gold_steps, t)
            else
                t = {
                    id=gold_index - sp_start + 1, 
                    source="support-prediction-layer",
                    ll=gold_ll
                }
                table.insert(outputs_info.gold_steps, t)
            end
        end

        for step=1,pred_tgt:size(2) do
            --local bb_start = 
            local lsm_step = self.log_softmax[step][batch]
           -- print(lsm_step:view(1, lsm_step:size(1)))

            local pred_index = pred_tgt[batch][step]
            local pred_ll = self.log_softmax[step][batch][pred_index]

            if pred_index <= self.out_vsize then
                t = {
                    id=pred_index, 
                    source="vocab",
                    ll=pred_ll
                }
                table.insert(outputs_info.predicted_steps, t)
            elseif pred_index <= bb_end then
                t = {
                    id=pred_index - bb_start + 1, 
                    source="backbone-prediction-layer",
                    ll=pred_ll
                }
                table.insert(outputs_info.predicted_steps, t)
            else
                t = {
                    id=pred_index - sp_start + 1, 
                    source="support-prediction-layer",
                    ll=pred_ll
                }
                table.insert(outputs_info.predicted_steps, t)
            end
        
            local bb_lsm_table = {}
            for a=bb_start,bb_end do
                local t = {
                    id=bb[batch][a - self.out_vsize],
                    ll=lsm_step[a]
                }
                table.insert(bb_lsm_table, t)
            end
            table.insert(backbone_pred_plate.steps, bb_lsm_table)

            local sp_lsm_table = {}
            for a=sp_start,sp_end do
                local t = {
                    id=sp[batch][a - bb_end],
                    ll=lsm_step[a]
                }
                table.insert(sp_lsm_table, t)
            end
            table.insert(support_pred_plate.steps, sp_lsm_table)

            local voc_k = 10
            local lsm_step_v = lsm_step:narrow(1, 1, self.out_vsize)
            local v_lls, v_idxs = torch.topk(lsm_step_v, voc_k, 1, true, true)
            v_lsm_table = {}
            for a=1,voc_k do
                local t = {id=v_idxs[a], ll=v_lls[a]}
                table.insert(v_lsm_table, t)
            end
            table.insert(vocab_pred_plate.steps, v_lsm_table)

            if pred_tgt[batch][step] == 3 then break end
        end
        table.insert(datum.plates, backbone_pred_plate)
        table.insert(datum.plates, support_pred_plate)
        table.insert(datum.plates, vocab_pred_plate)
        table.insert(data, datum)
    end

    return data

end

