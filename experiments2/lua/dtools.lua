dtools = {}

function dtools.read_vocab(path)
    local id2vocab = {}
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
    local backbone_tokens = {}
    local support_tokens = {}
    local doc_ids = {}

    for line in io.lines(path) do
        local backbone, support, target, bb_tokens, sp_tokens, doc_id = table.unpack(
            stringx.split(line, " | "))
        local backbone_tensor = torch.FloatTensor(stringx.split(backbone, " "))
        local targets_tensor = torch.FloatTensor(stringx.split(target, " "))
        local support_tensor = torch.FloatTensor(stringx.split(support, " "))

        -- data is originally 0 indexed. Oh torch/lua and your choices.
        
        backbone_tensor:add(1) 
        targets_tensor:add(1)
        support_tensor:add(1)
        
        if backbone_tensor:size(1) <= 100 and support_tensor:size(1) <= 300 and targets_tensor:size(1) <= 50 then
        table.insert(
            targets_in,
            targets_tensor:narrow(1, 1, targets_tensor:size(1) - 1))
        
        table.insert(
            targets_out,
            targets_tensor:narrow(1, 2, targets_tensor:size(1) - 1))

        table.insert(backbones, backbone_tensor)
        table.insert(supports, support_tensor)
        table.insert(backbone_tokens, stringx.split(bb_tokens, " "))
        table.insert(support_tokens, stringx.split(sp_tokens, " "))
        table.insert(doc_ids, doc_id)
        end
    end

    local bb_data, bb_sizes = dtools.pad(backbones, 0, "left", true)
    local tgt_in, tgt_sizes = dtools.pad(targets_in, 0, "right", true)
    local tgt_out = dtools.pad(targets_out, 0, "right", false)
    local sp_data, sp_sizes = dtools.pad(supports, 0, "right", true)
    local dataset = {bb_data, bb_sizes, 
                     sp_data, sp_sizes, 
                     tgt_in, tgt_out, tgt_sizes, 
                     backbone_tokens, support_tokens,
                     doc_ids}
    return dataset

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

    local sizes = torch.Tensor():typeAs(data[1]):resize(#data):zero()
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


function dtools:__init_bufffers(use_gpu)

    self.perm_buffer = self.perm_buffer or torch.LongTensor()
    
    self.bb_buffer = self.bb_buffer or torch.FloatTensor()
    local bb_buf = self.bb_buffer
    if use_gpu then
        self.bb_buf_cuda = self.bb_buf_cuda or torch.CudaTensor()
        self.backbone_buffer_safe = self.bb_buf_cuda
    end

    self.bb_sz_unsort_buffer = self.bb_sz_unsort_buffer or torch.FloatTensor()

    self.bb_sz_sort_buffer = self.bb_sz_sort_buffer or torch.FloatTensor()
    if use_gpu then
        self.bb_sizes_buf_cuda = self.bb_sizes_buf_cuda or torch.CudaTensor()
        self.backbone_sizes_buffer_safe = self.bb_sizes_buf_cuda
    end

    self.sp_buffer = self.sp_buffer or torch.FloatTensor()
    if use_gpu then
        self.sp_buf_cuda = self.sp_buf_cuda or torch.CudaTensor()
        self.support_buffer_safe = self.sp_buf_cuda
    end

    self.sp_sz_buffer = self.sp_sz_buffer or torch.FloatTensor()
    if use_gpu then
        self.sp_sizes_buf_cuda = self.sp_sizes_buf_cuda or torch.CudaTensor()
        self.support_sizes_buffer_safe = self.sp_sizes_buf_cuda
    end

    self.tgt_in_buffer = self.tgt_in_buffer or torch.FloatTensor()
    if use_gpu then
        self.tgt_in_buf_cuda = self.tgt_in_buf_cuda or torch.CudaTensor()
        self.target_input_buffer_safe = self.tgt_in_buf_cuda
    end

    self.tgt_out_buffer = self.tgt_out_buffer or torch.FloatTensor()
    if use_gpu then
        self.tgt_out_buf_cuda = self.tgt_out_buf_cuda or torch.CudaTensor()
        self.target_output_buffer_safe = self.tgt_out_buf_cuda
    end

    self.tgt_sz_buffer = self.tgt_sz_buffer or torch.FloatTensor()
    if use_gpu then
        self.tgt_sizes_buf_cuda = self.tgt_sizes_buf_cuda or torch.CudaTensor()
        self.target_sizes_buffer_safe = self.tgt_sizes_buf_cuda
    end

    self.indices_buffer = self.indices_buffer or torch.LongTensor()

end

function dtools:iterbatch(data, use_gpu, batch_size, buffer_size, output_vsize)
    local bb = data[1]
    local bb_sizes = data[2]
    local sp = data[3]
    local sp_sizes = data[4]
    local tgt_in = data[5]
    local tgt_out = data[6]
    local tgt_sizes = data[7]
    
    local num_examples = bb:size(1)
    local bb_length = bb:size(2)
    local sp_length = sp:size(2)
    local tgt_in_length = tgt_in:size(2)
    local tgt_out_length = tgt_out:size(2)

    local max_buffer_size = buffer_size * batch_size
    
    self:__init_bufffers(use_gpu)

    local indices_rand = torch.randperm(self.perm_buffer, num_examples)

    local bb_buf = self.bb_buffer
    local backbone_buffer_safe = self.backbone_buffer_safe

    local bb_sz_us_buf = self.bb_sz_unsort_buffer

    local bb_sizes_buf = self.bb_sz_sort_buffer
    local backbone_sizes_buffer_safe = self.backbone_sizes_buffer_safe
    local sp_buf = self.sp_buffer
    local support_buffer_safe = self.support_buffer_safe

    local sp_sizes_buf = self.sp_sz_buffer
    local support_sizes_buffer_safe = self.support_sizes_buffer_safe

    local tgt_in_buf = self.tgt_in_buffer
    local target_input_buffer_safe = self.target_input_buffer_safe

    local tgt_out_buf = self.tgt_out_buffer
    local target_output_buffer_safe = self.target_output_buffer_safe

    local tgt_sizes_buf = self.tgt_sz_buffer
    local target_sizes_buffer_safe = self.target_sizes_buffer_safe

    local idx_buf = self.indices_buffer

    local data_location = 1
    local buffer_location = max_buffer_size + 1
    local batch_num = 0
    local real_buffer_size = 0
    
    --{ Define helper functions }--

    local reset_buffer = function()
        local remaining_examples = num_examples - data_location + 1
        real_buffer_size = math.min(remaining_examples, max_buffer_size)
        local idx_us = indices_rand:narrow(1, data_location, real_buffer_size)
        
        bb_sz_us_buf:index(bb_sizes, 1, idx_us)
        torch.sort(bb_sizes_buf, idx_buf, bb_sz_us_buf)

        idx_buf:index(idx_us, 1, idx_buf)        
        bb_buf:index(bb, 1, idx_buf)
        sp_buf:index(sp, 1, idx_buf)
        sp_sizes_buf:index(sp_sizes, 1, idx_buf)
        tgt_in_buf:index(tgt_in, 1, idx_buf)
        tgt_out_buf:index(tgt_out, 1, idx_buf)
        tgt_sizes_buf:index(tgt_sizes, 1, idx_buf)

        if use_gpu then
            backbone_buffer_safe:resize(
                bb_buf:size()):copy(bb_buf)
            backbone_sizes_buffer_safe:resize(
                bb_sizes_buf:size()):copy(bb_sizes_buf)
            support_buffer_safe:resize(sp_buf:size()):copy(sp_buf)
            support_sizes_buffer_safe:resize(
                sp_sizes_buf:size()):copy(sp_sizes_buf)
            target_input_buffer_safe:resize(tgt_in_buf:size()):copy(tgt_in_buf)
            target_output_buffer_safe:resize(tgt_out_buf:size()):copy(tgt_out_buf)
            target_sizes_buffer_safe:resize(
                tgt_sizes_buf:size()):copy(tgt_sizes_buf)
 
        else
            backbone_buffer_safe = bb_buf
            backbone_sizes_buffer_safe = bb_sizes_buf
            support_buffer_safe = sp_buf
            support_sizes_buffer_safe = sp_sizes_buf
            target_input_buffer_safe = tgt_in_buf
            target_output_buffer_safe = tgt_out_buf
            target_sizes_buffer_safe = tgt_sizes_buf
        end

        buffer_location = 1
        data_location = data_location + real_buffer_size
    end


    local iter = function() 
        batch_num = batch_num + 1

        if buffer_location > real_buffer_size then
            if data_location > num_examples then 
                return nil
            else 
                reset_buffer()
            end
        end

        local remaining_buffer_data = real_buffer_size - buffer_location + 1
        local real_batch_size = math.min(batch_size, remaining_buffer_data)

        local backbone_batch = backbone_buffer_safe:narrow(
            1, buffer_location, real_batch_size)
        local backbone_sizes_batch = backbone_sizes_buffer_safe:narrow(
            1, buffer_location, real_batch_size)

        local support_batch = support_buffer_safe:narrow(
            1, buffer_location, real_batch_size)
        local support_sizes_batch = support_sizes_buffer_safe:narrow(
            1, buffer_location, real_batch_size)

        local target_input_batch = target_input_buffer_safe:narrow(
            1, buffer_location, real_batch_size)
        local target_output_batch = target_output_buffer_safe:narrow(
            1, buffer_location, real_batch_size)
        local target_sizes_batch = 
            target_sizes_buffer_safe:narrow(
                1, buffer_location, real_batch_size)
        
        local indices_batch = idx_buf:narrow(
            1, buffer_location, real_batch_size)

        local backbone_max_size = torch.max(backbone_sizes_batch)
        local support_max_size = torch.max(support_sizes_batch)
        local target_max_size = torch.max(target_sizes_batch)

        backbone_batch = backbone_batch:narrow(
            2, bb_length - backbone_max_size + 1, backbone_max_size)
        support_batch = support_batch:narrow(2, 1, support_max_size)
        target_input_batch = target_input_batch:narrow(
            2, 1, target_max_size)
        target_output_batch = target_output_batch:narrow(
            2, 1, target_max_size)

        self:correct_offset_inplace(
            backbone_batch, backbone_sizes_batch, 
            target_input_batch, output_vsize)
        self:correct_offset_inplace(
            backbone_batch, backbone_sizes_batch,
            target_output_batch, output_vsize)

        buffer_location = buffer_location + batch_size

        local batch_data = {iter=batch_num,
                            backbone=backbone_batch,
                            support=support_batch,
                            target_input=target_input_batch,
                            target_output=target_output_batch,
                            indices=indices_batch,
                            target_nnz=target_sizes_batch:sum(),
                            target_sizes=target_sizes_batch,
                            backbone_sizes=backbone_sizes_batch,
                            support_sizes=support_sizes_batch
                           }
        return batch_data

    end

    return iter

end

function dtools:correct_offset_inplace(bb, bb_sizes, tgt, vsize)
    -- Adjust tgt copy indices to account for changes in the left padding
    -- of batch bb matrix. If its confusing, it is.

    local m = tgt:size(1)
    local n = tgt:size(2)

    self.offset_tensor = self.offset_tensor or torch.Tensor()
    self.offset_tensor = self.offset_tensor:typeAs(tgt):resizeAs(tgt)
    self.offset_tensor:fill(-1)
    self.offset_tensor:cmul(bb_sizes:view(m, 1):expand(m, n))
    self.offset_tensor:add(bb:size(2))
    local mask = torch.eq(torch.gt(tgt, vsize), 0)
    self.offset_tensor:maskedFill(mask, 0)
    tgt:add(self.offset_tensor)
    return tgt

end

function dtools.extract_tokens(backbone_tokens, support_tokens, offset, vocab, labels)

   local tokens = {}
   for i=1,labels:size(1) do
      if labels[i] > #vocab then
         local index = labels[i] - offset - #vocab + 1
         if index <= #backbone_tokens then
            table.insert(tokens, backbone_tokens[index])
         else
            table.insert(tokens, support_tokens[index - #backbone_tokens])
              
         end
      elseif labels[i] > 0 then
          table.insert(tokens, vocab[labels[i]])
      end
   end
   return tokens
end

function dtools:iterdoc(data, use_gpu, output_vsize)
    local bb = data[1]
    local bb_sizes = data[2]
    local sp = data[3]
    local sp_sizes = data[4]
    local tgt_in = data[5]
    local tgt_out = data[6]
    local tgt_sizes = data[7]
    local doc_ids = data[10]

    local num_examples = bb:size(1)
    local bb_length = bb:size(2)
    local sp_length = sp:size(2)
    local tgt_in_length = tgt_in:size(2)
    local tgt_out_length = tgt_out:size(2)

    --local max_buffer_size = buffer_size * batch_size
   
    local ids2range = {}
     
    local ids = {}

    for i=1,#doc_ids do
        if ids2range[doc_ids[i]] == nil then
            ids2range[doc_ids[i]] = {i, 0}
            table.insert(ids, doc_ids[i])
        end
        ids2range[doc_ids[i]][2] = ids2range[doc_ids[i]][2] + 1
    end

    self:__init_bufffers(use_gpu)

    --local indices_rand = torch.randperm(self.perm_buffer, num_examples)
    local indices = torch.range(self.perm_buffer, 1, #doc_ids, 1)

    local bb_buf = self.bb_buffer
    local backbone_buffer_safe = self.backbone_buffer_safe

    local bb_sz_us_buf = self.bb_sz_unsort_buffer

    local bb_sizes_buf = self.bb_sz_sort_buffer
    local backbone_sizes_buffer_safe = self.backbone_sizes_buffer_safe
    local sp_buf = self.sp_buffer
    local support_buffer_safe = self.support_buffer_safe

    local sp_sizes_buf = self.sp_sz_buffer
    local support_sizes_buffer_safe = self.support_sizes_buffer_safe

    local tgt_in_buf = self.tgt_in_buffer
    local target_input_buffer_safe = self.target_input_buffer_safe

    local tgt_out_buf = self.tgt_out_buffer
    local target_output_buffer_safe = self.target_output_buffer_safe

    local tgt_sizes_buf = self.tgt_sz_buffer
    local target_sizes_buffer_safe = self.target_sizes_buffer_safe

    local idx_buf = self.indices_buffer

    local batch_num = 0

    --{ Define helper functions }--

    local reset_buffer = function()
       
        local id = ids[batch_num]
        local start, size = table.unpack(ids2range[id])

        local index = indices:narrow(1, start, size)
        bb_buf:index(bb, 1, index)
        sp_buf:index(sp, 1, index)
        bb_sizes_buf:index(bb_sizes, 1, index)
        sp_sizes_buf:index(sp_sizes, 1, index)
        tgt_in_buf:index(tgt_in, 1, index)
        tgt_out_buf:index(tgt_out, 1, index)
        tgt_sizes_buf:index(tgt_sizes, 1, index)

        if use_gpu then
            backbone_buffer_safe:resize(
                bb_buf:size()):copy(bb_buf)
            backbone_sizes_buffer_safe:resize(
                bb_sizes_buf:size()):copy(bb_sizes_buf)
            support_buffer_safe:resize(sp_buf:size()):copy(sp_buf)
            support_sizes_buffer_safe:resize(
                sp_sizes_buf:size()):copy(sp_sizes_buf)
            target_input_buffer_safe:resize(tgt_in_buf:size()):copy(tgt_in_buf)
            target_output_buffer_safe:resize(tgt_out_buf:size()):copy(tgt_out_buf)
            target_sizes_buffer_safe:resize(
                tgt_sizes_buf:size()):copy(tgt_sizes_buf)
 
        else
            backbone_buffer_safe = bb_buf
            backbone_sizes_buffer_safe = bb_sizes_buf
            support_buffer_safe = sp_buf
            support_sizes_buffer_safe = sp_sizes_buf
            target_input_buffer_safe = tgt_in_buf
            target_output_buffer_safe = tgt_out_buf
            target_sizes_buffer_safe = tgt_sizes_buf
        end

    end


    local iter = function() 
        
        batch_num = batch_num + 1

        if batch_num > #ids then return nil end
        reset_buffer()

        local backbone_max_size = torch.max(backbone_sizes_buffer_safe)
        local support_max_size = torch.max(support_sizes_buffer_safe)
        local target_max_size = torch.max(target_sizes_buffer_safe)

        local backbone_batch = backbone_buffer_safe:narrow(
            2, bb_length - backbone_max_size + 1, backbone_max_size)
        local support_batch = support_buffer_safe:narrow(
            2, 1, support_max_size)
        local target_input_batch = target_input_buffer_safe:narrow(
            2, 1, target_max_size)
        local target_output_batch = target_output_buffer_safe:narrow(
            2, 1, target_max_size)

        self:correct_offset_inplace(
            backbone_batch, backbone_sizes_buffer_safe, 
            target_input_batch, output_vsize)
        self:correct_offset_inplace(
            backbone_batch, backbone_sizes_buffer_safe,
            target_output_batch, output_vsize)

        
        local id = ids[batch_num]
        local start, size = table.unpack(ids2range[id])
        local index = indices:narrow(1, start, size)
        local bb_tokens = {}
        local sp_tokens = {}
        for i=1,index:size(1) do
            table.insert(bb_tokens, data[8][index[i]])
            table.insert(sp_tokens, data[9][index[i]])
        end

        local batch_data = {iter=batch_num,
                            max_iter=#ids,
                            id=ids[batch_num],
                            backbone=backbone_batch,
                            support=support_batch,
                            target_input=target_input_batch,
                            target_output=target_output_batch,
                            target_nnz=target_sizes_buffer_safe:sum(),
                            target_sizes=target_sizes_buffer_safe,
                            backbone_sizes=backbone_sizes_buffer_safe,
                            support_sizes=support_sizes_buffer_safe,
                            backbone_tokens=bb_tokens,
                            support_tokens=sp_tokens
                           }
        return batch_data

    end

    return iter

end

return dtools
