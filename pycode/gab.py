def langevin_sample(negative_query, img, obs, model, lr, momentum, iteration, decay_step, decay_ratio, device='cuda', verbose=False, sort=False):
    
    for param in model.enc.parameters():
        param.requires_grad = False
    
    for param in model.dec.parameters():
        param.requires_grad = False
    
    for param in model.obs_encoder.parameters():
        param.requires_grad = False

    for key in negative_query:
        negative_query[key] = negative_query[key].to(device)

        if key in ["time"]:
            continue
        negative_query[key].requires_grad = True
        
    query_optimizer = torch.optim.SGD(negative_query.values(), lr=lr, momentum=momentum)
    query_scheduler = torch.optim.lr_scheduler.StepLR(query_optimizer, step_size=decay_step, gamma=decay_ratio)

    if iteration > 0:
        for i in range(iteration):
            query_optimizer.zero_grad()

            if i == 0:
                energy, _ = model(img, negative_query, obs)
            else:
                energy, _ = model(img, negative_query, obs, with_feature=True)

            mean_energy = torch.mean(energy['all'])
            mean_energy.backward()

            if verbose:
                print(f"iteration: {i}")
                print("energy")
                print(energy['all'][:,:5])
                print("min energy")
                print(torch.mean(torch.min(energy["all"], 1)[0]))
                print("mean energy")
                print(torch.mean(energy["all"]))

            query_optimizer.step()
            query_scheduler.step()

            lr = query_optimizer.param_groups[0]['lr']
            var = 2 * lr # see https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm

    #         if i != iteration - 1:
    #             with torch.no_grad():
    #                 for key in negative_query.keys():
    #                     if key in ["time", "grasp"]:
    #                         continue
    #                     negative_query[key] += torch.normal(0, var, size=negative_query[key].shape).to(device)

            with torch.no_grad():
                negative_query = cliping(negative_query)
    
    for key in negative_query.keys():
        negative_query[key] = negative_query[key].detach()
    
    for param in model.enc.parameters():
        param.requires_grad = True
    
    for param in model.dec.parameters():
        param.requires_grad = True
        
    for param in model.obs_encoder.parameters():
        param.requires_grad = True

    with torch.no_grad():
        if iteration:
            energy, _ = model(img, negative_query, obs, with_feature=True)
        else:
            energy, _ = model(img, negative_query, obs, with_feature=False)
            
    if sort:
        _, indices = torch.sort(energy['all'], 1)
        for key in energy.keys():
            energy[key] = torch.gather(energy[key], 1, indices)
        
        for key in negative_query.keys():
            new_indices = repeat(indices, 'B N -> B N D', D=negative_query[key].shape[-1])
            negative_query[key] = torch.gather(negative_query[key], 1, new_indices).cpu()

    if verbose:
        print("result")
        print("energy")
        print(energy['all'][:,:5])
        print("min energy")
        print(torch.mean(torch.min(energy["all"], 1)[0]))
        print("mean energy")
        print(torch.mean(energy["all"]))
    
    return negative_query

def DFO_sample(sample, img, obs, model, info_dict, step_size=0.1, iteration=100, decay_step=10, decay_ratio=0.5, device='cuda', verbose=False, sort=False):
    
    m = torch.nn.Softmax(dim=1)
    B, N, _ = sample['uv'].shape
    max_z, min_z, max_uv, min_uv, max_rot, min_rot = info_dict["max_z_frame"], info_dict["min_z_frame"], info_dict["max_uv_frame"], info_dict["min_uv_frame"], info_dict["max_rot_frame"], info_dict["min_rot_frame"]

    for key in sample.keys():
        sample[key] = sample[key].to(device)
        
    if iteration > 0:
        for i in range(1, iteration+1):
            
            with torch.no_grad():
                if i == 1:
                    energy, _ = model(img, sample, obs)
                else:
                    energy, _ = model(img, sample, obs, with_feature=True)
                
                prob = m(-energy["all"])
                index = torch.multinomial(prob, N, replacement=True)

                for key in sample.keys():
                    new_indices = repeat(index, 'B N -> B N D', D=sample[key].shape[-1])
                    sample[key] = torch.gather(sample[key], 1, new_indices).to(device)
                    
                # get z
                z_range = (max_z - min_z) / 2
                z_sigma = step_size * z_range
                z_normal = torch.normal(0, z_sigma, size=(B, N, 1), device=device)
                sample['z'] = sample['z'] + z_normal
                
                # get uv
                max_u, max_v = max_uv
                min_u, min_v = min_uv
                u_range, v_range = (max_u - min_u) / 2, (max_v - min_v) / 2
                u_sigma, v_sigma = u_range * step_size, v_range * step_size
                u_normal = torch.normal(mean=0., std=u_sigma, size=(B,N,1), device=device)
                v_normal = torch.normal(mean=0., std=v_sigma, size=(B,N,1), device=device)
                uv_normal = torch.cat([u_normal, v_normal], 2)
                sample['uv'] = sample['uv'] + uv_normal
                
                # get rotation
                rotation_range = torch.tensor(max_rot, dtype=torch.float, device=device) - torch.tensor(min_rot, dtype=torch.float, device=device) / 2
                rotation_sigma = rotation_range * step_size

                rotz_normal = torch.normal(mean=0., std=rotation_sigma[0], size=(B,N,1), device=device)
                roty_normal = torch.normal(mean=0., std=rotation_sigma[1], size=(B,N,1), device=device)
                rotx_normal = torch.normal(mean=0., std=rotation_sigma[2], size=(B,N,1), device=device)
                rot_normal = torch.cat([rotz_normal, roty_normal, rotx_normal], 2)
                
                base_rot_flat = rearrange(sample['rotation_quat'], 'B N D -> (B N) D')
                rot_normal = rearrange(rot_normal, 'B N D -> (B N) D')
                rot_normal = rot_normal.cpu().numpy()
                r1 = R.from_euler('zyx', rot_normal, degrees=True)
                r2 = R.from_quat(base_rot_flat.cpu().numpy())
                r3 = r1 * r2

                new_rotation = torch.tensor(r3.as_quat(), dtype=torch.float, device=device)
                sample['rotation_quat'] = rearrange(new_rotation, '(B N) D -> B N D', B=B)

            if verbose:
                print(f"iteration: {i}")
                print("energy")
                print(energy['all'][:,:5])
                print("min energy")
                print(torch.mean(torch.min(energy["all"], 1)[0]))
                print("mean energy")
                print(torch.mean(energy["all"]))

            with torch.no_grad():
                sample = cliping(sample)
                
            if iteration % decay_step == 0:
                step_size *= decay_ratio
    
    with torch.no_grad():
        if iteration:
            energy, _ = model(img, sample, obs, with_feature=True)
        else:
            energy, _ = model(img, sample, obs, with_feature=False)
    
    if sort:
        _, indices = torch.sort(energy['all'], 1)
        for key in energy.keys():
            energy[key] = torch.gather(energy[key], 1, indices)
        
        for key in sample.keys():
            new_indices = repeat(indices, 'B N -> B N D', D=sample[key].shape[-1])
            sample[key] = torch.gather(sample[key], 1, new_indices).cpu()
            
    if verbose:
        print("result")
        print("energy")
        print(energy['all'][:,:5])
        print("min energy")
        print(torch.mean(torch.min(energy["all"], 1)[0]))
        print("mean energy")
        print(torch.mean(energy["all"]))
    
    return sample

def random_negative_sample_RLBench(cfg, info_dict, device='cuda', batch_size=0):
    """
    pos_query:
     - uv
     - rotation_quat
     - grasp
     - z
     - time
    """
    max_z, min_z = info_dict["max_z_all"], info_dict["min_z_all"]
    num_query = cfg.SAMPLING.NUM_NEGATIVE
    negative_query = {}

    if batch_size != 0:
        sample_shape = [batch_size, num_query, 1]
        num_sample = cfg.SAMPLING.NUM_NEGATIVE * batch_size
    else:
        sample_shape = [num_query, 1]
        num_sample = cfg.SAMPLING.NUM_NEGATIVE

    negative_u = (torch.rand(sample_shape[:-1], device=device) * 2) - 1
    negative_v = (torch.rand(sample_shape[:-1], device=device) * 2) - 1
    negative_uv = torch.stack([negative_u, negative_v], dim=len(sample_shape)-1)
    # negative_uv = rearrange(negative_uv, '(B N) P -> B N P', B=B)
    negative_query["uv"] = negative_uv
    
    # get rotation_quat
    negative_rotation = R.random(num_sample).as_quat()
    negative_rotation = torch.tensor(negative_rotation, dtype=torch.float, device=device)
    if batch_size != 0:
        negative_rotation = rearrange(negative_rotation, '(B N) P -> B N P', B=batch_size)
    negative_query["rotation_quat"] = negative_rotation
    
    # get grasp
    negative_grasp = torch.randint(0, 2, sample_shape, dtype=torch.float, device=device)
    negative_query["grasp"] = negative_grasp
    
    # get z
    if max_z == None:
        negative_z = torch.rand(sample_shape, device=device) + 1
    else:
        negative_z = torch.rand(sample_shape, device=device) * (max_z - min_z) + min_z
    negative_query["z"] = negative_z
    
    # # get time
    # if cfg.SAMPLING.TIME_LIST == [0]:
    #     if max_time == None:
    #         negative_time = torch.randint(1, 100, (num_sample,), dtype=torch.float16, device=device)
    #     else:
    #         negative_time = torch.randint(1, max_time, (num_sample,), dtype=torch.float16, device=device)
    # else:
    #     time_list = torch.tensor(cfg.SAMPLING.TIME_LIST, device=device)
    #     p = torch.ones(len(time_list), device=device) / len(time_list)
    #     index = p.multinomial(num_samples=num_sample, replacement=True)
    #     negative_time = time_list[index]
    
    # if batch_size != 0:
    #     negative_time = rearrange(negative_time, '(B N) -> B N', B=batch_size)
    
    # negative_time = torch.unsqueeze(negative_time, len(sample_shape)-1) / 100
    # negative_query["time"] = negative_time.to(device)
    
    return negative_query