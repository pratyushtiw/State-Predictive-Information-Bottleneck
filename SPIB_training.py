"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import torch
import numpy as np
import time
import os

# Data Processing
# ------------------------------------------------------------------------------

def data_init(t0, dt, traj_data, traj_label, traj_weights):
    assert len(traj_data)==len(traj_label)
    
    # skip the first t0 data
    past_data = traj_data[t0:(len(traj_data)-dt)]
    future_data = traj_data[(t0+dt):len(traj_data)]
    label = traj_label[(t0+dt):len(traj_data)]
    
    # data shape
    data_shape = past_data.shape[1:]
    
    n_data = len(past_data)
    
    # 90% random test/train split
    p = np.random.permutation(n_data)
    past_data = past_data[p]
    future_data = future_data[p]
    label = label[p]
    
    past_data_train = past_data[0: (9 * n_data) // 10]
    past_data_test = past_data[(9 * n_data) // 10:]
    
    future_data_train = future_data[0: (9 * n_data) // 10]
    future_data_test = future_data[(9 * n_data) // 10:]
    
    label_train = label[0: (9 * n_data) // 10]
    label_test = label[(9 * n_data) // 10:]
    
    if traj_weights != None:
        assert len(traj_data)==len(traj_weights)
        weights = traj_weights[t0:(len(traj_data)-dt)]
        weights = weights[p]
        weights_train = weights[0: (9 * n_data) // 10]
        weights_test = weights[(9 * n_data) // 10:]
    else:
        weights_train = None
        weights_test = None
    
    return data_shape, past_data_train, future_data_train, label_train, weights_train,\
        past_data_test, future_data_test, label_test, weights_test


# Loss function
# ------------------------------------------------------------------------------

def calculate_loss(IB, data_inputs, data_future, data_targets, data_weights, beta=1.0, beta1 = 0.0):
    
    # pass through VAE
    outputs, z_sample, z_mean, z_logvar = IB.forward(data_inputs)
    future_pred = IB.decode_future(z_mean)           # shape: (B, prod(data_shape))
    target_future = data_future.view(future_pred.size(0), -1) 
    
    # KL Divergence
    log_p = IB.log_p(z_sample)
    log_q = -0.5 * torch.sum(z_logvar + torch.pow(z_sample-z_mean, 2)
                             /torch.exp(z_logvar), dim=1)
    
    if data_weights == None:
        # Reconstruction loss is cross-entropy
        reconstruction_error = torch.mean(torch.sum(-data_targets*outputs, dim=1))
        
        # KL Divergence
        kl_loss = torch.mean(log_q-log_p)

        # average per feature then average over batch 
        per_sample_mse = torch.mean((future_pred - target_future) ** 2, dim=1)
        mse_loss = torch.mean(per_sample_mse)
        
    else:
        # Reconstruction loss is cross-entropy
        # reweighed
        reconstruction_error = torch.mean(data_weights*torch.sum(-data_targets*outputs, dim=1))
        
        # KL Divergence
        kl_loss = torch.mean(data_weights*(log_q-log_p))

        # Weighted MSE between inputs and future (per-sample then weighted)
        per_sample_mse = torch.mean((future_pred - target_future) ** 2, dim=1)
        mse_loss = torch.mean(data_weights * per_sample_mse)
        
    
    loss = reconstruction_error + beta * kl_loss + beta1 * mse_loss
    return loss, reconstruction_error.float(), kl_loss.float(), mse_loss.float()


# Train and test model
# ------------------------------------------------------------------------------

def sample_minibatch(past_data, future_data, data_labels, data_weights, indices, device):
    sample_past_data = past_data[indices].to(device)
    sample_future_data = future_data[indices].to(device)
    sample_data_labels = data_labels[indices].to(device)

    
    if data_weights == None:
        sample_data_weights = None
    else:
        sample_data_weights = data_weights[indices].to(device)
    
    
    return sample_past_data, sample_future_data, sample_data_labels, sample_data_weights

def set_requires_grad(module, flag: bool):
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad = flag

##
#training only future decoder - mse loss
##
def fine_tune_future_decoder(
    IB,
    train_past_data, train_future_data, train_data_weights,
    batch_size=512, lr=1e-3, epochs=5,
    patience=None, threshold=None,
    device="cpu", log_interval=200
):
    """
    Train ONLY the future_decoder to reconstruct future_data from z_mean.
    Encoder/logvar/classifier decoder remain frozen and are not updated.
    """
    # 1) Freeze everything except the future decoder
    set_requires_grad(getattr(IB, "encoder", None), False)
    set_requires_grad(getattr(IB, "encoder_logvar", None), False)
    set_requires_grad(getattr(IB, "decoder", None), False)
    set_requires_grad(getattr(IB, "representative_weights", None), False)
    set_requires_grad(getattr(IB, "future_decoder", None), True)

    # 2) Optimizer for ONLY future_decoder parameters
    fd = getattr(IB, "future_decoder", None)
    assert fd is not None, "IB.future_decoder not found — add it in SPIB.py first."
    optimizer = torch.optim.Adam(fd.parameters(), lr=lr)

    N = len(train_past_data)
    num_steps = (N + batch_size - 1) // batch_size
    best = float("inf")
    bad = 0

    IB.train()
    for epoch in range(epochs):
        perm = torch.randperm(N)
        sum_loss = 0.0
        denom = 0.0

        for s in range(num_steps):
            idx = perm[s*batch_size : (s+1)*batch_size]
            batch_inputs  = train_past_data[idx].to(device)
            batch_future  = train_future_data[idx].to(device)
            batch_weights = None if train_data_weights is None else train_data_weights[idx].to(device)

            # 3) Get z_mean WITHOUT building grads through encoder/etc.
            with torch.no_grad():
                # Prefer an encode() if you have it; else use forward(...) to pull z_mean
                try:
                    z_mean, z_logvar = IB.encode(batch_inputs.view(batch_inputs.size(0), -1))
                except:
                    outputs, z_sample, z_mean, z_logvar = IB.forward(batch_inputs)

            # 4) Train future_decoder on MSE to future data
            future_pred = IB.future_decoder(z_mean)               # (B, prod(data_shape))
            target = batch_future.view(future_pred.size(0), -1)

            per_sample = torch.mean((future_pred - target) ** 2, dim=1)
            if batch_weights is None:
                loss = torch.mean(per_sample)
                sum_loss += per_sample.detach().sum().item()
                denom += per_sample.numel()
            else:
                wsum = torch.sum(batch_weights).item() + 1e-12
                loss = torch.sum(batch_weights * per_sample) / wsum
                sum_loss += (batch_weights * per_sample).detach().sum().item()
                denom += wsum

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (s % log_interval) == 0:
                print(f"[FD] epoch {epoch} step {s}/{num_steps}  mse={loss.item():.6f}")

        avg_mse = sum_loss / max(denom, 1e-12)
        print(f"[FD] epoch {epoch} avg MSE: {avg_mse:.6f}")

        # 5) Optional early stop on MSE improvement
        if patience is not None and threshold is not None:
            if (best - avg_mse) > threshold:
                best = avg_mse
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"[FD] early stop: no improvement > {threshold} for {patience} epochs")
                    break


def train(IB, beta, beta1, train_past_data, train_future_data, init_train_data_labels, train_data_weights, \
          test_past_data, test_future_data, init_test_data_labels, test_data_weights, \
              learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, batch_size, threshold, patience, refinements, output_path, log_interval, device, index):
    IB.train()
    
    step = 0
    start = time.time()
    log_path = output_path + '_train.log'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    IB_path = output_path + "cpt" + str(index) + "/IB"
    os.makedirs(os.path.dirname(IB_path), exist_ok=True)
    
    train_data_labels = init_train_data_labels
    test_data_labels = init_test_data_labels

    update_times = 0
    unchanged_epochs = 0
    epoch = 0

    # initial state population
    state_population0 = torch.sum(train_data_labels,dim=0).float()/train_data_labels.shape[0]

    # generate the optimizer and scheduler
    # --- Phase 1: IB-only training; do NOT update future_decoder ---
    set_requires_grad(getattr(IB, "future_decoder", None), False)

    # Build optimizer from only trainable (requires_grad=True) params
    optimizer = torch.optim.Adam(
        [p for p in IB.parameters() if p.requires_grad],
        lr=learning_rate
    )


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

    while True:
        
        train_permutation = torch.randperm(len(train_past_data))
        test_permutation = torch.randperm(len(test_past_data))
        
        
        for i in range(0, len(train_past_data), batch_size):
            step += 1
            
            if i+batch_size>len(train_past_data):
                break
            
            train_indices = train_permutation[i:i+batch_size]
            
            batch_inputs, batch_future_data, batch_future_labels, batch_weights = sample_minibatch(train_past_data, train_future_data, train_data_labels, \
                                                                       train_data_weights, train_indices, device)
                    
            loss, reconstruction_error, kl_loss, mse_loss = calculate_loss(IB, batch_inputs, batch_future_data, \
                                                                batch_future_labels, batch_weights, beta, beta1=0.0)
            
            # Stop if NaN is obtained
            if(torch.isnan(loss).any()):
                return True
    
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if step % 500 == 0:
                with torch.no_grad():
                    
                    batch_inputs, batch_future_data, batch_future_labels, batch_weights = sample_minibatch(train_past_data, train_future_data, train_data_labels, \
                                                                               train_data_weights, train_indices, device)
                            
                    loss, reconstruction_error, kl_loss, mse_loss = calculate_loss(IB, batch_inputs, batch_future_data,\
                                                                        batch_future_labels, batch_weights, beta, beta1=0.0)
                    train_time = time.time() - start
            
                    print(
                        "Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f\tMSE (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error, mse_loss))
                    print(
                       "Iteration %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f\tMSE (train) %f" % (
                            step, train_time, loss, kl_loss, reconstruction_error, mse_loss), file=open(log_path, 'a'))
                    j=i%len(test_permutation)
                    
                    
                    
                    test_indices = test_permutation[j:j+batch_size]
                    
                    batch_inputs, batch_future_data, batch_future_labels, batch_weights = sample_minibatch(test_past_data, test_future_data, test_data_labels, \
                                                                               test_data_weights, test_indices, device)
                    
                    loss, reconstruction_error, kl_loss, mse_loss = calculate_loss(IB, batch_inputs, batch_future_data, \
                                                                         batch_future_labels, batch_weights, beta, beta1=0.0)

                    train_time = time.time() - start
                    print(
                       "Loss (test) %f\tKL loss (test): %f\n"
                       "Reconstruction loss (test) %f\tMSE (test) %f" % (
                           loss, kl_loss, reconstruction_error, mse_loss))
                    print(
                       "Loss (test) %f\tKL loss (test): %f\n"
                       "Reconstruction loss (test) %f\tMSE (test) %f" % (
                           loss, kl_loss, reconstruction_error, mse_loss), file=open(log_path, 'a'))
        
            if step % log_interval == 0:
                # save model
                torch.save({'step': step,
                            'state_dict': IB.state_dict()},
                           IB_path+ '_%d_cpt.pt'%step)
                torch.save({'optimizer': optimizer.state_dict()},
                           IB_path+ '_%d_optim_cpt.pt'%step) 

        epoch+=1
        
        # check convergence
        new_train_data_labels = IB.update_labels(train_future_data, batch_size)

        # save the state population
        state_population = torch.sum(new_train_data_labels,dim=0).float()/new_train_data_labels.shape[0]

        print(state_population)
        print(state_population, file=open(log_path, 'a'))

        # print the state population change
        state_population_change = torch.sqrt(torch.square(state_population-state_population0).sum())
        
        print('State population change=%f'%state_population_change)
        print('State population change=%f'%state_population_change, file=open(log_path, 'a'))

        # update state_population
        state_population0 = state_population

        scheduler.step()
        if scheduler.gamma < 1:
            print("Update lr to %f"%(optimizer.param_groups[0]['lr']))
            print("Update lr to %f"%(optimizer.param_groups[0]['lr']), file=open(log_path, 'a'))

        # check whether the change of the state population is smaller than the threshold
        if state_population_change < threshold:
            unchanged_epochs += 1
            
            if unchanged_epochs > patience:

                # check whether only one state is found
                if torch.sum(state_population>0)<2:
                    print("Only one metastable state is found!")
                    break

                # Stop only if update_times >= refinements
                if IB.UpdateLabel and update_times < refinements:
                    
                    train_data_labels = new_train_data_labels
                    test_data_labels = IB.update_labels(test_future_data, batch_size)
    
                    update_times+=1
                    print("Update %d\n"%(update_times))
                    print("Update %d\n"%(update_times), file=open(log_path, 'a'))
                    
                    # reset epoch and unchanged_epochs
                    epoch = 0
                    unchanged_epochs = 0

                    # reset the representative-inputs
                    representative_inputs = IB.estimatate_representative_inputs(train_past_data, train_data_weights, batch_size)
                    IB.reset_representative(representative_inputs.to(device))
    
                    # reset the optimizer and scheduler
                    optimizer = torch.optim.Adam(IB.parameters(), lr=learning_rate)

                    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
                    
                else:
                    break

        else:
            unchanged_epochs = 0

        print("Epoch: %d\n"%(epoch))
        print("Epoch: %d\n"%(epoch), file=open(log_path, 'a'))

    # --- Phase 2: fine-tune future decoder ONLY ---
    fine_tune_future_decoder(
        IB,
        train_past_data, train_future_data, train_data_weights,
        batch_size=batch_size,
        lr=learning_rate,            # or a separate fd_lr if you prefer
        epochs=5,                    # tune as needed
        patience=2, 
        threshold=None,  # optional; set None to disable
        device=device,
        log_interval=log_interval
        )
# output the saving path
    total_training_time = time.time() - start
    print("Total training time: %f" % total_training_time)
    print("Total training time: %f" % total_training_time, file=open(log_path, 'a'))
    # save model
    torch.save({'step': step,
                'state_dict': IB.state_dict()},
               IB_path+ '_%d_cpt.pt'%step)
    torch.save({'optimizer': optimizer.state_dict()},
               IB_path+ '_%d_optim_cpt.pt'%step)
    
    torch.save({'step': step,
                'state_dict': IB.state_dict()},
               IB_path+ '_final_cpt.pt')
    torch.save({'optimizer': optimizer.state_dict()},
               IB_path+ '_final_optim_cpt.pt')

    return False

@torch.no_grad()
def output_final_result(IB, device, train_past_data, train_future_data, train_data_labels, train_data_weights, \
                        test_past_data, test_future_data, test_data_labels, test_data_weights, batch_size, output_path, \
                            path, dt, beta, beta1, learning_rate, index=0):
    
    with torch.no_grad():
        final_result_path = output_path + '_final_result' + str(index) + '.npy'
        os.makedirs(os.path.dirname(final_result_path), exist_ok=True)
        
        # label update
        if IB.UpdateLabel:
            train_data_labels = IB.update_labels(train_future_data, batch_size)
            test_data_labels = IB.update_labels(test_future_data, batch_size)
        
        final_result = []
        # output the result
        
        loss, reconstruction_error, kl_loss, mse_loss = [0 for i in range(4)]
        
        for i in range(0, len(train_past_data), batch_size):
            batch_inputs, batch_future_data, batch_future_labels, batch_weights = sample_minibatch(train_past_data, train_future_data, train_data_labels, train_data_weights, \
                                                                       range(i,min(i+batch_size,len(train_past_data))), IB.device)
            loss1, reconstruction_error1, kl_loss1, mse_loss1 = calculate_loss(IB, batch_inputs, batch_future_data, batch_future_labels, \
                                                                    batch_weights, beta, beta1=0.0)
            loss += loss1*len(batch_inputs) 
            reconstruction_error += reconstruction_error1*len(batch_inputs)
            kl_loss += kl_loss1*len(batch_inputs)
            mse_loss += mse_loss1*len(batch_inputs)
            
        
        # output the result
        loss/=len(train_past_data)
        reconstruction_error/=len(train_past_data)
        kl_loss/=len(train_past_data)
        mse_loss/=len(train_past_data)
                
        final_result += [loss.data.cpu().numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy(), mse_loss.cpu().data.numpy()]
        print(
            "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
                    "Reconstruction loss (train) %f\tMSE (train) %f" % (
                index, loss, kl_loss, reconstruction_error, mse_loss))
        print(
            "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
                    "Reconstruction loss (train) %f\tMSE (train) %f" % (
                index, loss, kl_loss, reconstruction_error, mse_loss),
            file=open(path, 'a'))
    
        loss, reconstruction_error, kl_loss, mse_loss = [0 for i in range(4)]
        
        for i in range(0, len(test_past_data), batch_size):
            batch_inputs, batch_future_data, batch_future_labels, batch_weights = sample_minibatch(test_past_data, test_future_data, test_data_labels, test_data_weights, \
                                                                                         range(i,min(i+batch_size,len(test_past_data))), IB.device)
            loss1, reconstruction_error1, kl_loss1, mse_loss1 = calculate_loss(IB, batch_inputs, batch_future_data, batch_future_labels, \
                                                                   batch_weights, beta, beta1=0.0)
            loss += loss1*len(batch_inputs)
            reconstruction_error += reconstruction_error1*len(batch_inputs)
            kl_loss += kl_loss1*len(batch_inputs)
            mse_loss += mse_loss1*len(batch_inputs)
            
        
        # output the result
        loss/=len(test_past_data)
        reconstruction_error/=len(test_past_data)
        kl_loss/=len(test_past_data)
        mse_loss/=len(test_past_data)
        
        final_result += [loss.cpu().data.numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy(), mse_loss.cpu().data.numpy()]
        print(
            "Loss (test) %f\tKL loss (test): %f\n"
            "Reconstruction loss (test) %f\tMSE (test) %f"
            % (loss, kl_loss, reconstruction_error, mse_loss))
        print( 
            "Loss (test) %f\tKL loss (test): %f\n"
            "Reconstruction loss (test) %f\tMSE (test) %f"
            % (loss, kl_loss, reconstruction_error, mse_loss), file=open(path, 'a'))
        
        print("dt: %d\t Beta: %d\t Beta1: %f\t Learning_rate: %f" % (
            dt, beta, beta1, learning_rate))
        print("dt: %d\t Beta: %d\t Beta1: %f\t Learning_rate: %f" % (
            dt, beta, beta1, learning_rate),
              file=open(path, 'a'))    
        
        
        # Save future-data reconstructions for train/test (no training changes)
        with torch.no_grad():
            # --- Train recon ---
            train_recon = []
            for i in range(0, len(train_past_data), batch_size):
                batch = train_past_data[i:i+batch_size].to(device)
                z_mean, z_logvar = IB.encode(batch.view(batch.size(0), -1))
                train_recon += [IB.decode_future(z_mean, reshape=False).cpu()]
            train_recon = torch.cat(train_recon, dim=0).numpy()
            np.save(output_path + "_train_future_recon" + str(index) + ".npy", train_recon)

            # --- Test recon ---
            test_recon = []
            for i in range(0, len(test_past_data), batch_size):
                batch = test_past_data[i:i+batch_size].to(device)
                z_mean, z_logvar = IB.encode(batch.view(batch.size(0), -1))
                test_recon += [IB.decode_future(z_mean, reshape=False).cpu()]
            test_recon = torch.cat(test_recon, dim=0).numpy()
            np.save(output_path + "_test_future_recon" + str(index) + ".npy", test_recon)

        final_result = np.array(final_result)
        np.save(final_result_path, final_result)
