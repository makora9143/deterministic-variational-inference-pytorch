import torch
import matplotlib.pyplot as plt

def toydata_result_plot(trainloader, model=None):
    mean_model = trainloader.dataset.base_model
    std_model = trainloader.dataset.noise_model
    
    xs, ys = iter(trainloader).next()
    
    train_x = torch.arange(torch.min(xs.reshape(-1)), torch.max(xs.reshape(-1)), 1/100).cpu()
    
    plt.plot(train_x.numpy(), mean_model(train_x).numpy(), 'red', label='data mean')
    plt.fill_between(train_x.cpu().numpy(),
                     (mean_model(train_x) - std_model(train_x)).numpy(),
                     (mean_model(train_x) + std_model(train_x)).numpy(),
                     color='orange', alpha=1, label='data 1-std')
    plt.plot(xs.cpu().numpy(), ys.cpu().numpy(), 'r.', alpha=0.2, label='train sample')
    
    
    if model is not None:
        with torch.no_grad():
            test_x = torch.arange(-1, 1, 1/100).reshape(-1, 1)
            pred = model(test_x)
            y_mean   = pred.mean[:,0].cpu()
            ell_mean = pred.mean[:,1].cpu()
            y_var    = pred.var[:,0,0].cpu()
            ell_var  = pred.var[:,1,1].cpu()

            heteroskedastic_part = torch.exp(0.5 * ell_mean)
            full_std = torch.sqrt(y_var + torch.exp(ell_mean + 0.5 * ell_var))

        plt.plot(test_x.cpu().numpy(), y_mean.numpy(), label='model mean')
        plt.fill_between(test_x.cpu().reshape(-1).numpy(),
                         (y_mean - heteroskedastic_part).numpy(),
                         (y_mean + heteroskedastic_part).numpy(),
                         color='g', alpha = 0.2, label='$\ell$ contrib')
        plt.fill_between(test_x.cpu().reshape(-1).numpy(),
                         (y_mean - full_std).numpy(),
                         (y_mean + full_std).numpy(),
                         color='b', alpha = 0.2, label='model 1-std')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim([-3,2])
    plt.legend()