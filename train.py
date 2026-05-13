import torch
import torch.nn.functional as F
from data import get_mnist_digits
from models import UNet
from sample import sample_and_plot

def train_step(model, opt, x0, x1, mode, device):
    model.train()
    batch_size = x0.size(0)
    t = torch.rand((batch_size, 1), device=device)
    t_img = t[:, :, None, None]
    
    # Linear interpolation between start (x0) and end (x1)
    xt = (1 - t_img) * x0 + t_img * x1

    if mode == "v":
        v_true = x1 - x0
        v_pred = model(xt, t)
        loss = F.mse_loss(v_pred, v_true)
    elif mode == "x":
        x1_pred = model(xt, t)
        loss = F.mse_loss(x1_pred, x1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()

def run_comparison():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Target digits: 0 to 2
    zeros = get_mnist_digits(0).to(device)
    twos = get_mnist_digits(2).to(device)
    
    modes = ["v", "x"]
    models = {}
    batch_size = 128
    epochs = 500  
    
    for mode in modes:
        print(f"\nTraining U-Net {mode}-prediction model...")
        model = UNet().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for step in range(epochs):
            idx0 = torch.randint(0, len(zeros), (batch_size,))
            idx1 = torch.randint(0, len(twos), (batch_size,))
            loss = train_step(model, opt, zeros[idx0], twos[idx1], mode, device)
            
            if step % 100 == 0:
                print(f"  Step {step:04d} | Loss: {loss:.4f}")
        models[mode] = model
        
    print("\nGenerating comparative samples...")
    sample_and_plot(models, zeros, device)

if __name__ == "__main__":
    run_comparison()