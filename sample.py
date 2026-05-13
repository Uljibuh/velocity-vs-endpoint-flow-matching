import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def sample_and_plot(models, zeros, device):
    n_samples = 3
    steps = 30
    idx = torch.randint(0, len(zeros), (n_samples,))
    x_start = zeros[idx].to(device)
    t_vals = torch.linspace(0, 1, steps, device=device)

    fig, axes = plt.subplots(n_samples * 2, 6, figsize=(15, n_samples * 3))
    plot_indices = torch.linspace(0, steps - 1, 6).long()

    for m_idx, (mode, model) in enumerate(models.items()):
        model.eval()
        xt = x_start.clone()
        trajectories = [xt.cpu()]

        for i in range(steps - 1):
            t = t_vals[i]
            dt = t_vals[i + 1] - t_vals[i]
            t_batch = torch.full((n_samples, 1), t, device=device)
            pred = model(xt, t_batch)

            if mode == "v":
                v_pred = pred
            elif mode == "x":
                # Convert endpoint prediction to velocity for the ODE step
                denom = torch.clamp(1 - t, min=1e-3)
                v_pred = (pred - xt) / denom

            xt = xt + v_pred * dt
            xt = xt.clamp(0, 1)
            trajectories.append(xt.cpu())

        for i in range(n_samples):
            row = (i * 2) + m_idx
            for j, step_idx in enumerate(plot_indices):
                ax = axes[row, j]
                img = trajectories[step_idx][i].squeeze().numpy()
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                if i == 0 and m_idx == 0:
                    ax.set_title(f"t={t_vals[step_idx].item():.2f}")
                if j == 0:
                    ax.set_ylabel(f"{mode.upper()}-Mode", fontsize=12)
                    ax.axis('on')
                    ax.set_xticks([])
                    ax.set_yticks([])

    plt.tight_layout()
    plt.show()