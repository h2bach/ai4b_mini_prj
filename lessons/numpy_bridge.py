"""CNN Digits NumPy với backward đầy đủ, để đối chiếu autograd CPU float64."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.datasets import load_digits
import torch
from torch import nn


def numpy_loss_grad(x, y, weights):
    wc, bc, wl, bl = weights
    n = len(x)
    xp = np.pad(x, ((0,0), (0,0), (1,1), (1,1)))
    patches = sliding_window_view(xp, (3,3), axis=(2,3))
    conv = np.einsum('nchwij,ocij->nohw', patches, wc) + bc[None,:,None,None]
    relu = np.maximum(conv, 0)
    pool = relu.reshape(n,8,4,2,4,2).mean(axis=(3,5))
    flat = pool.reshape(n,128)
    logits = flat @ wl.T + bl
    stable = logits - logits.max(1, keepdims=True)
    prob = np.exp(stable); prob /= prob.sum(1, keepdims=True)
    loss = -(stable[np.arange(n),y] - np.log(np.exp(stable).sum(1))).mean()
    dz = prob.copy(); dz[np.arange(n),y] -= 1; dz /= n
    dwl, dbl = dz.T @ flat, dz.sum(0)
    dpool = (dz @ wl).reshape(n,8,4,4)
    drelu = np.repeat(np.repeat(dpool, 2, axis=2), 2, axis=3) / 4
    dconv = drelu * (conv > 0)
    dwc = np.einsum('nohw,nchwij->ocij', dconv, patches)
    dbc = dconv.sum(axis=(0,2,3))
    dxp = np.zeros_like(xp)
    for i in range(3):
        for j in range(3):
            dxp[:,:,i:i+8,j:j+8] += np.einsum('nohw,oc->nchw', dconv, wc[:,:,i,j])
    return logits, loss, [dwc, dbc, dwl, dbl], dxp[:,:,1:-1,1:-1]


def run_bridge():
    torch.manual_seed(7)
    digits = load_digits()
    x = digits.images[:8,None].astype(np.float64) / 16
    y = digits.target[:8]
    model = nn.Sequential(nn.Conv2d(1,8,3,padding=1), nn.ReLU(), nn.AvgPool2d(2),
                          nn.Flatten(), nn.Linear(128,10)).double().cpu()
    params = list(model.parameters())
    weights = [p.detach().numpy().copy() for p in params]
    logits, loss, grads, dx = numpy_loss_grad(x, y, weights)
    tx = torch.tensor(x, requires_grad=True)
    torch_logits = model(tx)
    torch_loss = nn.functional.cross_entropy(torch_logits, torch.tensor(y))
    torch_loss.backward()
    differences = {'forward': float(np.max(np.abs(logits-torch_logits.detach().numpy()))),
                   'loss': abs(loss-torch_loss.item()), 'input_gradient': float(np.max(np.abs(dx-tx.grad.numpy())))}
    for i, (p, g) in enumerate(zip(params, grads)):
        differences[f'parameter_gradient_{i}'] = float(np.max(np.abs(g-p.grad.numpy())))
        np.testing.assert_allclose(g, p.grad.numpy(), atol=1e-6, rtol=1e-6)
    lr = .05
    updated = [w-lr*g for w,g in zip(weights,grads)]
    torch.optim.SGD(model.parameters(), lr=lr).step()
    differences['one_update'] = max(float(np.max(np.abs(w-p.detach().numpy()))) for w,p in zip(updated,params))
    assert max(differences.values()) < 1e-6
    return {'input_shape': list(x.shape), 'loss': float(loss), 'max_absolute_errors': differences}


if __name__ == '__main__':
    print(run_bridge())
