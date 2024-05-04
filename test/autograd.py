import torch
import torch.nn.functional as F

class Lamb(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, lamb, reward, cost, coeff):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(lamb, reward, cost, coeff)
        return lamb

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        lamb, reward, cost, coeff = ctx.saved_tensors
        objectives = (reward.squeeze(dim=-1) + cost @ lamb) / coeff
        objectives = F.softmax(objectives, dim=0)
        gradient = objectives @ cost
        return grad_output * gradient, None, None, None

lamb = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
reward = torch.randn(3, 1, requires_grad=False)
cost = torch.randn(3, 1, requires_grad=False)
coeff = torch.tensor(1.0)
Lamb_func = Lamb.apply
loss = Lamb_func(lamb, reward, cost, coeff)


optimizer = torch.optim.SGD([lamb, reward, cost, coeff], lr=0.1)
loss.backward()
optimizer.step()
