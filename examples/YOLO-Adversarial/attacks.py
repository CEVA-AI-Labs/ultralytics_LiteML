import torch


def FGSM2(model, batch, eps=0.01):
    batch["img"].requires_grad = True
    loss = model.model(batch)  # loss(box, cls, dfl)
    model.zero_grad()
    loss[1].backward()  # for my modified loss function with 3 separate losses
    # loss[0].backward()  # for original loss function
    data_grad = batch["img"].grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = batch["img"] + eps * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


class FGSM:
    def __init__(self, model, eps=0.1):
        self.model = model
        self.eps = eps

    def __call__(self, batch):
        batch["img"].requires_grad = True
        loss = self.model.model(batch)  # loss(box, cls, dfl)
        self.model.zero_grad()
        loss[0].backward()
        data_grad = batch["img"].grad.data
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = batch["img"] + self.eps * sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image


class PGD:
    """
    Implementation of PGD attack from the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083] with L2 norm on ultralytics YOLO models.

    """
    def __init__(self, model, eps=1.0, alpha=0.2, steps=10, random_start=True, eps_for_division=1e-10):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division

    def __call__(self, batch):
        images = batch["img"].clone().detach()
        adv_images = images.clone().detach()
        batch_size = len(images)
        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_images).normal_()
            d_flat = delta.view(adv_images.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(adv_images.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps
            adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

        for _ in range(self.steps):
            batch["img"] = adv_images
            batch["img"].requires_grad = True
            loss = self.model.model(batch)
            loss = loss[0]
            # Update adversarial images
            grad = torch.autograd.grad(
                loss, adv_images, retain_graph=False, create_graph=False
            )[0]
            grad_norms = (
                    torch.norm(grad.view(batch_size, -1), p=2, dim=1)
                    + self.eps_for_division
            )  # nopep8
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_images = adv_images.detach() + self.alpha * grad

            delta = adv_images - images
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)

            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images