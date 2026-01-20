import os
import torch
import torch.autograd as autograd
from tqdm import tqdm

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(*real_samples.shape[:1], 1, device=device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    fake = torch.ones(*d_interpolates.shape, device=device)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                              grad_outputs=fake, create_graph=True,
                              retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train_wgangp(args, generator, discriminator,
                 dataloader, device, lambda_gp=10):
    
    if not os.path.exists(os.path.join(args.ad_save_path, 'fanogan', args.env)):
        os.makedirs(os.path.join(args.ad_save_path, 'fanogan', args.env))

    generator.to(device)
    discriminator.to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=args.lr_ad, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=args.lr_ad, betas=(0.5, 0.999))

    padding_epoch = len(str(args.epochs_ad))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(args.epochs_ad):
        pbar = tqdm(enumerate(dataloader))
        for i, x in pbar:

            # Configure input
            real_data = x.to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.randn(x.shape[0], args.fanogan_latent_dim, device=device)

            # Generate a batch of images
            fake_data = generator(z)

            # Real images
            real_validity = discriminator(real_data)
            # Fake images
            fake_validity = discriminator(fake_data.detach())
            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator,
                                                        real_data.data,
                                                        fake_data.data,
                                                        device)
            # Adversarial loss
            d_loss = (-torch.mean(real_validity) + torch.mean(fake_validity)
                      + lambda_gp * gradient_penalty)

            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()

            # Train the generator and output log every n_critic steps
            if i % args.fanogan_n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_data = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                fake_validity = discriminator(fake_data)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

                pbar.set_description(f"[Epoch {epoch:{padding_epoch}}/{args.epochs_ad}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[D loss: {d_loss.item():3f}] "
                      f"[G loss: {g_loss.item():3f}]")
                
            batches_done += args.fanogan_n_critic

    torch.save({
                'Generator': generator.state_dict(),
                'Discriminator': discriminator.state_dict(),
            }, os.path.join(args.ad_save_path, 'fanogan', args.env, 'gan.pth'))