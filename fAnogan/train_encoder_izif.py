import os
import torch
import torch.nn as nn
from tqdm import tqdm

def train_encoder_izif(args, generator, discriminator, encoder,
                       dataloader, device, kappa=1.0):
    
    if not os.path.exists(os.path.join(args.ad_save_path, 'fanogan', args.env)):
        os.makedirs(os.path.join(args.ad_save_path, 'fanogan', args.env))

    statedict = torch.load(os.path.join(args.ad_save_path, 'fanogan', args.env, 'gan.pth'))
    generator.load_state_dict(statedict['Generator'])
    discriminator.load_state_dict(statedict['Discriminator'])

    generator.to(device).eval()
    discriminator.to(device).eval()
    encoder.to(device)

    criterion = nn.MSELoss()

    optimizer_E = torch.optim.Adam(encoder.parameters(),
                                   lr=args.lr_ad, betas=(0.5, 0.999))

    padding_epoch = len(str(args.epochs_ad))
    padding_i = len(str(len(dataloader)))

    batches_done = 0
    for epoch in range(args.epochs_ad):
        pbar = tqdm(enumerate(dataloader))
        for i, x in pbar:

            # Configure input
            real_data = x.to(device)

            # ----------------
            #  Train Encoder
            # ----------------

            optimizer_E.zero_grad()

            # Generate a batch of latent variables
            z = encoder(real_data)

            # Generate a batch of images
            fake_data = generator(z)

            # Real features
            real_features = discriminator.forward_features(real_data)
            # Fake features
            fake_features = discriminator.forward_features(fake_data)

            # izif architecture
            loss_data = criterion(fake_data, real_data)
            loss_features = criterion(fake_features, real_features)
            e_loss = loss_data + kappa * loss_features

            e_loss.backward()
            optimizer_E.step()

            # Output training log every n_critic steps
            if i % args.fanogan_n_critic == 0:
                pbar.set_description(f"[Epoch {epoch:{padding_epoch}}/{args.epochs_ad}] "
                      f"[Batch {i:{padding_i}}/{len(dataloader)}] "
                      f"[E loss: {e_loss.item():3f}]")

                batches_done += args.fanogan_n_critic
    torch.save(encoder.state_dict(), os.path.join(args.ad_save_path, 'fanogan', args.env, 'encoder.pth'))