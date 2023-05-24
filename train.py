import os
import torch
import torch.nn.functional as F
from config import *
import time 
import matplotlib.pyplot as plt
from tqdm import tqdm

    
def train_model(model, train_loader, val_loader, optimizer):
    # initialize a dictionary to store training history
    H = {"epochs": [], "train_reconstruction_loss": [], "train_kl_div_loss": [], "train_total_loss": [],
         "val_reconstruction_loss": [], "val_kl_div_loss": [], "val_total_loss": [],}
    # loop over epochs
    print("[INFO] Training the network...")
    start_time = time.time()

    # Train the CNN
    for epoch in tqdm(range(NUM_EPOCHS)):
        reconstruction_loss = 0.0
        kl_div_loss = 0.0
        total_loss = 0.0
        val_reconstruction_loss = 0.0
        val_kl_div_loss = 0.0
        val_total_loss = 0.0
        
        for i, (images) in enumerate(train_loader):
            images = images.to(DEVICE)
            recon_images, mu, log_var = model(images)
            # Compute the reconstruction loss and KL divergence
            r_loss = F.binary_cross_entropy(recon_images, images, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            t_loss = r_loss + BETA * kl_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            t_loss.backward()
            optimizer.step()
            
            reconstruction_loss += r_loss.item() 
            kl_div_loss += kl_loss.item()
            total_loss += t_loss.item() 
        avg_reconstruction_loss = round(reconstruction_loss / len(train_loader.dataset), 2)
        avg_kl_div_loss = round(kl_div_loss / len(train_loader.dataset), 2)
        avg_total_loss = round(total_loss / len(train_loader.dataset), 2)

        # update our training history
        H["epochs"].append(epoch + 1)
        H["train_reconstruction_loss"].append(avg_reconstruction_loss)
        H["train_kl_div_loss"].append(avg_kl_div_loss)
        H["train_total_loss"].append(avg_total_loss)
        
        model.eval()
        with torch.no_grad():
            for i, (images) in enumerate(val_loader):
                images = images.to(DEVICE)
                recon_images, mu, log_var = model(images)
                # Compute the reconstruction loss and KL divergence
                r_loss = F.binary_cross_entropy(recon_images, images, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                t_loss = r_loss + BETA * kl_loss

                
                val_reconstruction_loss += r_loss.item() 
                val_kl_div_loss += kl_loss.item()
                val_total_loss += t_loss.item() 
                
            avg_val_reconstruction_loss = round(val_reconstruction_loss / len(val_loader.dataset), 2)
            avg_val_kl_div_loss = round(val_kl_div_loss / len(val_loader.dataset), 2)
            avg_val_total_loss = round(val_total_loss / len(val_loader.dataset), 2)
            
        # update our validation history
        H["val_reconstruction_loss"].append(avg_val_reconstruction_loss)
        H["val_kl_div_loss"].append(avg_val_kl_div_loss)
        H["val_total_loss"].append(avg_val_total_loss)
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Train Reconstruction Loss: {avg_reconstruction_loss}, Train KL Divergence Loss: {avg_kl_div_loss}, Train Total Loss: {avg_total_loss} \
            Val Reconstruction Loss: {avg_val_reconstruction_loss}, Val KL Divergence Loss: {avg_val_kl_div_loss}, Val Total Loss: {avg_val_total_loss}')
        
    # display the total time needed to perform the training
    end_time = time.time()
    print("[INFO] total time taken to train the model: {:.2f}s".format(end_time - start_time))

    # serialize the model to disk
    MODEL_PATH = f"{NAME}.pth"
    print(MODEL_PATH)
    torch.save(model, os.path.join(BASE_OUTPUT, MODEL_PATH))
    
    print("Plotting the Loss curves...")
    # plot the reconstruction loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["epochs"], H["train_reconstruction_loss"], label="Train Reconstruction Loss")
    plt.plot(H["epochs"], H["val_reconstruction_loss"], label="Val Reconstruction Loss")
    plt.title("Training and Validation Reconstruction Loss")
    plt.xlabel("Number of Epochs")
    plt.xticks([i for i in range(0, NUM_EPOCHS + 2, 20)])
    plt.legend(loc="best")
    plt.savefig(os.path.join(BASE_OUTPUT, f"{NAME} | Reconstruction Loss.jpg"))
    plt.close()
    
    # plot the KL Divergence loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["epochs"], H["train_kl_div_loss"], label="Train KL_Div Loss")
    plt.plot(H["epochs"], H["val_kl_div_loss"], label="Val KL_Div Loss")
    plt.title("Training and Validation KL Divergence Loss")
    plt.xlabel("Number of Epochs")
    plt.xticks([i for i in range(0, NUM_EPOCHS + 2, 20)])
    plt.legend(loc="best")
    plt.savefig(os.path.join(BASE_OUTPUT, f"{NAME} | KL Divergence Loss.jpg"))
    plt.close()
    
    # plot the total loss
    plt.style.use("ggplot")
    # plt.figure()
    plt.plot(H["epochs"], H["train_total_loss"], label="Train Total Loss")
    plt.plot(H["epochs"], H["val_total_loss"], label="Val Total Loss")
    plt.title("Training and Validation Total Loss")
    plt.xlabel("Number of Epochs")
    plt.xticks([i for i in range(0, NUM_EPOCHS + 2, 20)])
    plt.legend(loc="best")
    plt.savefig(os.path.join(BASE_OUTPUT, f"{NAME} | Total Loss.jpg"))
    plt.close()
