import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim

# Import your model + loss
from vae import VAE, vae_loss   # assumes you saved your provided code in vae_model.py

def main():
    parser = argparse.ArgumentParser(description="Train a VAE on (p1,p2,p3) dataset.")
    parser.add_argument("--csv", help="Path to CSV dataset containing columns p1,p2,p3", default='/Users/marcobarbacci/foam/4th-Year-Research-Project/2D_Reactor/datasets/run_20251002_115523/vae_params_20251002_115524.csv')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-model", default="/Users/marcobarbacci/foam/4th-Year-Research-Project/2D_Reactor/VAE/Weights/vae_model.pt", help="Path to save trained model")
    args = parser.parse_args()

    # ----------------------------
    # Load dataset
    # ----------------------------
    df = pd.read_csv(args.csv)
    if not set(["p1", "p2", "p3"]).issubset(df.columns):
        raise ValueError("CSV must contain columns: p1, p2, p3")

    X = df[["p1", "p2", "p3"]].values.astype("float32")

    # Normalize to [0,1]
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    dataset = TensorDataset(torch.tensor(X_norm))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ----------------------------
    # Initialize model
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(input_dim=3, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ----------------------------
    # Training loop
    # ----------------------------
    model.train()
    for epoch in range(args.epochs):
        train_loss = 0
        for batch in dataloader:
            batch = batch[0].to(device)  # unpack TensorDataset
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

    # ----------------------------
    # Save trained model + normalization
    # ----------------------------
    torch.save({
        "model_state": model.state_dict(),
        "X_min": X_min,
        "X_max": X_max,
        "args": vars(args),
    }, args.save_model)

    print(f"Model saved to {args.save_model}")


if __name__ == "__main__":
    main()
