import torch
import argparse
import numpy as np
from envs import env_name
from torchvision import transforms
from data.loaders import RolloutObservationDataset, ROOT
from utils.misc import IMG_DIM
from models.worldmodel.vae import VAE

parser = argparse.ArgumentParser(description="VAE Trainer")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the VAE")
parser.add_argument("--iternum", type=int, default=0, help="Which iteration of world model to save VAE")
args = parser.parse_args()

torch.manual_seed(123)
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 32
NUM_WORKERS = 2

def get_data_loaders(dataset_path=ROOT):
	transform_train = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMG_DIM, IMG_DIM)), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
	transform_test = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMG_DIM, IMG_DIM)), transforms.ToTensor()])
	dataset_train = RolloutObservationDataset(transform_train, dataset_path, train=True)
	dataset_test = RolloutObservationDataset(transform_test, dataset_path, train=False)
	train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
	test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
	return train_loader, test_loader

def train_loop(train_loader, vae):
	vae.train()
	batch_losses = []
	train_loader.dataset.load_next_buffer()
	for states in train_loader:
		loss = vae.optimize(states)
		batch_losses.append(loss)
	return np.sum(batch_losses) / len(train_loader.dataset)

def test_loop(test_loader, vae):
	vae.eval()
	batch_losses = []
	test_loader.dataset.load_next_buffer()
	with torch.no_grad():
		for states in test_loader:
			loss = vae.get_loss(states).item()
			batch_losses.append(loss)
	return np.sum(batch_losses) / len(test_loader.dataset)

def run(epochs=250, checkpoint_dirname="pytorch"):
	train_loader, test_loader = get_data_loaders()
	vae = VAE(load=False)
	ep_train_losses = []
	ep_test_losses = []
	for ep in range(epochs):
		train_loss = train_loop(train_loader, vae)
		test_loss = test_loop(test_loader, vae)
		ep_train_losses.append(train_loss)
		ep_test_losses.append(test_loss)
		vae.schedule(test_loss)
		vae.save_model(checkpoint_dirname, "latest")
		vae.sample(1, save=True, folder=checkpoint_dirname)
		if ep_test_losses[-1] <= np.min(ep_test_losses): vae.save_model(checkpoint_dirname)
		print(f"Ep: {ep+1} / {epochs}, Train: {ep_train_losses[-1]:.4f}, Test: {ep_test_losses[-1]:.4f}")
		
if __name__ == "__main__":
	run(args.epochs, f"{env_name}/iter{args.iternum}")