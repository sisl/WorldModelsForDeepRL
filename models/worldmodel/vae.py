import os
import torch
import numpy as np
import matplotlib.pyplot as plt

IMG_CHANNELS = 3				# The number of channels of the input state image
LATENT_SIZE = 32				# The size of the latent vector output by the encoder network

class Encoder(torch.nn.Module):
	def __init__(self, img_channels, latent_size):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(img_channels, 32, 4, stride=2)
		self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2)
		self.conv3 = torch.nn.Conv2d(64, 128, 4, stride=2)
		self.conv4 = torch.nn.Conv2d(128, 256, 4, stride=2)
		self.fc_mu = torch.nn.Linear(2*2*256, latent_size)
		self.fc_logsigma = torch.nn.Linear(2*2*256, latent_size)

	def forward(self, x):
		x = torch.nn.functional.relu(self.conv1(x))
		x = torch.nn.functional.relu(self.conv2(x))
		x = torch.nn.functional.relu(self.conv3(x))
		x = torch.nn.functional.relu(self.conv4(x))
		x = x.view(x.size(0), -1)
		mu = self.fc_mu(x)
		logsigma = self.fc_logsigma(x)
		return mu, logsigma

class Decoder(torch.nn.Module):
	def __init__(self, img_channels, latent_size):
		super().__init__()
		self.fc1 = torch.nn.Linear(latent_size, 1024)
		self.deconv1 = torch.nn.ConvTranspose2d(1024, 128, 5, stride=2)
		self.deconv2 = torch.nn.ConvTranspose2d(128, 64, 5, stride=2)
		self.deconv3 = torch.nn.ConvTranspose2d(64, 32, 6, stride=2)
		self.deconv4 = torch.nn.ConvTranspose2d(32, img_channels, 6, stride=2)

	def forward(self, x):
		x = torch.nn.functional.relu(self.fc1(x))
		x = x.unsqueeze(-1).unsqueeze(-1)
		x = torch.nn.functional.relu(self.deconv1(x))
		x = torch.nn.functional.relu(self.deconv2(x))
		x = torch.nn.functional.relu(self.deconv3(x))
		reconstruction = torch.sigmoid(self.deconv4(x))
		return reconstruction

class VAE(torch.nn.Module):
	def __init__(self, img_channels=IMG_CHANNELS, latent_size=LATENT_SIZE, load="", gpu=True):
		super().__init__()
		self.device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
		self.encoder = Encoder(img_channels, latent_size).to(self.device)
		self.decoder = Decoder(img_channels, latent_size).to(self.device)
		self.optimizer = torch.optim.Adam(self.parameters())
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=5)
		self.latent_size = latent_size
		if load: self.load_model(load)

	def forward(self, x):
		x = x.to(self.device)
		mu, logsigma = self.encoder(x)
		sigma = logsigma.exp()
		eps = torch.randn_like(sigma)
		z = eps.mul(sigma).add_(mu)
		x_hat = self.decoder(z)
		return x_hat, mu, logsigma

	def get_loss(self, x):
		x = x.to(self.device)
		x_hat, mu, logsigma = self.forward(x)
		KLD = -0.5 * torch.sum(1 + 2*logsigma - mu.pow(2) - (2*logsigma).exp())
		BCE = torch.nn.functional.mse_loss(x_hat, x, reduction="sum")
		# BCE = torch.nn.functional.binary_cross_entropy_with_logits(x_hat, x, reduction="sum")
		return BCE + KLD

	def get_latents(self, x):
		with torch.no_grad():
			out_dims = x.size()[:-3]
			x = x.view(-1, *x.size()[-3:]).to(self.device)
			mu, logsigma = self.encoder(x)
			eps = torch.randn_like(mu)
			z = mu + logsigma.exp()*eps
			z = z.view(*out_dims, -1)
			return z

	def optimize(self, x):
		self.optimizer.zero_grad()
		loss = self.get_loss(x)
		loss.backward()
		self.optimizer.step()
		return loss.item()

	def schedule(self, test_loss):
		self.scheduler.step(test_loss)

	def sample(self, batch=1, x=None, save=False, folder="pytorch"):
		z = torch.randn([batch, self.latent_size]) if not x else self.get_latents(x)
		x_hat = self.decoder(z.to(self.device)).permute(0,2,3,1)*255
		images = x_hat.cpu().detach().numpy().astype(np.uint8)
		if save:
			filepath = f"./tests/vae_samples/{folder}/"
			os.makedirs(os.path.dirname(filepath), exist_ok=True)
			img_number = len([n for n in os.listdir(filepath)])
			plt.imsave(f"{filepath}vae_{img_number+1}.png", images[0])
		return images

	def save_model(self, dirname="pytorch", name="best"):
		filepath = get_checkpoint_path(dirname, name)
		os.makedirs(os.path.dirname(filepath), exist_ok=True)
		torch.save(self.encoder.state_dict(), filepath.replace(".pth", "_e.pth"))
		torch.save(self.decoder.state_dict(), filepath.replace(".pth", "_d.pth"))
		
	def load_model(self, dirname="pytorch", name="best"):
		filepath = get_checkpoint_path(dirname, name)
		if os.path.exists(filepath.replace(".pth", "_e.pth")):
			self.encoder.load_state_dict(torch.load(filepath.replace(".pth", "_e.pth"), map_location=self.device))
			self.decoder.load_state_dict(torch.load(filepath.replace(".pth", "_d.pth"), map_location=self.device))
			print(f"Loaded VAE model at {filepath}")
		return self

def get_checkpoint_path(dirname="pytorch", name="best"):
	return f"./saved_models/vae/{dirname}/{name}.pth"