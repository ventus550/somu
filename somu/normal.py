from .somu import som
from numpy import random

def normal_som(data, count, epochs=1, sigma_initial=1.0, device=None, seed=None):
	units = random.randn(count, 2) / 4
	return som(data, units, epochs=epochs, sigma_initial=sigma_initial, device=device, seed=seed)