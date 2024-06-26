# Tugas_13_pengelolaan-citra
```
import numpy as np
import matplotlib.pyplot as plt

# Contoh gambar dan dekomposisi SVD
# Misalkan `image` adalah gambar Anda dalam format grayscale
# u, s, v adalah hasil dekomposisi SVD dari `image`
# u, s, v = np.linalg.svd(image, full_matrices=False)

# Jika Anda belum memiliki image atau dekomposisi SVD, gunakan contoh berikut:
from scipy.linalg import svd
image = np.random.rand(64, 64)  # Contoh gambar acak
u, s, v = svd(image)

comps = [3648, 1, 5, 10, 15, 20]
plt.figure(figsize=(12, 6))

for i in range(len(comps)):
    # Pastikan jumlah komponen tidak melebihi dimensi dari u, s, v
    num_components = min(comps[i], len(s))
    
    low_rank = u[:, :num_components] @ np.diag(s[:num_components]) @ v[:num_components, :]

    plt.subplot(2, 3, i + 1)
    plt.imshow(low_rank, cmap='gray')
    
    if i == 0:
        plt.title(f'Actual Image with n_components = {comps[i]}')
    else:
        plt.title(f'n_components = {comps[i]}')

plt.tight_layout()
plt.show()

```
### hasil 
![Screenshot 2024-06-26 095545](https://github.com/riskibowo/Tugas_13_pengelolaan-citra/assets/115862112/9fd0e955-cbcb-4d56-b66b-fb4bd124a601)


hasil gambar yg ber berbeda
```
import requests
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Assign and open image
url = 'https://get.pxhere.com/photo/grass-animal-pet-kitten-cat-feline-mammal-child-pets-animals-mammals-vertebrate-dog-breed-cute-cat-dog-like-mammal-small-to-medium-sized-cats-cat-like-mammal-saarloos-wolfdog-dog-breed-group-1044952.jpg'
response = requests.get(url, stream=True)

with open('image.png', 'wb') as f:
    f.write(response.content)

img = cv2.imread('image.png')

# Converting the image into gray scale for faster computation
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculating the SVD
u, s, v = np.linalg.svd(gray_image, full_matrices=False)

# Inspect shapes of the matrices
print("Shape of u:", u.shape)
print("Shape of s:", s.shape)
print("Shape of v:", v.shape)

# Plot images with different number of components
comps = [3648, 1, 5, 10, 15, 20]
plt.figure(figsize=(12, 6))

for i in range(len(comps)):
    # Ensure number of components does not exceed the dimensions of u, s, v
    num_components = min(comps[i], len(s))
    
    low_rank = u[:, :num_components] @ np.diag(s[:num_components]) @ v[:num_components, :]

    plt.subplot(2, 3, i + 1)
    plt.imshow(low_rank, cmap='gray')
    
    if i == 0:
        plt.title(f'Actual Image with n_components = {comps[i]}')
    else:
        plt.title(f'n_components = {comps[i]}')

plt.tight_layout()
plt.show()

```
![Screenshot 2024-06-26 100018](https://github.com/riskibowo/Tugas_13_pengelolaan-citra/assets/115862112/a0e452d5-9e79-4b9b-945c-500ff7b33d51)
