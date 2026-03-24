import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import fftconvolve
from skimage.metrics import structural_similarity as ssim

# =====================================================
# 1. FUNGSI UTILITAS & METRIK
# =====================================================
def calculate_mse(original, restored):
    return np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)

def calculate_psnr(original, restored):
    mse = calculate_mse(original, restored)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def magnitude_spectrum(image):
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(1 + np.abs(fshift))
    return magnitude

# =====================================================
# 2. LOAD ORIGINAL IMAGE
# =====================================================
original_uint8 = cv2.imread('foto1.jpeg')
if original_uint8 is None:
    raise FileNotFoundError("Error: 'foto1.jpeg' tidak ditemukan di direktori saat ini.")

original_uint8 = cv2.cvtColor(original_uint8, cv2.COLOR_BGR2RGB)
# Resize opsional untuk mempercepat komputasi (bisa dihapus jika ingin resolusi asli)
original_uint8 = cv2.resize(original_uint8, (512, 512))
original = original_uint8.astype(np.float32)

# =====================================================
# 3. ESTIMASI PSF (Motion Blur 30°, Panjang 15)
# =====================================================
def motion_psf(length=15, angle=30):
    psf = np.zeros((length, length))
    center = length // 2
    angle_rad = np.deg2rad(angle)
    for i in range(length):
        x = int(center + (i - center) * np.cos(angle_rad))
        y = int(center + (i - center) * np.sin(angle_rad))
        if 0 <= x < length and 0 <= y < length:
            psf[y, x] = 1
    psf /= psf.sum()
    return psf

psf = motion_psf(15, 30)

# =====================================================
# 4. DEGRADASI CITRA
# =====================================================
def apply_blur_color(image, psf):
    result = np.zeros_like(image)
    for c in range(3):
        result[:,:,c] = fftconvolve(image[:,:,c], psf, mode='same')
    return result

# A. Motion Blur
blurred = apply_blur_color(original, psf)

# B. Gaussian Noise + Motion Blur (σ=20)
gaussian_noise = np.random.normal(0, 20, original.shape)
blur_gaussian = blurred + gaussian_noise

# C. Salt-and-Pepper Noise (5%) + Motion Blur
sp_noise = blurred.copy()
prob = 0.05
rand = np.random.rand(*original.shape[:2])
for c in range(3):
    sp_noise[:,:,c][rand < prob/2] = 0
    sp_noise[:,:,c][rand > 1 - prob/2] = 255

# Clip nilai ke rentang 0-255
blurred = np.clip(blurred, 0, 255)
blur_gaussian = np.clip(blur_gaussian, 0, 255)
sp_noise = np.clip(sp_noise, 0, 255)

# =====================================================
# 5. METODE RESTORASI
# =====================================================
def inverse_filter(img_blur, psf, threshold=1e-3):
    result = np.zeros_like(img_blur)
    for c in range(3):
        G = np.fft.fft2(img_blur[:,:,c])
        H = np.fft.fft2(psf, s=img_blur[:,:,c].shape)
        
        # Thresholding untuk menghindari pembagian dengan nol
        H_inv = np.zeros_like(H)
        mask = np.abs(H) > threshold
        H_inv[mask] = 1.0 / H[mask]
        
        F_hat = G * H_inv
        result[:,:,c] = np.real(np.fft.ifft2(F_hat))
    return np.clip(result, 0, 255)

def wiener_filter(img_blur, psf, K=0.01):
    result = np.zeros_like(img_blur)
    for c in range(3):
        G = np.fft.fft2(img_blur[:,:,c])
        H = np.fft.fft2(psf, s=img_blur[:,:,c].shape)
        H_conj = np.conj(H)
        
        F_hat = (H_conj / (np.abs(H)**2 + K)) * G
        result[:,:,c] = np.real(np.fft.ifft2(F_hat))
    return np.clip(result, 0, 255)

def lucy_richardson_scipy(img_blur, psf, iterations=15):
    result = np.zeros_like(img_blur)
    psf_mirror = psf[::-1, ::-1] # PSF dibalik untuk korelasi silang
    
    for c in range(3):
        img_channel = img_blur[:,:,c] / 255.0
        est = np.full(img_channel.shape, 0.5) # Tebakan awal
        
        for _ in range(iterations):
            est_conv = fftconvolve(est, psf, mode='same')
            est_conv[est_conv == 0] = 1e-5 # Stabilitas numerik
            relative_blur = img_channel / est_conv
            error_est = fftconvolve(relative_blur, psf_mirror, mode='same')
            est = est * error_est
            
        result[:,:,c] = est * 255.0
    return np.clip(result, 0, 255)

# =====================================================
# 6. PIPELINE EVALUASI & EKSEKUSI
# =====================================================
datasets = {
    "Motion Blur": (blurred, 0.001),                   # K sangat kecil karena hampir tidak ada noise
    "Gaussian + Blur": (blur_gaussian, (20**2)/2500),  # Estimasi K = var(noise) / var(signal)
    "SP + Blur": (sp_noise, 0.05)                      # Estimasi kasar K untuk S&P
}

results_table = []
restored_images = {}

print("Memulai proses restorasi. Harap tunggu...")

for name, (degraded, K_est) in datasets.items():
    print(f"\nMemproses: {name}...")
    restored_images[name] = {'degraded': degraded}
    
    # 1. Inverse Filter
    start = time.time()
    inv_img = inverse_filter(degraded, psf)
    inv_time = time.time() - start
    
    # 2. Wiener Filter
    start = time.time()
    wnr_img = wiener_filter(degraded, psf, K=K_est)
    wnr_time = time.time() - start
    
    # 3. Lucy-Richardson
    start = time.time()
    lucy_img = lucy_richardson_scipy(degraded, psf)
    lucy_time = time.time() - start
    
    # Evaluasi Metrik
    for method, img, t in [("Inverse", inv_img, inv_time), 
                           ("Wiener", wnr_img, wnr_time), 
                           ("Lucy-R", lucy_img, lucy_time)]:
        
        img_uint8 = img.astype(np.uint8)
        mse_val = calculate_mse(original_uint8, img_uint8)
        psnr_val = calculate_psnr(original_uint8, img_uint8)
        ssim_val = ssim(original_uint8, img_uint8, channel_axis=2, data_range=255)
        
        results_table.append([name, method, f"{mse_val:.2f}", f"{psnr_val:.2f}", f"{ssim_val:.4f}", f"{t:.4f}"])
        restored_images[name][method.lower()] = img_uint8

print("\nProses selesai. Membuka visualisasi...")

# =====================================================
# 7. VISUALISASI HASIL RESTORASI (Gambar)
# =====================================================
fig_img, axes_img = plt.subplots(3, 4, figsize=(16, 12))
fig_img.suptitle("Evaluasi Visual Metode Restorasi Citra", fontsize=18)

row = 0
for name, imgs in restored_images.items():
    axes_img[row, 0].imshow(imgs['degraded'].astype(np.uint8))
    axes_img[row, 0].set_title(f"Degraded:\n{name}")
    axes_img[row, 0].axis('off')
    
    axes_img[row, 1].imshow(imgs['inverse'])
    axes_img[row, 1].set_title("Inverse Filter")
    axes_img[row, 1].axis('off')
    
    axes_img[row, 2].imshow(imgs['wiener'])
    axes_img[row, 2].set_title("Wiener Filter")
    axes_img[row, 2].axis('off')
    
    axes_img[row, 3].imshow(imgs['lucy-r'])
    axes_img[row, 3].set_title("Lucy-Richardson")
    axes_img[row, 3].axis('off')
    row += 1

plt.tight_layout()
plt.show()

# =====================================================
# 8. TABEL EVALUASI METRIK
# =====================================================
fig_table = plt.figure(figsize=(10, 4))
ax = fig_table.add_subplot(111)
ax.axis('off')
column_labels = ["Skenario Degradasi", "Metode", "MSE", "PSNR", "SSIM", "Time (s)"]

table = ax.table(cellText=results_table, colLabels=column_labels, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

plt.title("Tabel Perbandingan Kinerja Metode Restorasi", fontsize=14, pad=10)
plt.tight_layout()
plt.show()

# =====================================================
# 9. VISUALISASI SPEKTRUM FREKUENSI
# =====================================================
plt.figure(figsize=(14, 4))
plt.subplot(1,4,1)
plt.imshow(magnitude_spectrum(original_uint8), cmap='gray')
plt.title("Spektrum Original")
plt.axis('off')

plt.subplot(1,4,2)
plt.imshow(magnitude_spectrum(blurred), cmap='gray')
plt.title("Spektrum Motion Blur")
plt.axis('off')

plt.subplot(1,4,3)
plt.imshow(magnitude_spectrum(blur_gaussian), cmap='gray')
plt.title("Spektrum Gauss + Blur")
plt.axis('off')

plt.subplot(1,4,4)
plt.imshow(magnitude_spectrum(sp_noise), cmap='gray')
plt.title("Spektrum S&P + Blur")
plt.axis('off')

plt.tight_layout()
plt.show()