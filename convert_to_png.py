import nibabel as nib
import numpy as np
import os
import cv2

images_dir = r"C:\Users\memir\Downloads\prostate_dataset\Task05_Prostate\imagesTr"
masks_dir  = r"C:\Users\memir\Downloads\prostate_dataset\Task05_Prostate\labelsTr"

output_img_dir  = r"C:\Users\memir\Downloads\images"
output_mask_dir = r"C:\Users\memir\Downloads\masks"
os.makedirs(output_img_dir,  exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz") and not f.startswith("._")])
mask_files  = sorted([f for f in os.listdir(masks_dir)  if f.endswith(".nii.gz") and not f.startswith("._")])

for i in range(10):  # 10 dosya
    try:
        img_path  = os.path.join(images_dir, image_files[i])
        mask_path = os.path.join(masks_dir,  mask_files[i])

        img_nii  = nib.load(img_path)
        mask_nii = nib.load(mask_path)

        img_data  = img_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        for z in range(-2, 3):  # 5 orta slice: -2, -1, 0, 1, 2
            slice_index = img_data.shape[2] // 2 + z
            img_slice  = img_data[:, :, slice_index, 0]  # modalite 0
            mask_slice = mask_data[:, :, slice_index]

            # NaN temizliği
            img_slice = np.nan_to_num(img_slice)

            # Normalize et
            img_min, img_max = img_slice.min(), img_slice.max()
            if img_max - img_min == 0:
                img_norm = np.zeros_like(img_slice, dtype=np.uint8)
            else:
                img_norm = ((img_slice - img_min) / (img_max - img_min) * 255).astype(np.uint8)

            # Maske ikili hale getir
            mask_binary = (mask_slice > 0).astype(np.uint8) * 255

            # Boyut kontrolü ve kayıt
            if img_norm.ndim == 2 and mask_binary.ndim == 2:
                filename = f"{i}_{z + 2}"  # örnek: 0_0, 0_1, ... 9_4
                cv2.imwrite(os.path.join(output_img_dir,  f"img_{filename}.png"),  img_norm)
                cv2.imwrite(os.path.join(output_mask_dir, f"mask_{filename}.png"), mask_binary)
                print(f"[{filename}] ✅ Kaydedildi")
            else:
                print(f"[{i}] ❌ Geçersiz boyut: {img_norm.shape}")

    except Exception as e:
        print(f"[{i}] ⚠️ HATA: {str(e)}")
