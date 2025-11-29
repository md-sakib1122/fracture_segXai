import os
from PIL import Image
import numpy as np

DATA_DIR = "FracAtlas"
splits = ["train", "val", "test"]

for split in splits:
   img_dir = os.path.join(DATA_DIR, split, "images")
   mask_dir = os.path.join(DATA_DIR, split, "masks")

   img_files = sorted(os.listdir(img_dir))
   mask_files = sorted(os.listdir(mask_dir))

   print(f"\n=== {split.upper()} ===")
   print(f"Images: {len(img_files)}, Masks: {len(mask_files)}")

   # Check missing masks
   missing_masks = [f for f in img_files if f.rsplit('.', 1)[0] + "_mask.png" not in mask_files]
   if missing_masks:
       print(f"Missing masks for {len(missing_masks)} images: {missing_masks[:5]} ...")
   else:
       print("All images have corresponding masks.")

   # Count fracture vs non-fracture masks
   num_blank = 0
   num_fracture = 0

   for f in mask_files:
       mask_path = os.path.join(mask_dir, f)
       mask = np.array(Image.open(mask_path))

       if mask.max() == 0:
           num_blank += 1
       else:
           num_fracture += 1



   print(f"Non-fractured (blank) masks: {num_blank}")
   print(f"Fractured masks: {num_fracture}")

   # Show some examples
   print("Example blank masks:", [f for f in mask_files if np.array(Image.open(os.path.join(mask_dir, f))).max() == 0][:3])
   print("Example fractured masks:", [f for f in mask_files if np.array(Image.open(os.path.join(mask_dir, f))).max() > 0][:3])