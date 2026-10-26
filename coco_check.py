import os
from PIL import Image
import numpy as np
def find_broken_images(image_dir, output_txt="broken_images.txt", log_to_console=True):
    broken = []
    total = 0

    for root, _, files in os.walk(image_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(root, fname)
            total += 1
            try:
                with Image.open(fpath) as img:
                    image_data = np.array(img)
                    image = Image.fromarray(np.array(image_data)).convert('RGB')
                    image = Image.open(os.path.join(fpath)).convert('RGB')

            except Exception as e:
                broken.append(fpath)
                if log_to_console:
                    print(f"[BROKEN] {fpath} ({type(e).__name__}: {e})")

    # 결과 저장
    if broken:
        with open(output_txt, "w") as f:
            for b in broken:
                f.write(b + "\n")
        print(f"\n❌ 총 {len(broken)}개의 깨진 이미지 발견 → '{output_txt}'에 저장됨.")
    else:
        print("✅ 깨진 이미지 없음!")
    print(f"총 검사한 이미지 수: {total}")

# 예시 실행
if __name__ == "__main__":
    image_root = "/data1/dataset/LLaVA-Instruct-150K/tuning_data/coco/train2017"
    find_broken_images(image_root, output_txt="broken_train2017.txt")

    image_root = "/data1/dataset/LLaVA-Instruct-150K/tuning_data/coco/val2017"
    find_broken_images(image_root, output_txt="broken_val2017.txt")
