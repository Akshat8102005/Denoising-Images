import os
import time
import torch
import numpy as np
import torch.nn.functional as F

from models.baseline_cnn import SimpleCNN
from metrics.psnr import calculate_psnr
from metrics.ssim import calculate_ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Configuration
# -------------------------------------------------

TEST_NOISY = r"C:\\Users\\Aksha\\Downloads\\test_NoisyLR"
TEST_GT = r"C:\\Users\\Aksha\\Downloads\\test_GT"

MODEL_PATH = "results/checkpoints/baseline_cnn.pth"

OUTPUT_DIR = "results/baseline/sample_outputs"
METRICS_PATH = "results/baseline/metrics.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------------------------
# Load Model
# -------------------------------------------------

def load_model():

    model = SimpleCNN().to(device)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    return model


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

def evaluate(model):

    files = sorted(os.listdir(TEST_NOISY))

    total_psnr = 0.0
    total_ssim = 0.0
    total_time = 0.0
    total_images = 0

    with torch.no_grad():

        for file in files:

            if not file.endswith(".npy"):
                continue

            # -----------------------
            # Load Noisy Image
            # -----------------------

            noisy = np.load(os.path.join(TEST_NOISY, file))

            noisy = (
                torch.tensor(noisy)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )

            noisy = F.interpolate(
                noisy,
                scale_factor=2,
                mode="bilinear",
                align_corners=False
            )

            noisy = noisy.to(device)

            # -----------------------
            # Load Ground Truth
            # -----------------------

            clean = np.load(os.path.join(TEST_GT, file))

            clean = (
                torch.tensor(clean)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            # -----------------------
            # Inference Timing
            # -----------------------

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.time()

            output = model(noisy)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.time()

            total_time += (end - start)

            # -----------------------
            # Metrics
            # -----------------------

            total_psnr += calculate_psnr(output, clean)
            total_ssim += calculate_ssim(output, clean)

            total_images += 1

            # -----------------------
            # Save Prediction
            # -----------------------

            output_np = output.squeeze().cpu().numpy()

            np.save(
                os.path.join(OUTPUT_DIR, file),
                output_np
            )

    avg_psnr = total_psnr / total_images
    avg_ssim = total_ssim / total_images
    avg_time = total_time / total_images

    print("\n========== Evaluation ==========")
    print(f"Images Processed : {total_images}")
    print(f"Average PSNR     : {avg_psnr:.2f} dB")
    print(f"Average SSIM     : {avg_ssim:.4f}")
    print(f"Inference Time   : {avg_time:.6f} sec/image")
    print("================================")

    with open(METRICS_PATH, "w") as f:

        f.write("Baseline CNN Results\n")
        f.write("====================\n\n")

        f.write(f"Images Processed : {total_images}\n")
        f.write(f"Average PSNR     : {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM     : {avg_ssim:.4f}\n")
        f.write(f"Inference Time   : {avg_time:.6f} sec/image\n")

    print(f"\nMetrics saved to {METRICS_PATH}")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    print(f"Using device : {device}")

    model = load_model()

    evaluate(model)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()