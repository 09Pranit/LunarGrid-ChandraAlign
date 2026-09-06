import cv2
import numpy as np

from registration import register_images


def synthetic_pair():
    rng = np.random.default_rng(42)
    image = np.full((620, 920), 105, np.uint8)
    for _ in range(80):
        x, y = rng.integers(30, 890), rng.integers(30, 590)
        r = int(rng.integers(5, 38))
        cv2.circle(image, (int(x), int(y)), r, int(rng.integers(30, 210)), 2)
        cv2.circle(image, (int(x-r/4), int(y-r/4)), max(2, r//5), 230, -1)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    matrix = cv2.getRotationMatrix2D((460, 310), 2.4, 1.025)
    matrix[:, 2] += (18, -11)
    moved = cv2.warpAffine(image, matrix, (920, 620))
    return moved, image


if __name__ == "__main__":
    source, reference = synthetic_pair()
    result = register_images(source, reference)
    assert result.metrics["accepted_matches"] >= 8
    assert result.metrics["rmse_px"] < 3.0
    assert len(result.tie_points) == result.metrics["candidate_matches"]
    assert sum(p["status"] == "accepted" for p in result.tie_points) == result.metrics["accepted_matches"]
    assert sum(p["status"] == "rejected" for p in result.tie_points) == result.metrics["rejected_matches"]
    assert all(p["confidence"] is None for p in result.tie_points)
    assert result.metrics["registration_method"] == "Homography (USAC_MAGSAC)"
    print(result.metrics)
