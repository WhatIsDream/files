# HW1 Panoramic Stitching
Install: `pip install numpy opencv-python` (opencv >= 4.4 for cv2.SIFT_create)
1. `python synthetic_tests.py --out out`   (DLT + warp synthetic checks)
2. `python run_core.py --set1 a1.jpg b1.jpg --set2 a2.jpg b2.jpg --out out`
Image A is the reference; H maps B -> A. Library use: I/O, SIFT, drawing, (optional) detector alternatives only.
