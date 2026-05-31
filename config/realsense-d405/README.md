# RealSense D405 — config files

| File | Purpose |
|------|---------|
| `charuco.png` | ChArUco calibration target — print at **1:1**, no scaling/fit-to-page |
| `movit_calib.yaml` | Hand-eye calibration output (`effector_wrt_world` / `object_wrt_sensor` pairs) |
| `d405.json` | RealSense Viewer advanced-mode preset for the D405 |

## ChArUco board dimensions

When printed at 1:1, `charuco.png` is a **5 × 7** ChArUco board with:

- **Square length:** `0.028 m` (28 mm) per checkerboard square
- **Long side:** `0.198 m` (7 squares ≈ 0.196 m plus printing margin)

Pass these to whichever tool consumes the board (e.g. `easy_handeye2`, OpenCV
`cv2.aruco.CharucoBoard`, MoveIt calibration). Square length is the only
dimension most APIs need — the long/short totals are listed for sanity-checking
the print after it comes out of the printer (measure a square, then measure the
full board edge; both must match the values above).

After printing, **verify with a ruler**: a square that prints at anything other
than 28 mm will silently bias every hand-eye result in `movit_calib.yaml`.
