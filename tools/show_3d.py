""" Original Author: Haoqiang Fan, Added object orientation control & preset views """
import numpy as np
import ctypes as ct
import sys
import os
from pathlib import Path
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
showsz = 800
changed = True

# ===================== View & object orientation parameters =====================
view_xangle = np.deg2rad(-20)   # Camera pitch (radians), negative = look down
view_yangle = np.deg2rad(0)     # Camera yaw (radians)
object_yaw = np.deg2rad(-30)    # Object horizontal rotation (Y-axis), default -30deg = front-facing oblique view
zoom = 1.0                       # Zoom level

# Trackbar range configuration
XANGLE_TRACK_MAX = 180    # Pitch: -90° ~ 90°
YANGLE_TRACK_MAX = 360    # View yaw: -180° ~ 180°
OBJ_YAW_TRACK_MAX = 360   # Object yaw: -180° ~ 180°
ZOOM_TRACK_MAX = 300      # Zoom: 10% ~ 300%

ROT_SENSITIVITY = 0.006   # Mouse drag rotation sensitivity
ZOOM_SENSITIVITY = 0.1    # Mouse wheel zoom sensitivity
left_button_down = False
last_mouse_x, last_mouse_y = 0, 0


# ===================== Trackbar callback functions =====================
def on_trackbar_xangle(val):
    global view_xangle, changed
    deg = val - XANGLE_TRACK_MAX // 2
    view_xangle = np.deg2rad(deg)
    changed = True

def on_trackbar_yangle(val):
    global view_yangle, changed
    deg = val - YANGLE_TRACK_MAX // 2
    view_yangle = np.deg2rad(deg)
    changed = True

def on_trackbar_obj_yaw(val):
    global object_yaw, changed
    deg = val - OBJ_YAW_TRACK_MAX // 2
    object_yaw = np.deg2rad(deg)
    changed = True

def on_trackbar_zoom(val):
    global zoom, changed
    zoom = max(val / 100.0, 0.1)
    changed = True


# ===================== Mouse interaction callback =====================
def onmouse(event, x, y, flags, param):
    global changed, left_button_down, last_mouse_x, last_mouse_y
    global view_xangle, view_yangle, zoom
    import cv2

    if event == cv2.EVENT_LBUTTONDOWN:
        left_button_down = True
        last_mouse_x = x
        last_mouse_y = y

    elif event == cv2.EVENT_LBUTTONUP:
        left_button_down = False

    elif event == cv2.EVENT_MOUSEMOVE and left_button_down:
        dx = x - last_mouse_x
        dy = y - last_mouse_y
        view_yangle += dx * ROT_SENSITIVITY
        view_xangle += dy * ROT_SENSITIVITY
        last_mouse_x = x
        last_mouse_y = y
        
        # Sync trackbar positions
        cv2.setTrackbarPos('Pitch (deg)', 'show3d', 
                          int(np.rad2deg(view_xangle)) + XANGLE_TRACK_MAX//2)
        cv2.setTrackbarPos('View Yaw (deg)', 'show3d', 
                          int(np.rad2deg(view_yangle)) + YANGLE_TRACK_MAX//2)
        changed = True

    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom *= (1 + ZOOM_SENSITIVITY)
        else:
            zoom /= (1 + ZOOM_SENSITIVITY)
        cv2.setTrackbarPos('Zoom (%)', 'show3d', int(zoom * 100))
        changed = True


_gui_ready = False
dll = None


def _ensure_gui():
    """Lazy-init the OpenCV window so import does not crash without DISPLAY."""
    global _gui_ready, dll
    if _gui_ready:
        return
    import cv2

    cv2.namedWindow('show3d')
    cv2.moveWindow('show3d', 0, 0)
    cv2.setMouseCallback('show3d', onmouse)
    cv2.createTrackbar('Pitch (deg)', 'show3d',
                       int(np.rad2deg(view_xangle)) + XANGLE_TRACK_MAX//2,
                       XANGLE_TRACK_MAX, on_trackbar_xangle)
    cv2.createTrackbar('View Yaw (deg)', 'show3d',
                       int(np.rad2deg(view_yangle)) + YANGLE_TRACK_MAX//2,
                       YANGLE_TRACK_MAX, on_trackbar_yangle)
    cv2.createTrackbar('Obj Yaw (deg)', 'show3d',
                       int(np.rad2deg(object_yaw)) + OBJ_YAW_TRACK_MAX//2,
                       OBJ_YAW_TRACK_MAX, on_trackbar_obj_yaw)
    cv2.createTrackbar('Zoom (%)', 'show3d',
                       int(zoom * 100),
                       ZOOM_TRACK_MAX, on_trackbar_zoom)
    dll = np.ctypeslib.load_library('render_ball', '.')
    _gui_ready = True


# ===================== Load single PCD file =====================
def load_single_pcd(pcd_path):
    """Load PCD file, return coordinates and colors (auto filter invalid points)"""
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")
    
    pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points, dtype='float32')
    
    # Filter NaN/Inf invalid points (required for real-world data like KITTI)
    valid_mask = ~np.isnan(xyz).any(axis=1) & ~np.isinf(xyz).any(axis=1)
    xyz = xyz[valid_mask]
    
    # Read color information (if available)
    colors = None
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype='float32')
        colors = colors[valid_mask]
        colors = colors * 255.0  # Convert from 0-1 range to 0-255 for renderer
    
    return xyz, colors


def showpoints(xyz, c_gt=None, c_pred=None, waittime=0, showrot=False, magnifyBlue=0, freezerot=False,
               background=(255, 255, 255), normalizecolor=True, ballradius=10):
    global zoom, changed, view_xangle, view_yangle, object_yaw, dll
    import cv2

    _ensure_gui()

    xyz = xyz - xyz.mean(axis=0)
    radius = ((xyz ** 2).sum(axis=-1) ** 0.5).max()
    xyz /= (radius * 2.2) / showsz
    
    if c_gt is None:
        c0 = np.zeros((len(xyz),), dtype='float32') + 198
        c1 = np.zeros((len(xyz),), dtype='float32') + 203
        c2 = np.zeros((len(xyz),), dtype='float32') + 210
    else:
        c0 = c_gt[:, 0]
        c1 = c_gt[:, 1]
        c2 = c_gt[:, 2]

    if normalizecolor:
        c0 /= (c0.max() + 1e-14) / 255.0
        c1 /= (c1.max() + 1e-14) / 255.0
        c2 /= (c2.max() + 1e-14) / 255.0

    c0 = np.require(c0, 'float32', 'C')
    c1 = np.require(c1, 'float32', 'C')
    c2 = np.require(c2, 'float32', 'C')

    show = np.zeros((showsz, showsz, 3), dtype='uint8')

    def render():
        nonlocal c0, c1, c2
        # ========== Step 1: Object self rotation (adjust front direction) ==========
        obj_rot = np.array([
            [np.cos(object_yaw), 0.0, -np.sin(object_yaw)],
            [0.0, 1.0, 0.0],
            [np.sin(object_yaw), 0.0, np.cos(object_yaw)],
        ])
        xyz_rot = xyz.dot(obj_rot)

        # ========== Step 2: Camera view rotation ==========
        if not freezerot:
            xangle = view_xangle
            yangle = view_yangle
        else:
            xangle = 0.0
            yangle = 0.0

        view_rot = np.eye(3)
        # Pitch (rotate around X axis)
        view_rot = view_rot.dot(np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(xangle), -np.sin(xangle)],
            [0.0, np.sin(xangle), np.cos(xangle)],
        ]))
        # Yaw (rotate around Y axis)
        view_rot = view_rot.dot(np.array([
            [np.cos(yangle), 0.0, -np.sin(yangle)],
            [0.0, 1.0, 0.0],
            [np.sin(yangle), 0.0, np.cos(yangle)],
        ]))
        view_rot *= zoom

        nxyz = xyz_rot.dot(view_rot) + [showsz / 2, showsz / 2, 0]
        nxyz[:, 1] = showsz - nxyz[:, 1]
        ixyz = nxyz.astype('int32')

        show[:] = background
        dll.render_ball(
            ct.c_int(show.shape[0]),
            ct.c_int(show.shape[1]),
            show.ctypes.data_as(ct.c_void_p),
            ct.c_int(ixyz.shape[0]),
            ixyz.ctypes.data_as(ct.c_void_p),
            c0.ctypes.data_as(ct.c_void_p),
            c1.ctypes.data_as(ct.c_void_p),
            c2.ctypes.data_as(ct.c_void_p),
            ct.c_int(ballradius)
        )

        # ===================== Display current parameters =====================
        x_deg = int(np.rad2deg(view_xangle))
        y_deg = int(np.rad2deg(view_yangle))
        obj_deg = int(np.rad2deg(object_yaw))
        zoom_pct = int(zoom * 100)
        info_lines = [
            f"Pitch:     {x_deg:4d} deg",
            f"View Yaw:  {y_deg:4d} deg",
            f"Obj Yaw:   {obj_deg:4d} deg",
            f"Zoom:      {zoom_pct:4d} %"
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(show, line, (20, 35 + i * 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

        if magnifyBlue > 0:
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=0))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=0))
            show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], 1, axis=1))
            if magnifyBlue >= 2:
                show[:, :, 0] = np.maximum(show[:, :, 0], np.roll(show[:, :, 0], -1, axis=1))
        
        if showrot:
            text_color = (0, 0, 255)
            cv2.putText(show, 'xangle %d' % (int(xangle / np.pi * 180)), (30, showsz - 30), 0, 0.5, text_color)
            cv2.putText(show, 'yangle %d' % (int(yangle / np.pi * 180)), (30, showsz - 50), 0, 0.5, text_color)
            cv2.putText(show, 'zoom %d%%' % (int(zoom * 100)), (30, showsz - 70), 0, 0.5, text_color)

    changed = True
    while True:
        if changed:
            render()
            changed = False
        cv2.imshow('show3d', show)
        
        if waittime == 0:
            cmd = cv2.waitKey(10) % 256
        else:
            cmd = cv2.waitKey(waittime) % 256
        
        if cmd == ord('q'):
            break
        elif cmd == ord('Q'):
            sys.exit(0)

        # ========== Preset view shortcuts ==========
        # Press 1: Classic 45-degree oblique front view (recommended for papers)
        if cmd == ord('1'):
            object_yaw = np.deg2rad(-30)
            view_xangle = np.deg2rad(-22)
            view_yangle = np.deg2rad(0)
            zoom = 1.05
            # Sync trackbars
            cv2.setTrackbarPos('Obj Yaw (deg)', 'show3d', -30 + OBJ_YAW_TRACK_MAX//2)
            cv2.setTrackbarPos('Pitch (deg)', 'show3d', -22 + XANGLE_TRACK_MAX//2)
            cv2.setTrackbarPos('View Yaw (deg)', 'show3d', 0 + YANGLE_TRACK_MAX//2)
            cv2.setTrackbarPos('Zoom (%)', 'show3d', 105)
            changed = True
            print("Switched to Preset 1: 45deg oblique front view")

        # Press 2: Direct front view (show_3d preset 2)
        if cmd == ord('2'):
            object_yaw = np.deg2rad(-90)
            view_xangle = np.deg2rad(-15)
            view_yangle = np.deg2rad(0)
            zoom = 1.0
            cv2.setTrackbarPos('Obj Yaw (deg)', 'show3d', -90 + OBJ_YAW_TRACK_MAX//2)
            cv2.setTrackbarPos('Pitch (deg)', 'show3d', -15 + XANGLE_TRACK_MAX//2)
            cv2.setTrackbarPos('View Yaw (deg)', 'show3d', 0 + YANGLE_TRACK_MAX//2)
            cv2.setTrackbarPos('Zoom (%)', 'show3d', 100)
            changed = True
            print("Switched to Preset 2: Direct front view")

        # Press g: Manually input exact values
        if cmd == ord('g'):
            print("\n--- Manual view parameter input ---")
            try:
                obj_deg = float(input("Object yaw (deg, use -90 for front-facing): "))
                x_deg = float(input("Pitch (deg, negative for top-down): "))
                y_deg = float(input("View yaw (deg): "))
                zoom_pct = float(input("Zoom percentage: "))
                
                object_yaw = np.deg2rad(obj_deg)
                view_xangle = np.deg2rad(x_deg)
                view_yangle = np.deg2rad(y_deg)
                zoom = max(zoom_pct / 100.0, 0.1)
                
                cv2.setTrackbarPos('Obj Yaw (deg)', 'show3d', int(obj_deg) + OBJ_YAW_TRACK_MAX//2)
                cv2.setTrackbarPos('Pitch (deg)', 'show3d', int(x_deg) + XANGLE_TRACK_MAX//2)
                cv2.setTrackbarPos('View Yaw (deg)', 'show3d', int(y_deg) + YANGLE_TRACK_MAX//2)
                cv2.setTrackbarPos('Zoom (%)', 'show3d', int(zoom_pct))
                changed = True
                print("Parameters updated")
            except:
                print("Invalid input, cancelled")

        # Switch GT/Pred colors
        if cmd == ord('t') or cmd == ord('p'):
            if cmd == ord('t'):
                if c_gt is None:
                    c0[:] = 255
                    c1[:] = 255
                    c2[:] = 255
                else:
                    c0 = c_gt[:, 0].copy()
                    c1 = c_gt[:, 1].copy()
                    c2 = c_gt[:, 2].copy()
            else:
                if c_pred is None:
                    c0[:] = 255
                    c1[:] = 255
                    c2[:] = 255
                else:
                    c0 = c_pred[:, 0].copy()
                    c1 = c_pred[:, 1].copy()
                    c2 = c_pred[:, 2].copy()
            
            if normalizecolor:
                c0 /= (c0.max() + 1e-14) / 255.0
                c1 /= (c1.max() + 1e-14) / 255.0
                c2 /= (c2.max() + 1e-14) / 255.0
            changed = True

        # Zoom controls
        if cmd == ord('n'):
            zoom *= 1.1
            cv2.setTrackbarPos('Zoom (%)', 'show3d', int(zoom * 100))
            changed = True
        elif cmd == ord('m'):
            zoom /= 1.1
            cv2.setTrackbarPos('Zoom (%)', 'show3d', int(zoom * 100))
            changed = True
        elif cmd == ord('r'):
            # Reset all view parameters
            object_yaw = np.deg2rad(-30)
            view_xangle = np.deg2rad(-20)
            view_yangle = np.deg2rad(0)
            zoom = 1.0
            cv2.setTrackbarPos('Obj Yaw (deg)', 'show3d', -30 + OBJ_YAW_TRACK_MAX//2)
            cv2.setTrackbarPos('Pitch (deg)', 'show3d', -20 + XANGLE_TRACK_MAX//2)
            cv2.setTrackbarPos('View Yaw (deg)', 'show3d', 0 + YANGLE_TRACK_MAX//2)
            cv2.setTrackbarPos('Zoom (%)', 'show3d', 100)
            changed = True
        elif cmd == ord('s'):
            cv2.imwrite('show3d.png', show)
            print("Screenshot saved as show3d.png")
        if waittime != 0:
            break
    return cmd


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Single PCD visualization (with object orientation control & preset views)')
    parser.add_argument('--pcd_path', type=str, required=True, help='Path to PCD file')
    parser.add_argument('--ballradius', type=int, default=4, help='Sphere rendering radius')
    parser.add_argument('--white_bg', action='store_true', default=True, help='Use white background')
    parser.add_argument(
        '--headless',
        action='store_true',
        help='headless PNG export (required on SSH / no DISPLAY)',
    )
    parser.add_argument(
        '--export_png',
        type=str,
        default=None,
        help='headless PNG path; default same stem as the pcd',
    )
    parser.add_argument(
        '--view_preset',
        type=str,
        default='front',
        choices=['default', 'oblique', 'front', 'upright'],
        help='front/upright=default; oblique/default=oblique view',
    )
    opt = parser.parse_args()

    xyz, colors = load_single_pcd(opt.pcd_path)
    print(f"Successfully loaded point cloud, valid points: {xyz.shape[0]}")

    use_headless = opt.headless or not os.environ.get('DISPLAY')
    if use_headless:
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from types import SimpleNamespace

        presets = {
            'oblique': SimpleNamespace(
                object_yaw_deg=-30.0, view_xangle_deg=-22.0, view_yangle_deg=0.0, zoom=1.05, pca_align=False,
            ),
            'front': SimpleNamespace(
                object_yaw_deg=-90.0, view_xangle_deg=-15.0, view_yangle_deg=0.0, zoom=1.0, pca_align=False,
            ),
        }
        presets['upright'] = presets['front']
        presets['default'] = presets['oblique']
        view = presets[opt.view_preset]
        out_png = Path(opt.export_png) if opt.export_png else Path(opt.pcd_path).with_suffix('.png')
        if colors is None:
            colors = np.tile(np.array([180.0, 180.0, 180.0], dtype=np.float32), (len(xyz), 1))
        # reuse visualize_crops3d headless render; no X11 cv2 window
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from visualize_crops3d import render_png_headless

        render_png_headless(
            xyz,
            colors,
            out_png,
            view,
            ballradius=opt.ballradius,
            white_bg=opt.white_bg,
            normalizecolor=False,
        )
        print(f"No DISPLAY; wrote PNG: {out_png}")
        print("Tip: you can also open gt.png / pred.png / compare.png from visualize_crops3d.py")
        raise SystemExit(0)

    print("\nControls:")
    print("  Left click + drag: Rotate camera view")
    print("  Mouse wheel: Zoom in/out")
    print("  Side trackbars: Precise parameter adjustment")
    print("  Obj Yaw trackbar: Adjust object front direction")
    print("  Press 1: Preset 1 - 45deg oblique front view (recommended)")
    print("  Press 2: Preset 2 - Direct front view")
    print("  Press r: Reset to default view")
    print("  Press s: Save screenshot")
    print("  Press q: Quit\n")

    bg = (255, 255, 255) if opt.white_bg else (0, 0, 0)

    showpoints(
        xyz,
        c_gt=colors,
        c_pred=colors,
        waittime=0,
        showrot=False,
        magnifyBlue=0,
        freezerot=False,
        background=bg,
        normalizecolor=True,
        ballradius=opt.ballradius
    )