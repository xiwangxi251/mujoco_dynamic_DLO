"""Project target and allowed material nodes as compact camera markers."""
import numpy as np
import mujoco


def project(scene, point, width, height):
    camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
    forward = camera.forward / np.linalg.norm(camera.forward)
    up = camera.up / np.linalg.norm(camera.up)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    relative = np.asarray(point)-camera.pos
    depth = float(relative @ forward)
    frustum_height = camera.frustum_top-camera.frustum_bottom
    frustum_width = camera.frustum_width or frustum_height*width/height
    horizontal = float(relative @ right)
    vertical = float(relative @ up)
    if not camera.orthographic:
        horizontal *= camera.frustum_near/max(abs(depth), 1e-9)
        vertical *= camera.frustum_near/max(abs(depth), 1e-9)
    x = width*(.5+(horizontal-camera.frustum_center)/frustum_width)
    y = height*(camera.frustum_top-vertical)/frustum_height
    return float(x), float(y), depth


def mixed_label(draw, position, label, font, latin_font, color):
    """Draw CJK and ASCII using faces which actually contain their glyphs."""
    runs = []
    for character in label:
        face = latin_font if character.isascii() else font
        if runs and runs[-1][1] is face:
            runs[-1] = (runs[-1][0]+character,face)
        else:
            runs.append((character,face))
    boxes = [draw.textbbox((0,0), text,font=face) for text,face in runs]
    height = max(b[3]-b[1] for b in boxes)
    x,y = position
    for (text,face),box in zip(runs,boxes):
        draw.text((x,y+(height-box[3]+box[1])*.5),text,font=face,fill=color,anchor='lt')
        x += draw.textlength(text,font=face)
    return x-position[0],height


def clip_segment(first, second, width=1280, height=960):
    """Clip projected lines to the scene below the HUD (Liang-Barsky)."""
    x0,y0 = first
    dx,dy = np.asarray(second)-np.asarray(first)
    lo,hi = 0.,1.
    for p,q in [(-dx,x0-16),(dx,width-16-x0),(-dy,y0-165),(dy,height-16-y0)]:
        if abs(p) < 1e-12:
            if q < 0:
                return None
        else:
            r = q/p
            if p < 0:
                lo = max(lo,r)
            else:
                hi = min(hi,r)
            if lo > hi:
                return None
    return [(x0+lo*dx,y0+lo*dy),(x0+hi*dx,y0+hi*dy)]


def annotate_tolerance(draw, scene, positions, indices, font, latin_font, width=1280, height=960):
    projected = [project(scene, xyz,width,height) for xyz in positions]
    green = '#64f2b1'
    for a,b in zip(projected,projected[1:]):
        if a[2] <= 0 or b[2] <= 0:
            continue
        segment = clip_segment(a[:2],b[:2],width,height)
        if segment:
            draw.line(segment,fill='#102237',width=12)
            draw.line(segment,fill=green,width=6)
    visible = []
    for node,(x,y,depth) in zip(indices,projected):
        onscreen = depth > 0 and 16 <= x <= width-16 and 165 <= y <= height-16
        visible.append(dict(node=int(node),pixel_xy=[x,y],depth_m=depth,onscreen=bool(onscreen)))
        if onscreen:
            draw.ellipse((x-7,y-7,x+7,y+7),fill=green,outline='#102237',width=2)
            if node in (indices[0],indices[-1]):
                text = str(node)
                bounds = draw.textbbox((0,0),text,font=latin_font)
                tx = np.clip(x+13,18,width-(bounds[2]-bounds[0])-18)
                ty = np.clip(y+10,170,height-45)
                draw.text((tx,ty),text,font=latin_font,fill=green,stroke_width=3,stroke_fill='#102237',anchor='lt')
    return visible


def annotate(draw, scene, xyz, node, font, width=1280, height=960, latin_font=None):
    x, y, depth = project(scene, xyz, width, height)
    onscreen = depth > 0 and 20 <= x <= width-20 and 165 <= y <= height-20
    if onscreen:
        draw.ellipse((x-19, y-19, x+19, y+19), outline='#102237', width=8)
        draw.ellipse((x-19, y-19, x+19, y+19), outline='#ffe553', width=4)
        draw.ellipse((x-3, y-3, x+3, y+3), fill='#ffe553')
        label = f'目标节点 {node}'
    else:
        if depth <= 0:
            label = f'目标节点 {node}（相机后方）'
        else:
            label = f'目标节点 {node}（视野外）'
    return dict(pixel_xy=[x,y], depth_m=depth, onscreen=bool(onscreen), label=label)
