import mglyph as mg
import numpy as np
import random
import numbers
from typing import Callable
import matplotlib.colors
import colorsys
import string

class Glyph:
    """
    Base class for all Glyphs. Handles visual feature processing.
    """
    def __init__(self) -> None:
        pass

    def random_color(self) -> tuple:
        return tuple(np.random.uniform(0, 1, 3)) + (1,)
    
    def random_gray_color(self) -> tuple:
        val = np.random.uniform(0, 1)
        return (val, val, val, 1)

    def random_width(self, min_w: float = 3, max_w: float = 150) -> str:
        return f"{random.uniform(min_w, max_w)}p"

    def _process_visual_features(self, x: float, canvas: mg.Canvas, 
                                      base_fill_color, 
                                      size, displacement, opacity, hue, saturation, rotation, border):
        # Size
        if size:
            # Lerp from 0.2 to full size
            radius = mg.lerp(x, 0.2, canvas.ysize / 3)
        else:
            radius = canvas.ysize / 3

        # Color Processing
        if isinstance(base_fill_color, str):
            if base_fill_color.startswith('#'):
                base_rgb = matplotlib.colors.hex2color(base_fill_color)
            else:
                base_rgb = matplotlib.colors.to_rgb(base_fill_color)
        else:
            base_rgb = base_fill_color[:3]

        r, g, b = base_rgb
        h_base, s_base, v_base = colorsys.rgb_to_hsv(r, g, b)
        
        # Hue / Saturation
        final_h = mg.lerp(x, 0.0, 1.0) if hue else h_base
        final_s = mg.lerp(x, 0.0, 1.0) if saturation else s_base
        
        r_new, g_new, b_new = colorsys.hsv_to_rgb(final_h, final_s, v_base)
        
        # Opacity
        alpha = mg.lerp(x, 0.1, 1.0) if opacity else 1.0
        final_color = (r_new, g_new, b_new, alpha)

        # Displacement
        if displacement:
            radius_disp = mg.lerp(x, 0.0, 0.5)
            angle = np.random.randint(0, 360)
            canvas.tr.translate(*mg.orbit(canvas.center, angle, radius_disp))
            
        # Rotation
        if isinstance(rotation, bool) and rotation:
            canvas.tr.rotate(mg.lerp(x, 0, 360))
        elif isinstance(rotation, numbers.Number):
            canvas.tr.rotate(rotation)

        # Border Width
        linewidth = "30p"
        if border:
            linewidth = f"{mg.lerp(x, 0, 75)}p"

        return radius, final_color, linewidth

    # Drawing Primitives
    def draw_quadratic_bezier(self, canvas: mg.Canvas, p0: tuple, p1: tuple, p2: tuple, **kwargs):
        points = []
        steps = 20
        for i in range(steps + 1):
            t = i / steps
            x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
            y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
            points.append((x, y))
        
        color = kwargs.get('color', 'black')
        width = kwargs.get('width', '20p')
        style = kwargs.get('style', 'stroke')
        canvas.polygon(points, color=color, width=width, style=style, closed=False, linecap='round')

    def draw_line_strip(self, canvas: mg.Canvas, points: list, **kwargs):
        if len(points) < 2: return
        color = kwargs.get('color', 'black')
        width = kwargs.get('width', '20p')
        style = kwargs.get('style', 'stroke')
        canvas.polygon(points, color=color, width=width, style=style, closed=False, linecap='round')

    # DYNAMIC STYLE DISPATCHER
    def __getattr__(self, name: str):
        shape_name = self.__class__.__name__.lower()
        if not name.endswith(f"_{shape_name}"):
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
        
        style = name.replace(f"_{shape_name}", "")
        if not hasattr(self, shape_name):
             raise AttributeError(f"Drawing method '{shape_name}' not defined.")
        
        draw_method = getattr(self, shape_name)

        def style_wrapper(**kwargs):
            # default params
            params = {
                'size': False, 'displacement': False, 'opacity': False, 'hue': False, 
                'saturation': False, 'rotation': False, 'border': False, 'linewidth': '30p'
            }
            
            # spply style rules
            
            # default
            if style == 'default': 
                pass
            # combinations
            elif style == 'combined':
                params.update({'size': True, 'displacement': True, 'opacity': True, 'hue': True, 'saturation': True, 'rotation': True, 'border': True, 'linewidth': self.random_width()})
            elif style == 'variational':
                params.update({
                    'size': np.random.choice([True, False]), 'displacement': np.random.choice([True, False]),
                    'opacity': np.random.choice([True, False]), 'saturation': np.random.choice([True, False]),
                    'border': np.random.choice([True, False]), 'hue': False, 'rotation': False
                })
            elif style == 'scaled':
                params.update({'displacement': True, 'opacity': True, 'saturation': True, 'rotation': True, 'border': True})
            # border
            elif style == 'border':
                params.update({'border': True})
                if 'bordercolor' not in kwargs: kwargs['bordercolor'] = self.random_color()
            elif style == 'thin':
                params.update({'linewidth': self.random_width(1, 5)})
                if 'bordercolor' not in kwargs: kwargs['bordercolor'] = self.random_color()
                if 'fillcolor' not in kwargs: kwargs['fillcolor'] = self.random_color()
            elif style == 'thick':
                params.update({'linewidth': self.random_width(50, 100)})
                if 'bordercolor' not in kwargs: kwargs['bordercolor'] = self.random_color()
                if 'fillcolor' not in kwargs: kwargs['fillcolor'] = self.random_color()
            elif style == 'constant_border':
                params.update({'linewidth': '50p'})
                if 'bordercolor' not in kwargs: kwargs['bordercolor'] = self.random_color()
            # color
            elif style == 'grayscale':
                for color in ['border', 'petal_', 'center_', 'fg_', 'fill', 'flame_', 'fill_', 'iris_', 'target_', 'eclipse_', 'paint_', 'plate_', 'face_', 'sun_', 'cloud_', 'skin_', 'sand_', 'cover_', '']:
                    params.update({color + 'color': self.random_gray_color()})
            elif style == 'random_border':
                params.update({'linewidth': self.random_width(10, 100)})
                if 'bordercolor' not in kwargs: kwargs['bordercolor'] = self.random_color()
            elif style == 'random_bg':
                if 'fillcolor' not in kwargs: kwargs['fillcolor'] = self.random_color()
            elif style == 'random_color':
                if 'bordercolor' not in kwargs: kwargs['bordercolor'] = self.random_color()
                if 'fillcolor' not in kwargs: kwargs['fillcolor'] = self.random_color()
            # one scaled parameter
            elif style in ['size', 'displacement', 'rotation', 'opacity', 'hue', 'saturation']:
                for k in params: 
                    if isinstance(params[k], bool): params[k] = False
                params[style] = True
            
            # Merge User Kwargs (e.g. char='A')
            params.update(kwargs)
            return lambda x, c: draw_method(x, c, **params)

        return style_wrapper

# letter class

# scaling by size → set size to True 
class Letter(Glyph):
    def letter(self, x: float, canvas: mg.Canvas, char=None, text=None, color='teal', fillcolor='teal', **kwargs) -> None:
        radius, final_color, _ = self._process_visual_features(
            x, canvas, fillcolor, True, kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        # char is empty and text is not empty
        if not text is None and char is None:
            char = text
        # both are empty
        if char is None and text is None:
            char = random.choice(string.ascii_uppercase)
        if kwargs.get('bordercolor'):
            border = mg.lerp(x, 0, 100) if kwargs.get('border') else 100
            canvas.text(char, (0, 0), size=f"{radius * (1500 + border)}p", color=kwargs.get('bordercolor'))
        canvas.text(char, (0, 0), size=f"{radius * 1500}p", color=final_color)

    # specific overrides to handle char randomness logic
    def default_letter(self, **kwargs) -> Callable:
        return lambda x, c: self.letter(x, c, color='teal', size=True, **kwargs)
    def combined_letter(self, **kwargs) -> Callable: 
        return lambda x, c: self.letter(x, c, char=kwargs.get('char', None), color=self.random_color(), size=True, **kwargs)
    def variational_letter(self, **kwargs) -> Callable:
        return lambda x, c: self.letter(x, c, char=kwargs.get('char', None), color=self.random_color(), size=np.random.choice([True, False]), displacement=np.random.choice([True, False]), opacity=np.random.choice([True, False]), saturation=np.random.choice([True, False]), rotation=False, **kwargs)
    def grayscale_letter(self, **kwargs):
        return lambda x, c: self.letter(x, c, color=self.random_gray_color(), size=True, **kwargs)


# concrete classes

# scaling by size → set size to True 
class Star(Glyph):
    def star(self, x: float, canvas: mg.Canvas, bordercolor=None, fillcolor='goldenrod', linewidth=None, **kwargs) -> None:        
        radius, final_fill, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, True, kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else (linewidth or '30p')
        vertices = []
        for segment in range(5):
            vertices.append(mg.orbit(canvas.center, segment*2*np.pi/5, radius))
            vertices.append(mg.orbit(canvas.center, (segment+0.5)*2*np.pi/5, np.cos(2*np.pi/5)/np.cos(np.pi/5)*radius))
        canvas.polygon(vertices, linecap='round', style='fill', color=final_fill)
        if bordercolor: canvas.polygon(vertices, width=lw, linecap='round', style='stroke', color=bordercolor)
        
# scaling by size → set size to True
class Square(Glyph):
    def square(self, x: float, canvas: mg.Canvas, bordercolor=None, fillcolor='cornflowerblue', linewidth=None, **kwargs) -> None:
        radius, final_fill, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, True, kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else (linewidth or '30p')
        half_w = radius 
        canvas.rect((-half_w, -half_w), (half_w, half_w), color=final_fill, style='fill')
        if bordercolor: canvas.rect((-half_w, -half_w), (half_w, half_w), color=bordercolor, style='stroke', width=lw)
        
# scaling by size → set size to True
class Circle(Glyph):
    def circle(self, x: float, canvas: mg.Canvas, bordercolor=None, fillcolor='limegreen', linewidth=None, **kwargs) -> None:
        radius, final_fill, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, True, kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else (linewidth or '30p')
        canvas.circle((0,0), radius, color=final_fill, style='fill')
        if bordercolor: canvas.circle((0,0), radius, color=bordercolor, style='stroke', width=lw)

# scaling by size → set size to True
class VUT(Glyph):
    @staticmethod
    def _get_circle_quarter(x_offset=0, y_offset=0, quarter=1, points=20, radius=1):
        start_angle, end_angle = (quarter - 1) * (np.pi / 2), quarter * (np.pi / 2)
        return [(x_offset + radius * np.cos(start_angle + (end_angle - start_angle) * i / points), y_offset + radius * np.sin(start_angle + (end_angle - start_angle) * i / points)) for i in range(points + 1)]

    def vut(self, x: float, canvas: mg.Canvas, mode='stroke', fg_color='#e4002b', bg_color=None, linewidth="30p", **kwargs) -> None:
        global_scale_radius, final_fg, border_width_calc = self._process_visual_features(
            x, canvas, fg_color, True, kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        final_bg = None
        if bg_color:
            if kwargs.get('opacity'):
                alpha = final_fg[3]
                if isinstance(bg_color, str) and bg_color.startswith('#'):
                    b_rgb = matplotlib.colors.hex2color(bg_color)
                    final_bg = (b_rgb[0], b_rgb[1], b_rgb[2], alpha)
                else: final_bg = bg_color[:3] + (alpha,)
            else: final_bg = bg_color

        curve_radius = 0.25 * (x / 100)
        circle_arc = self._get_circle_quarter(quarter=3, radius=curve_radius, x_offset=mg.lerp(x, canvas.xcenter, canvas.xcenter + 0.35), points=50)
        raw_path = [
            (mg.lerp(x, canvas.xcenter, canvas.xcenter - 0.1), mg.lerp(x, canvas.ycenter, canvas.ytop + 0.5)),
            (mg.lerp(x, canvas.xcenter, canvas.xleft + 0.2),   mg.lerp(x, canvas.ycenter, canvas.ytop + 0.5)),
            (mg.lerp(x, canvas.xcenter, canvas.xleft + 0.2),   mg.lerp(x, canvas.ycenter, canvas.ytop + 0.25)),
            (mg.lerp(x, canvas.xcenter, canvas.xcenter - 0.1), mg.lerp(x, canvas.ycenter, canvas.ytop + 0.25)),
            (mg.lerp(x, canvas.xcenter, canvas.xcenter - 0.1), mg.lerp(x, canvas.ycenter, canvas.ybottom - 0.25)),
            (mg.lerp(x, canvas.xcenter, canvas.xcenter + 0.1), mg.lerp(x, canvas.ycenter, canvas.ybottom - 0.25)),
            (mg.lerp(x, canvas.xcenter, canvas.xcenter + 0.1), canvas.ycenter),
            *circle_arc,
            (mg.lerp(x, canvas.xcenter, canvas.xright - 0.2), mg.lerp(x, canvas.ycenter, canvas.ycenter - 0.25)),
            (mg.lerp(x, canvas.xcenter, canvas.xright - 0.2), mg.lerp(x, canvas.ycenter, canvas.ytop + 0.5)),
        ]
        base_size = canvas.ysize / 2 if canvas.ysize else 1.0
        scale_factor = 1.0
        scaled_path = [(p[0] * scale_factor, p[1] * scale_factor) for p in raw_path]

        if mode == 'fill':
            bg_size = 2.0 * scale_factor
            canvas.rect((-bg_size, -bg_size), (bg_size, bg_size), color=final_bg, style='fill')
            canvas.polygon(scaled_path, width=lw, style="fill", color=final_fg)
        else: canvas.polygon(scaled_path, width=lw, style="stroke", color=final_fg)

    def variational_vut(self, **kwargs):
        mode = random.choice(['stroke'])
        bg = self.random_color() if mode == 'fill' else None
        return lambda x, c: self.vut(x, c, mode=mode, fg_color=self.random_color(), bg_color=bg, size=np.random.choice([True, False]), displacement=np.random.choice([True, False]), opacity=np.random.choice([True, False]), saturation=np.random.choice([True, False]), border=np.random.choice([True, False]), rotation=False, **kwargs)
        
# scaling by size → set size to True
class Flower(Glyph):
    def flower(self, x: float, canvas: mg.Canvas, petal_color='#D94878', center_color='#74CEF7', bordercolor=None, linewidth='5p', **kwargs) -> None:
        radius, final_petal, border_width_calc = self._process_visual_features(
            x, canvas, petal_color, True, kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        final_center = center_color
        if kwargs.get('opacity'):
            alpha = final_petal[3]
            if isinstance(center_color, str) and center_color.startswith('#'):
                c_rgb = matplotlib.colors.hex2color(center_color)
                final_center = (c_rgb[0], c_rgb[1], c_rgb[2], alpha)
            elif len(center_color) == 3: final_center = center_color + (alpha,)

        num_petals = 6
        spin_angle = -x / 100 * 2 * np.pi
        petal_dist, petal_size, center_size = radius * 0.6, radius * 0.25, radius * 0.3

        for petal in range(num_petals):
            angle = petal * 2 * np.pi / num_petals + spin_angle
            petal_center = mg.orbit((0, 0), angle, petal_dist)
            canvas.circle(petal_center, petal_size, color=final_petal, style='fill')
            if bordercolor: canvas.circle(petal_center, petal_size, color=bordercolor, width=lw, style='stroke')
        canvas.circle((0, 0), center_size, color=final_center, style='fill')
        if bordercolor: canvas.circle((0, 0), center_size, color=bordercolor, width=lw, style='stroke')

class Ripple(Glyph):
    def ripple(self, x: float, canvas: mg.Canvas, fillcolor='navy', linewidth='15p', **kwargs) -> None:
        radius, final_color, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        frequency, amplitude, deformity = mg.lerp(x, 3, 20), mg.lerp(x, 0.1, 0.25), mg.lerp(x, 0.0, 0.75)
        points, steps, scale = [], 200, radius * 1.5
        for i in range(steps + 1):
            angle = (2 * np.pi) * (i / steps)
            mod = 0.5 + amplitude * np.sin(frequency * angle)
            random_shift = (random.uniform(-1, 1) * deformity)
            r = (mod + random_shift * amplitude) * scale
            points.append((r * np.cos(angle), r * np.sin(angle)))
        for idx in range(len(points) - 1):
            canvas.line(points[idx], points[idx + 1], color=final_color, width=lw, linecap='round')

class Blob(Glyph):
    def blob(self, x: float, canvas: mg.Canvas, bordercolor=None, fillcolor='tomato', style='fill', freq1=6, freq2=18, linewidth=None, **kwargs) -> None:
        radius, final_fill, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else (linewidth or '30p')
        t = x / 100.0
        k = 0.4
        f = (1 - k) * t + k * (t ** 2)
        min_amp, max_amp = 0.05, 0.5
        amp = min_amp + (max_amp - min_amp) * f
        base_r = 0.4 + 0.1 * t
        pts, npoints =[], 500
        for i in range(npoints):  
            theta = 2 * np.pi * i / npoints
            noise = np.sin(freq1 * theta) + 0.5 * np.sin(freq2 * theta)
            r_norm = base_r + amp * noise 
            r = r_norm * radius * 1.5
            pts.append((np.cos(theta) * r, np.sin(theta) * r))
        canvas.polygon(pts, color=final_fill, style=style, width=lw)
        if bordercolor:
            canvas.polygon(pts, color=bordercolor, style='stroke', width=lw)

    def variational_blob(self, **kwargs):
        return lambda x, c: self.blob(x, c, fillcolor=self.random_color(), style='fill', freq1=random.randint(3, 9), freq2=random.randint(12, 24), size=np.random.choice([True, False]), displacement=np.random.choice([True, False]), opacity=np.random.choice([True, False]), saturation=np.random.choice([True, False]), border=np.random.choice([True, False]), rotation=False, **kwargs)

class Flame(Glyph):
    def flame(self, x: float, canvas: mg.Canvas, flame_color='orange', wood_color='saddlebrown', linewidth='5p', **kwargs) -> None:
        radius, final_flame, _ = self._process_visual_features(
            x, canvas, flame_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        log_len = radius
        canvas.line((-log_len, -log_len*0.2+0.5), (log_len, log_len*0.2+0.5), color=wood_color, width='100p', linecap='square')
        canvas.line((-log_len, log_len*0.2+0.5), (log_len, -log_len*0.2+0.5), color=wood_color, width='100p', linecap='square')
        t = x / 100.0
        if t > 0.05:
            f_w, f_h = radius * 1.5, radius * 2 * t
            pts, steps = [(-f_w * 0.5, 0.5)], 20
            for i in range(steps + 1):
                nx = (i/steps) * 2 - 1 
                px = nx * f_w * 0.6
                h_profile = (1 - nx**2) * f_h
                noise = 0.3 + 0.5 * np.sin(i * 1.5 + t * 5)
                py = -max(0, h_profile) * noise
                pts.append((px, py))
            pts.append((f_w * 0.5, 0.5)) 
            canvas.polygon(pts, color=final_flame, style='fill')
            if t > 0.5:
                inner_pts = [(p[0]*0.5, p[1]*0.6+0.2) for p in pts]
                canvas.polygon(inner_pts, color='yellow', style='fill')


class Battery(Glyph):
    def battery(self, x: float, canvas: mg.Canvas, fill_color='lime', case_color='black', linewidth='5p', **kwargs) -> None:
        radius, final_fill, border_width_calc = self._process_visual_features(
            x, canvas, fill_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        t = x / 100.0
        w, h = radius, radius * 0.5
        canvas.rect((-w, -h), (w, h), color='#e0e0e0', style='fill')
        if t > 0.0:
            fill_w = min((2 * w) * t, 2*w)
            canvas.rect((-w, -h), (-w + fill_w, h), color=final_fill, style='fill')
        canvas.rect((-w, -h), (w, h), color=case_color, width=lw, style='stroke')
        canvas.rect((w, -h*0.3), (w + w*0.1, h*0.3), color=case_color, style='fill')


class Running(Glyph):
    def running(self, x: float, canvas: mg.Canvas, fillcolor='black', linewidth='5p', **kwargs) -> None:
        radius, final_color, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        s = radius * 1.25
        hip, lean_angle = (0, 0), mg.lerp(x, 0.0, np.pi/5)
        torso_len = s * 0.55
        neck_x = hip[0] - np.sin(lean_angle) * torso_len
        neck_y = hip[1] - np.cos(lean_angle) * torso_len
        head_r = s * 0.15
        head_cx = neck_x - np.sin(lean_angle) * head_r
        head_cy = neck_y - np.cos(lean_angle) * head_r - head_r*0.5 
        canvas.circle((head_cx, head_cy), head_r, color=final_color, width=lw, style='stroke')
        canvas.line(hip, (neck_x, neck_y), color=final_color, width=lw)
        leg_len = s * 0.4
        spread, knee_bend = mg.lerp(x, 0.0, np.pi/3), mg.lerp(x, 0.0, np.pi/2)
        r_hip_a = lean_angle - spread 
        r_knee = (hip[0] + np.sin(r_hip_a)*leg_len, hip[1] + np.cos(r_hip_a)*leg_len)
        r_calf_a = r_hip_a + knee_bend
        r_foot = (r_knee[0] + np.sin(r_calf_a)*leg_len, r_knee[1] + np.cos(r_calf_a)*leg_len)
        self.draw_line_strip(canvas, [hip, r_knee, r_foot], color=final_color, width=lw)
        l_hip_a = lean_angle + spread
        l_knee = (hip[0] + np.sin(l_hip_a)*leg_len, hip[1] + np.cos(l_hip_a)*leg_len)
        l_calf_a = l_hip_a + knee_bend
        l_foot = (l_knee[0] + np.sin(l_calf_a)*leg_len, l_knee[1] + np.cos(l_calf_a)*leg_len)
        self.draw_line_strip(canvas, [hip, l_knee, l_foot], color=final_color, width=lw)
        arm_len = s * 0.35
        swing, elbow_bend = mg.lerp(x, 0.0, np.pi/1.8), mg.lerp(x, 0.1, np.pi/2)
        ra_a = lean_angle - swing if x >= 0.05 else np.pi
        r_elbow = (neck_x + np.sin(ra_a)*arm_len, neck_y + np.cos(ra_a)*arm_len)
        r_hand = (r_elbow[0] + np.sin(ra_a - elbow_bend)*arm_len, r_elbow[1] + np.cos(ra_a - elbow_bend)*arm_len)
        self.draw_line_strip(canvas, [(neck_x, neck_y), r_elbow, r_hand], color=final_color, width=lw)
        la_a = lean_angle + swing if x >= 0.05 else np.pi
        l_elbow = (neck_x + np.sin(la_a)*arm_len, neck_y + np.cos(la_a)*arm_len)
        l_hand = (l_elbow[0] + np.sin(la_a - elbow_bend)*arm_len, l_elbow[1] + np.cos(la_a - elbow_bend)*arm_len)
        self.draw_line_strip(canvas, [(neck_x, neck_y), l_elbow, l_hand], color=final_color, width=lw)


class Eye(Glyph):
    def eye(self, x: float, canvas: mg.Canvas, iris_color='cornflowerblue', border_color='black', linewidth='20p', **kwargs) -> None:
        radius, final_iris, border_width_calc = self._process_visual_features(
            x, canvas, iris_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        w, h = radius * 1.5, mg.lerp(x, 0.0, radius * 0.7)
        if x > 0.02:
            ir = mg.lerp(x, 0.0, radius * 0.6)
            canvas.circle((0,0), ir, color=final_iris, style='fill')
            canvas.circle((0,0), ir*0.4, color='black', style='fill') 
        self.draw_quadratic_bezier(canvas, (-w, 0), (0, -h*2.5), (w, 0), color=border_color, width=lw)
        self.draw_quadratic_bezier(canvas, (-w, 0), (0, h*2.5), (w, 0), color=border_color, width=lw)
        num_lashes = 9
        for i in range(num_lashes):
            lx = mg.lerp((i+1)/(num_lashes+1)*100, -w*0.9, w*0.9)
            norm_x = lx / w
            ly = -h * 1.5 * (1 - norm_x**2)
            angle = mg.lerp(i/(num_lashes-1)*100, -np.pi/3, np.pi/3)
            lash_len = radius * 0.2
            ex = lx + np.sin(angle)*lash_len
            ey = ly - np.cos(angle)*lash_len - (radius*0.2) 
            canvas.line((lx, ly), (ex, ey), color=border_color, width=lw)
            canvas.line((-lx, -ly), (-ex, -ey), color=border_color, width=lw)


class Target(Glyph):
    def target(self, x: float, canvas: mg.Canvas, target_color='red', arrow_color='black', linewidth='5p', **kwargs) -> None:
        radius, final_target, border_width_calc = self._process_visual_features(
            x, canvas, target_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        for r in [1.0, 0.6, 0.2]:
            style = 'stroke' if r != 0.2 else 'fill'
            canvas.circle((0,0), radius*r, color=final_target, width=lw, style=style)
        dist = 0.5 - mg.lerp(x, 0.25, 0.5)
        tip_x, tip_y = dist, dist
        arrow_len = radius * 0.8
        tail_x, tail_y = tip_x + arrow_len, tip_y - arrow_len
        canvas.line((tail_x, tail_y), (tip_x, tip_y), color=arrow_color, width=linewidth)
        f_sz = radius * 0.15
        canvas.line((tail_x, tail_y), (tail_x + f_sz, tail_y), color=arrow_color, width=linewidth)
        canvas.line((tail_x, tail_y), (tail_x, tail_y - f_sz), color=arrow_color, width=linewidth)


class Eclipse(Glyph):
    def eclipse(self, x: float, canvas: mg.Canvas, eclipse_color='gold', shadow_color='white', linewidth='2p', **kwargs) -> None:
        radius, final_eclipse, border_width_calc = self._process_visual_features(
            x, canvas, eclipse_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        canvas.circle((0,0), radius, color=final_eclipse, style='fill')
        shift = mg.lerp(x, 0.0, radius*2.1)
        if (x/100.0) < 1.0:
            canvas.circle((shift, 0), radius, color=shadow_color, style='fill')
        canvas.circle((0,0), radius, color='black', width='5p', style='stroke')


class Fingerprint(Glyph):
    def fingerprint(self, x: float, canvas: mg.Canvas, bordercolor='navy', fillcolor='navy', linewidth='4p', **kwargs) -> None:
        color = bordercolor
        if bordercolor == 'navy':
            if fillcolor == 'navy':
                color = 'navy'
            color = fillcolor
        radius, final_color, border_width_calc = self._process_visual_features(
            x, canvas, color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        scan_y = mg.lerp(x, -radius*1.2, radius*1.2)
        rings = 6
        for i in range(1, rings+1):
            r_w, r_h = radius * (i/rings), radius * 1.3 * (i/rings)
            pts, steps = [], 40
            for j in range(steps+1):
                ang = 2 * np.pi * j / steps
                px, py = np.cos(ang) * r_w, np.sin(ang) * r_h
                if (j % 10) > 8 or py >= scan_y:
                    if len(pts) > 1: self.draw_line_strip(canvas, pts, color=final_color, width=lw)
                    pts = []
                    continue
                pts.append((px, py))
            if len(pts) > 1: self.draw_line_strip(canvas, pts, color=final_color, width=lw)


class Bucket(Glyph):
    def bucket(self, x: float, canvas: mg.Canvas, bucket_color='gray', paint_color='red', linewidth='40p', **kwargs) -> None:
        radius, final_paint, border_width_calc = self._process_visual_features(
            x, canvas, paint_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        rot_deg = mg.lerp(x, 0.0, 90.0)
        ang = np.radians(rot_deg)
        c, s = np.cos(ang), np.sin(ang)
        def rot(p): return (p[0]*c - p[1]*s, p[0]*s + p[1]*c)
        b_w, b_h = radius * 0.8, radius * 1.2
        poly = [(-b_w, -b_h/2), (-b_w*0.8, b_h/2), (b_w*0.8, b_h/2), (b_w, -b_h/2)]
        r_poly = [rot(p) for p in poly]
        t = x / 100
        if t < 1.0:
            fill_amount = 1.0 - (t * 0.8)
            y_liquid_top = b_h/2 - (b_h * fill_amount)
            w_liquid_top = b_w * (0.8 + 0.2 * fill_amount)
            margin = 0.95
            w_top_m, w_bot_m = w_liquid_top * margin, b_w * 0.8 * margin
            i_poly = [(-w_top_m, y_liquid_top), (-w_bot_m, b_h/2), (w_bot_m, b_h/2), (w_top_m, y_liquid_top)]
            r_in = [rot(p) for p in i_poly]
            canvas.polygon(r_in, color=final_paint, style='fill')
        if t > 0.1:
            start = rot((b_w, -b_h/2))
            drop_len = radius * (t - 0.5) + 0.2
            if t < 0.99: canvas.line(start, (start[0], start[1] + drop_len), color=final_paint, width=f"{radius*15}p")
            if t > 0.6:
                pr = radius * (t - 0.6) * 1.5
                canvas.ellipse((start[0], start[1] + drop_len), pr, pr*0.4, color=final_paint, style='fill')
        canvas.polygon(r_poly, color=bucket_color, width=lw, style='stroke')


class Gym(Glyph):
    def gym(self, x: float, canvas: mg.Canvas, plate_color='black', bar_color='gray', linewidth='10p', **kwargs) -> None:
        radius, final_plate, border_width_calc = self._process_visual_features(
            x, canvas, plate_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        t = x / 100.0
        bar_len, bar_thickness = radius * 2.5, mg.lerp(x, 20, 100)
        canvas.line((-bar_len/2, 0), (bar_len/2, 0), color=bar_color, width=f"{bar_thickness}p")
        max_plates = 3
        num_plates = int(t * max_plates)
        rem = (t * max_plates) - num_plates
        start_x, spacing = radius * 0.75, radius * 0.25
        base_thick = 20
        thickness = base_thick + t * 120
        for side in [-1, 1]:
            for i in range(num_plates + 1):
                if i == num_plates and rem < 0.1: continue
                h, curr_thick = radius * 1.5, thickness
                if i == num_plates: 
                    h *= rem
                    curr_thick *= rem
                px = side * (start_x + i * spacing)
                canvas.line((px, -h/2), (px, h/2), color=final_plate, width=f"{curr_thick}p", linecap="square")


class Grin(Glyph):
    def grin(self, x: float, canvas: mg.Canvas, face_color='yellow', feature_color='black', linewidth='40p', **kwargs) -> None:
        if not kwargs.get('hue') and not kwargs.get('saturation'):
            f = x / 50.0 if x < 50 else (x - 50) / 50.0
            face_color = (1.0, f*0.65, 0.0, 1.0) if x < 50 else (1.0 - f, 0.65 + f*0.35, 0.0, 1.0)
        radius, final_face, border_width_calc = self._process_visual_features(
            x, canvas, face_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        canvas.circle((0,0), radius, color=final_face, style='fill')
        e_y = -radius * 0.3
        eye_r = mg.lerp(x, 0.04, radius*0.11)
        canvas.circle((-radius*0.35, e_y), eye_r, color=feature_color, style='fill')
        canvas.circle((radius*0.35, e_y), eye_r, color=feature_color, style='fill')
        m_w, m_y = radius * 0.5, radius * 0.25
        offset = mg.lerp(x, -radius*0.4, radius*0.4)
        self.draw_quadratic_bezier(canvas, (-m_w, m_y), (0, m_y + offset), (m_w, m_y), color=feature_color, width=lw)


class Starsky(Glyph):
    def starsky(self, x: float, canvas: mg.Canvas, fillcolor='gold', linewidth='0p', **kwargs) -> None:
        radius, final_color, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )        
        for i in range(int(x)):
            rx, ry = np.random.uniform(-0.8, 0.8), np.random.uniform(-0.8, 0.8)
            px, py = rx * radius * 2.5, ry * radius * 2.5
            r_scale = np.random.uniform(0.1, 2)
            sz, inner_f = radius * 0.15 * r_scale, mg.lerp(x, 0.01, 0.5)
            c_choice = final_color
            if fillcolor == 'gold':
                c_choice = random.choice(['#FFD700', '#DAA520', '#B8860B', '#F0E68C'])
            verts = []
            angle_off = np.random.uniform(0, np.pi)
            for s in range(5):
                verts.append(mg.orbit((px,py), angle_off + s*2*np.pi/5, sz))
                verts.append(mg.orbit((px,py), angle_off + (s+0.5)*2*np.pi/5, sz * inner_f))
            canvas.polygon(verts, color=c_choice, style='fill')


class Paw(Glyph):
    def paw(self, x: float, canvas: mg.Canvas, fillcolor='saddlebrown', linewidth='5p', **kwargs) -> None:
        radius, final_color, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        scale_toe = mg.lerp(x, 0.05, 0.2)
        scale_pad = mg.lerp(x, 0.05, 2)
        main_y, w, h = radius * 0.1, radius * 0.6 * scale_pad, radius * 0.4 * scale_pad
        self.draw_quadratic_bezier(canvas, (0, main_y+h), (-w, main_y), (0, main_y-h*0.5), color=final_color, width=f"{30*scale_pad}p", style='fill')
        self.draw_quadratic_bezier(canvas, (0, main_y+h), (w, main_y), (0, main_y-h*0.5), color=final_color, width=f"{30*scale_pad}p", style='fill')
        toe_dist, toe_r = radius * 0.6, radius * scale_toe
        for i in [-1, 0, 1]:
            angle = i * (np.pi / 2.5)
            tx, ty = np.sin(angle) * toe_dist, main_y - h - np.cos(angle) * toe_dist * 0.9
            canvas.circle((tx, ty), toe_r, color=final_color, style='fill')


class CloudSun(Glyph):
    def cloudsun(self, x: float, canvas: mg.Canvas, sun_color='gold', cloud_color='lightgray', linewidth='2p', **kwargs) -> None:
        radius, final_sun, border_width_calc = self._process_visual_features(
            x, canvas, sun_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        canvas.circle((0,0), radius*0.6, color=final_sun, style='fill')
        for i in range(8):
            p1, p2 = mg.orbit((0,0), i*np.pi/4, radius*0.7), mg.orbit((0,0), i*np.pi/4, radius*1.0)
            canvas.line(p1, p2, color=final_sun, width=lw)
        num_clouds = 3
        np.random.seed(10)
        for i in range(num_clouds):
            dir_angle, dist = i * (2*np.pi/3), mg.lerp(x, 0.0, radius*1.5)
            cx, cy = np.cos(dir_angle) * dist, np.sin(dir_angle) * dist
            c_scale = 1.0 - mg.lerp(x, 0.0, 1.0)
            if c_scale > 0.01:
                cr = radius * 0.5 * c_scale
                bubbles = [(0,0,1.0), (-0.8, 0.2, 0.8), (0.8, 0.2, 0.8)]
                for bx, by, bsz in bubbles:
                    pos = (cx + bx*cr, cy + by*cr)
                    canvas.circle(pos, cr*bsz, color=cloud_color, style='fill')


class Banana(Glyph):
    def banana(self, x: float, canvas: mg.Canvas, skin_color='yellow', flesh_color='ivory', linewidth='3p', **kwargs) -> None:
        radius, final_skin, border_width_calc = self._process_visual_features(
            x, canvas, skin_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        t = x / 100.0
        flesh_h = mg.lerp(t, radius*2, radius*0.2)
        canvas.rect((-radius*0.25, -radius), (radius*0.25, -radius + flesh_h), color=flesh_color, style='fill')
        if t < 0.1: self.draw_quadratic_bezier(canvas, (-radius, -radius), (0,0), (radius, -radius), color=final_skin, width='25p')
        else:
            peel_drop = radius * t
            p1 = (-radius*0.25, -radius + flesh_h if t<0.5 else 0)
            p2 = (-radius, p1[1] + peel_drop)
            canvas.line(p1, p2, color=final_skin, width='15p')
            p3 = (radius*0.25, -radius + flesh_h if t<0.5 else 0)
            p4 = (radius, p3[1] + peel_drop)
            canvas.line(p3, p4, color=final_skin, width='15p')


class Hourglass(Glyph):
    def hourglass(self, x: float, canvas: mg.Canvas, fillcolor='gold', glass_color='black', linewidth='20p', **kwargs) -> None:
        radius, final_sand, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        t, w, h = x / 100.0, radius *0.85, radius
        if t < 1.0:
            rem = 1.0 - t
            curr_h = h * np.sqrt(rem) 
            curr_w = w * (curr_h / h)
            canvas.polygon([(0,0), (curr_w, -curr_h), (-curr_w, -curr_h)], color=fillcolor, style='fill')
        if t > 0.0:
            pile_h = h * np.sqrt(t)
            pile_top_y = h - pile_h
            width_at_top = w * (pile_top_y / h)
            canvas.polygon([(-w, h), (w, h), (width_at_top, pile_top_y), (-width_at_top, pile_top_y)], color=fillcolor, style='fill')
            if t < 0.98: canvas.line((0,0), (0, pile_top_y), color=fillcolor, width='2p')
        canvas.polygon([(-w, -h), (w, -h), (0, 0)], color=glass_color, width=lw, style='stroke')
        canvas.polygon([(-w, h), (w, h), (0, 0)], color=glass_color, width=lw, style='stroke')


class Book(Glyph):
    def book(self, x: float, canvas: mg.Canvas, cover_color='firebrick', page_color='white', linewidth='20p', **kwargs) -> None:
        radius, final_cover, border_width_calc = self._process_visual_features(
            x, canvas, cover_color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        t = x / 100.0
        w, h = radius, radius * 0.3
        left_h, right_h = h * t, h * (1.0 - t)
        canvas.rect((-w, 0), (0, radius*0.1), color=final_cover, style='fill')
        canvas.rect((0, 0), (w, radius*0.1), color=final_cover, style='fill')
        if left_h > 0.01:
            canvas.rect((-w, -left_h), (0, 0), color=page_color, style='fill')
            canvas.rect((-w, -left_h), (0, 0), color='black', width=lw, style='stroke')
        if right_h > 0.01:
            canvas.rect((0, -right_h), (w, 0), color=page_color, style='fill')
            canvas.rect((0, -right_h), (w, 0), color='black', width=lw, style='stroke')
        canvas.line((0, radius*0.1), (0, -max(left_h, right_h)), color=final_cover, width=lw)


class Wind(Glyph):
    def wind(self, x: float, canvas: mg.Canvas, fillcolor='skyblue', linewidth='50p', **kwargs) -> None:
        radius, final_color, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        t = x / 100.0
        offsets, lengths = [-0.5, 0, 0.5], [0.8, 1.0, 0.7]
        for i, off in enumerate(offsets):
            y = off * radius
            base_len = radius * lengths[i] * (0.3 + 0.8*t)
            start_x, end_x = -base_len / 2, base_len / 2
            if x < 0.2: 
                canvas.line((start_x, y), (end_x, y), color=final_color, width=lw)
            else:
                curve_sz = radius * 0.3 * t
                canvas.line((start_x, y), (end_x, y), color=final_color, width=lw)
                cp = (end_x + curve_sz, y - curve_sz)
                end_curve = (end_x, y - curve_sz*1.5)
                self.draw_quadratic_bezier(canvas, (end_x, y), cp, end_curve, color=final_color, width=lw)


class Cookie(Glyph):
    def cookie(self, x: float, canvas: mg.Canvas, color='peru', chip_color='saddlebrown', linewidth='2p', **kwargs) -> None:
        radius, final_color, border_width_calc = self._process_visual_features(
            x, canvas, color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        t = x / 100.0
        if t > 0.01:
            pts, steps = [(0,0)], 40
            angle_span = 2 * np.pi * t
            for i in range(steps + 1):
                angle = (i / steps) * angle_span
                pts.append(mg.orbit((0,0), angle, radius))
            canvas.polygon(pts, color=final_color, style='fill')
            for i in range(6):
                if (i+0.5)/6 < t:
                     a = (i / 6) * 2 * np.pi + 0.5
                     pos = mg.orbit((0,0), a, radius * 0.6)
                     canvas.circle(pos, radius*0.15, color=chip_color, style='fill')


class Rainbow(Glyph):
    def rainbow(self, x: float, canvas: mg.Canvas, color='purple', linewidth='40p', **kwargs) -> None:
        radius, final_color, border_width_calc = self._process_visual_features(
            x, canvas, color, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        cols = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
        max_r = radius
        stroke_w = radius * 0.3 
        lw = border_width_calc if kwargs.get('border') else linewidth
        # print(stroke_w)
        # stroke_w = mg.lerp(x, 0, stroke_w)
        arc_angle = mg.lerp(x, np.pi/4, np.pi)
        for i, c in enumerate(cols):
            r_band = max_r - (i * stroke_w * 0.5)
            pts, steps = [], 20
            for j in range(steps+1):
                a = -arc_angle/2 + (j/steps) * arc_angle
                pts.append(mg.orbit((0, radius*0.5), a, r_band))
            draw_c = c
            self.draw_line_strip(canvas, pts, color=draw_c, width=lw) 


class Water(Glyph):
    def water(self, x: float, canvas: mg.Canvas, fillcolor='blue', linewidth='5p', **kwargs) -> None:
        radius, final_color, border_width_calc = self._process_visual_features(
            x, canvas, fillcolor, kwargs.get('size'), kwargs.get('displacement'), kwargs.get('opacity'), kwargs.get('hue'), kwargs.get('saturation'), kwargs.get('rotation'), kwargs.get('border')
        )
        lw = border_width_calc if kwargs.get('border') else linewidth
        amp = mg.lerp(x, radius*0.05, radius*0.1)
        freq, width, steps = mg.lerp(x, 3, 6), radius * 2, 200
        for line in range(3):
            y_base = (line - 1) * radius * 0.5
            pts = []
            for i in range(steps+1):
                lx = -width/2 + width * (i/steps)
                phase = line * np.pi / 4
                ly = y_base + np.sin(i/steps * freq * 2 * np.pi + phase) * amp
                pts.append((lx, ly))
            self.draw_line_strip(canvas, pts, color=final_color, width=lw)
