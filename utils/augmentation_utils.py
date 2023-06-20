import os
import time
import random
from datetime import datetime

from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import albumentations as A

from .FloatToChange import FloatToChange

class AugTransform:
    def __init__(self, transform, Marker_class):

        self.factor = 1
        self.Marker_class = Marker_class

        self.original_transform = transform
        self.transform = None
        self.set_factor(self.factor) #change FloatToChange to float

    def set_new_value(self, obj,key, new_value, object_level):
        if object_level: setattr(obj,key,new_value)
        else: obj[key]= new_value
        

    def object_unwrapper(self, obj, got_key, got_attr, object_level=True):

        changed = False
        if isinstance(got_attr, tuple) or isinstance(got_attr, list) or isinstance(got_attr, dict):
            if isinstance(got_attr, tuple): got_attr = list(got_attr); was_tuple = True
            else: was_tuple = False
            if isinstance(got_attr, dict): iterator = got_attr.items()
            else: iterator = enumerate(got_attr)
            for key, got_attr2 in iterator:
                changed = self.object_unwrapper(got_attr, key, got_attr2, object_level=False)
            if was_tuple: got_attr = tuple(got_attr)
            if changed: self.set_new_value(obj, got_key, got_attr, object_level)
            return False #none of the values was changed

        elif isinstance(got_attr, self.Marker_class):
            self.set_new_value(obj, got_key, got_attr(self.factor), object_level)
            return True #some of the values was changed


    def set_factor(self, factor=1, print_state=False):

        if print_state: print("\n____TRANSFORMS TO APPLY:")
        self.factor = factor
        self.transform = deepcopy(self.original_transform)
        if not isinstance(self.transform, A.Compose): self.transform = [self.transform]
        for t in self.transform:
            attrs = [a for a in t.__dict__.keys() if not a.startswith('__') and not a.endswith('__')]
            for attr in attrs:
                got_attr = getattr(t, attr)
                self.object_unwrapper(t, attr, got_attr, object_level=True)
            if print_state: print(t)
        
        return self

    def __call__(self, image):
        return self.transform(image=image)

class LogoSequenceGenerator:
    def __init__(self, folder_path, logo_path, output_path, fps=25, bg_transforms=None, final_transforms=None):
        self.file_paths = [os.path.join(folder_path, entry) for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry))]
        self.logo_paths = [os.path.join(logo_path, entry) for entry in os.listdir(logo_path) if os.path.isfile(os.path.join(logo_path, entry))]
        self.bg_transforms = bg_transforms
        self.final_transforms = final_transforms
        self.fourcc = cv2.VideoWriter_fourcc(*"flv1")
        self.output_path = output_path
        self.fps = fps
    
    def check_initial_scale(self,image):
        #method to avoid black borders when traversing
        factor = 4
        if image.shape[0]<factor*max(WIDTH,HEIGHT) or image.shape[1]<factor*max(WIDTH,HEIGHT):
            scale_transformer = A.SmallestMaxSize(max_size=factor*max(WIDTH,HEIGHT), always_apply=True)
            image = scale_transformer(image=image)['image']
        return image
    
    def traversing(self, seq_length, t_scale=(0.8,1.12), t_rotate=(-20,20), t_trans={"x":[-0.25,0.2], "y":[-0.1,0.25]}, max_random_scale=None): #trans Є -0.25;0.25
        frames = []

        idx = random.randint(0,len(self.file_paths)-1)
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = self.check_initial_scale(image)
        initial_transforms = A.Compose([ #bg transforms should be as random as possible, as we are teaching our network to ignore the background
                                self.bg_transforms,
                                A.Rotate(limit=(t_rotate[0],t_rotate[0]),border_mode=cv2.BORDER_REFLECT,crop_border=False, always_apply=True),
                                A.RandomScale(scale_limit=[1,10]),
                                ])
        image = initial_transforms(image=image)['image']

        for i in range(seq_length):
            factor = i/seq_length
            t_s = t_scale[0]+(t_scale[1]-t_scale[0])*factor
            t_r = (t_rotate[1]-t_rotate[0])*factor
            t_t_x = t_trans['x'][0]+(t_trans['x'][1]-t_trans['x'][0])*factor
            t_t_y = t_trans['y'][0]+(t_trans['y'][1]-t_trans['y'][0])*factor
            traversing = A.Compose([
                                A.Affine(scale=t_s,rotate=t_r,translate_percent={'x':t_t_x,'y':t_t_y}, always_apply=True),
                                A.CenterCrop(HEIGHT,WIDTH,always_apply=True)
                                 ])
            frame = traversing(image=image)['image']
            frames.append(frame)
        return frames
    
    def video_writer(self, frames, label):

        path_0 = os.path.join(self.output_path,str(label))
        if not os.path.exists(path_0): os.mkdir(path_0)

        now = datetime.now()    
        video_name = now.strftime("%Y%m%d%H%M%S%f")+".flv"
        save_video_to = os.path.join(self.output_path,str(label),video_name)

        return video_writer(frames, save_video_to, self.fps, fourcc=self.fourcc)


    def generator(self, seq_length):
        frames = []

        with_logo = random.randint(0,1)
        num_traverses = clamp(1, int(base_devi_gaus(1,0.7,one_sided="plus")), 3)
        subseq_lengths = [len(e) for e in np.split(np.arange(seq_length),np.sort(np.random.randint(seq_length, size=num_traverses-1)))]

        for subseq_length in subseq_lengths:
            t_scale=(base_devi_gaus(1,0.05),base_devi_gaus(1,0.05)) 
            t_rotate=(base_devi_gaus(0,33),base_devi_gaus(0,33))
            t_trans={"x":[base_devi_gaus(0,0.1),base_devi_gaus(0,0.1)], "y":[base_devi_gaus(0,0.1),base_devi_gaus(0,0.1)]}

            frames.extend(self.traversing(subseq_length, t_scale=t_scale, t_rotate=t_rotate, t_trans=t_trans))
        
        if with_logo:
            logo_img = generate_logo(self.logo_paths)
            blend_opacity = gen_blend_opacity()
            for i, f in enumerate(frames):
                frames[i] = blend_images(f, logo_img,blend_opacity)

        seed = random.randint(0,2**32-1)
        for i, f in enumerate(frames):
            random.seed(seed)
            np.random.seed(seed)
            frames[i] = self.final_transforms(image=f)['image']

        self.video_writer(frames, with_logo)

    def __call__(self, num_videos):
        for _ in range(num_videos):
            self.generator(pick_seq_length())

class LogoImageGenerator:

    def __init__(self, folder_path, logo_path, output_path, w, h, bg_transform=None, final_transforms=None, dr_graphical=1, dr_geometrical=1, print_state=False, logo_name="Untitled", logo_base_scale=1.2, create_subfolder=True):
        self.logo_name = logo_name
        if not os.path.exists(os.path.join(output_path,self.logo_name)): os.mkdir(os.path.join(output_path,self.logo_name))

        self.file_paths = [os.path.join(folder_path, entry) for entry in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, entry))]
        self.logo_paths = logo_path
        
        self.w = w
        self.h = h

        #dr_ stands for Diversity rate
        assert dr_graphical <= 1 and dr_graphical >= 0, "diverse_rate is a percentage, should Є [0,1]"
        assert dr_geometrical <= 1 and dr_geometrical >= 0, "diverse_rate is a percentage, should Є [0,1]"
        self.dr_graphical = dr_graphical
        self.dr_geometrical = dr_geometrical

        self.bg_transform = bg_transform.set_factor(factor=self.dr_graphical, print_state=print_state)
        self.final_transforms = final_transforms.set_factor(factor=self.dr_graphical, print_state=print_state)
        self.logo_base_scale = logo_base_scale

        self.output_path = os.path.join(output_path,self.logo_name)
        if create_subfolder:
            subfolder = "diversity_geo={:1.2f}_diversity_gra={:1.2f}".format(self.dr_geometrical,self.dr_graphical)
            self.output_path = os.path.join(self.output_path, subfolder)
        if not os.path.exists(self.output_path): os.mkdir(self.output_path)

    def __call__(self, num_pics, with_logo=None):
        for _ in range(num_pics):
            self.generator(with_logo)

    def generator(self, with_logo):
        if with_logo is None:
            with_logo = bool(random.randint(0,1))

        idx = random.randint(0,len(self.file_paths)-1)
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)

        if self.bg_transform:
            image = self.bg_transform(image=image)['image']
        
        if with_logo:
            logo_img, logo_scale = generate_logo(self.logo_paths, self.dr_geometrical, self.w, self.h, logo_base_scale=self.logo_base_scale)

            blend_opacity = gen_blend_opacity(diverse_rate=(self.dr_geometrical+self.dr_graphical)/2)
            image = blend_images(image, logo_img,blend_opacity)

        else:  logo_scale = random.uniform(0,1)

        if self.final_transforms:
            image = self.final_transforms(image=image)['image']

        #perform occlusion
        for _ in range(int(4*self.dr_geometrical)):
            colour = np.random.randint(255, size=3).tolist()
            h, w = random.randint(0, int(min(self.w, self.h)*logo_scale*self.dr_geometrical*0.6)), random.randint(0, int(min(self.w, self.h)*logo_scale*self.dr_geometrical*0.6)) #0.6 stands for occluding logo only by 60%
            dropout = A.dropout.cutout.Cutout(num_holes=1,max_h_size=h,max_w_size=w,fill_value=colour,p=0.33)
            image = dropout(image=image)['image']

            # image = draw_text(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_writer(self.output_path, image, int(with_logo))

def image_writer(output_path, img, label):
    path_0 = os.path.join(output_path,str(label))
    if not os.path.exists(path_0): os.mkdir(path_0)

    now = datetime.now()
    pic_name = now.strftime("%Y%m%d%H%M%S%f")+".jpg"

    save_to = os.path.join(output_path,str(label),pic_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_to, img)


def crop_transperent_fields(logo_dir):
    # util to crop transparent fields of the logo images 
    # for instance, when you have a PNG file, it used to have transperent fields around
    for logo in os.listdir(logo_dir):
        logo_img = cv2.imread(os.path.join(logo_dir,logo),cv2.IMREAD_UNCHANGED)
        b, g, r, alpha = cv2.split(logo_img)

        raw_0 = np.sum(alpha, axis=1)==0
        col_0 = np.sum(alpha, axis=0)==0

        raw_pos_1, raw_pos_2 = 0, None
        col_pos_1, col_pos_2 = 0, None

        for i, (e1, e2) in enumerate(zip(raw_0,raw_0[1:])):
            if raw_0[0]==True and e1==True and e2==False and raw_pos_1==0: raw_pos_1 = i
            if e1==False and e2==True: raw_pos_2 = i
        if raw_pos_2==None: raw_pos_2=raw_0.shape[0]

        for i, (e1, e2) in enumerate(zip(col_0,col_0[1:])):
            if col_0[0]==True and e1==True and e2==False and col_pos_1==0: col_pos_1 = i
            if e1==False and e2==True: col_pos_2 = i
        if col_pos_2==None: col_pos_2=col_0.shape[0]

        new_img = logo_img[raw_pos_1:raw_pos_2,col_pos_1:col_pos_2]
        cv2.imwrite(os.path.join(logo_dir,logo), new_img)

def squarify_image(image):
    h, w, _ = image.shape
    L = max(h,w)
    h_rest = L - h
    w_rest = L - w
    h1 = h_rest//2
    h2 = h_rest-h1
    w1 = w_rest//2
    w2 = w_rest-w1
    T_pad = A.CropAndPad(px=(h1,w1,h2,w2))
    image = T_pad(image=image)['image']
    return image

def gen_blend_opacity(diverse_rate=1):
    return (1 - clamp(0,abs(random.gauss(0,0.3*diverse_rate)),0.95))

def base_devi_gaus(base, deviation, one_sided=None):
    if one_sided == "plus": return base+abs(random.gauss(0, deviation))
    elif one_sided == "minus": return base-abs(random.gauss(0, deviation))
    else: return (base + random.gauss(0,deviation))

def pick_seq_length():
    sl = np.random.exponential(70, 1).astype(int)
    if sl > 500 or sl < 15: return pick_seq_length()
    else: return sl

def clamp(min_value, num, max_value):
   if   min_value!=None and max_value!=None: return max(min(num, max_value), min_value)
   elif min_value==None and max_value!=None: return min(num, max_value)
   elif min_value!=None and max_value==None: return max(num, min_value)

def gaussian_color_jitter(image):

    h = random.gauss(0,4)
    s = random.gauss(0,12)
    v = random.gauss(0,12)
    T_channeled = A.HueSaturationValue (hue_shift_limit=(h,h), sat_shift_limit=(s,s), val_shift_limit=(v,v), always_apply=True)
    image = T_channeled(image=image)['image']
    return image

def blend_images(bottom_image, top_image, opacity):

    h, w, c = bottom_image.shape
    bottom_image = bottom_image.astype(float)

    top_image = cv2.resize(top_image, (w,h))

    b, g, r, alpha = cv2.split(top_image)

    top_image = cv2.merge([b, g, r]).astype(float)

    alpha = alpha/255*opacity # Convert alpha channel to float and scale it to range [0, 1]
    mask = cv2.merge([alpha, alpha, alpha]).astype(float)

    # Apply the mask to the top image
    masked_top = cv2.multiply(top_image, mask) 
    mask = cv2.subtract(np.ones(bottom_image.shape), mask)# Invert the mask
    masked_bottom = cv2.multiply(bottom_image, mask)
    blended_image = cv2.add(masked_top, masked_bottom).astype(np.uint8)# Add the two masked images together

    return blended_image


def generate_logo(logo_path, dr_geo, w, h, logo_base_scale=1):

    if os.path.isdir(logo_path): logo_path = sample_random_file(logo_path)

    logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo_img.shape[-1] == 3: logo_img = cv2.cvtColor(logo_img, cv2.COLOR_RGB2RGBA) #force add alpha
    logo_img = squarify_image(logo_img)

    downscale_value = clamp(0.05, base_devi_gaus(1,0.3*dr_geo,one_sided="minus"),.999)
    scale_xy = logo_base_scale*clamp(0.05, base_devi_gaus(1,0.33*dr_geo), None)
    blur_value = int(base_devi_gaus(1,6*dr_geo*scale_xy,one_sided="plus"))

    logo_transforms = A.Compose([
                        A.Rotate(limit=1,crop_border=True,p=1),
                        A.LongestMaxSize(max_size=int(max(w,h)*1.5)), #so that our downscale and blur augmentations would take generic effect
                        A.Blur(blur_limit=blur_value,p=1 if blur_value < 3 else 0),#used this trick because either 0x0 or 3x3 kernels provide blur
                        A.Downscale(scale_min=downscale_value,scale_max=downscale_value, interpolation=cv2.INTER_NEAREST, p=1),
                        A.Affine(translate_percent=base_devi_gaus(0,0.1*dr_geo),scale={"x":scale_xy*base_devi_gaus(1,0.03*dr_geo), "y":scale_xy*base_devi_gaus(1,0.03*dr_geo)},p=1),
                        A.Sharpen (alpha=(0, 0.5*dr_geo), lightness=(1 - dr_geo, 1 + dr_geo), always_apply=True),
                    ])

    logo_img = logo_transforms(image=logo_img)['image']
    logo_img[:,:,:3] = gaussian_color_jitter(logo_img[:,:,:3])
    
    return logo_img, scale_xy

def sample_random_file(folder):

    font_paths = [os.path.join(folder, entry) for entry in os.listdir(folder) if os.path.isfile(os.path.join(folder, entry))]
    font_path = font_paths[random.randint(0,len(font_paths)-1)]
    return font_path

def sample_phrase(file_path):
    with open(file_path, 'r') as file:
        data = file.read().split("\n")
    
    data = np.random.choice(data, 1)[0]
    
    return data

def center_crop(image, border):
    h, w = image.shape[0] - (2*border), image.shape[1] - (2*border)
    center = np.array(image.shape) / 2
    y = int(center[0] - h/2)
    x = int(center[1] - w/2)
    crop_img = image[y:y+h, x:x+w]
    return crop_img

def sample_colour_from_image(image):
    h = np.random.randint(image.shape[0])
    w = np.random.randint(image.shape[1])
    sample = np.flip(image[h,w,:])
    return sample

def sample_2_colours(image, min_dist=300):
    sample1 = sample_colour_from_image(image)

    dist = 0
    attempts = 0

    while dist < min_dist:
        attempts += 1
        sample2 = sample_colour_from_image(image)
        dist = np.linalg.norm(sample1-sample2)
        if attempts % 10 ==0: min_dist -= 10

    return tuple(sample1), tuple(sample2)

def get_text_dimensions(text_string, font):
    ascent, descent = font.getmetrics()

    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent

    return (text_width, text_height)

def draw_text(h, w, font_source, text_source, convert_RGB=True):

    # scale = random.randint(10, image_scale // 4)
    scale_initial = 20
    required_text_w = w - (w//5)

    font_path = sample_random_file(font_source)

    font = ImageFont.truetype(font_path, scale_initial)  
    text = sample_phrase(text_source)

    text_w, text_h = get_text_dimensions(text, font)
    random_factor = random.gauss(1,0.1)
    scale_proper = int(scale_initial * required_text_w / text_w * random_factor)

    font = ImageFont.truetype(font_path, scale_proper)  

    #image pre-processing
    image = np.zeros((h, w, 3), dtype=int) + 255

    bias_x = random.gauss(1,0.1)
    bias_y = random.gauss(1,0.1)
    origin = [w*bias_x//2,h*bias_y//2]

    # image post-processing
    image = Image.fromarray(image.astype(np.uint8))  
    draw = ImageDraw.Draw(image)  
    draw.text(origin, text, (0,0,0), font=font, anchor="mm")  
    if convert_RGB:
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  

    return np.array(image)