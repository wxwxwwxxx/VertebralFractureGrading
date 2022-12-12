import random
import numpy as np
from scipy import ndimage
class RandomAlign3D(object):
    def __init__(self, length):
        self.length = length
    def __call__(self, img):
        x, y, z, c = np.shape(img)
        crop_x_l = random.randint(0, x - self.length) if x >= self.length else 0
        crop_y_l = random.randint(0, y - self.length) if y >= self.length else 0
        crop_z_l = random.randint(0, z - self.length) if z >= self.length else 0
        img = img[crop_x_l:crop_x_l + self.length, crop_y_l:crop_y_l + self.length, crop_z_l:crop_z_l + self.length, :]

        pad_x_l = random.randint(0, self.length - x) if x < self.length else 0
        pad_y_l = random.randint(0, self.length - y) if y < self.length else 0
        pad_z_l = random.randint(0, self.length - z) if z < self.length else 0
        pad_x_r = self.length - x - pad_x_l if x < self.length else 0
        pad_y_r = self.length - y - pad_y_l if y < self.length else 0
        pad_z_r = self.length - z - pad_z_l if z < self.length else 0

        img = np.pad(img, ((pad_x_l, pad_x_r), (pad_y_l, pad_y_r), (pad_z_l, pad_z_r), (0, 0)),
                     'constant',
                     constant_values=(0, 0))

        return img

class FixedAlign3D(object):
    def __init__(self, length):
        self.length = length
    def __call__(self, img):
        x, y, z, c = np.shape(img)
        crop_x_l = (x - self.length) // 2 if x >= self.length else 0
        crop_y_l = (y - self.length) // 2 if y >= self.length else 0
        crop_z_l = (z - self.length) // 2 if z >= self.length else 0
        img = img[crop_x_l:crop_x_l + self.length, crop_y_l:crop_y_l + self.length, crop_z_l:crop_z_l + self.length, :]

        pad_x_l = (self.length - x) // 2 if x < self.length else 0
        pad_y_l = (self.length - y) // 2 if y < self.length else 0
        pad_z_l = (self.length - z) //2  if z < self.length else 0
        pad_x_r = self.length - x - pad_x_l if x < self.length else 0
        pad_y_r = self.length - y - pad_y_l if y < self.length else 0
        pad_z_r = self.length - z - pad_z_l if z < self.length else 0

        img = np.pad(img, ((pad_x_l, pad_x_r), (pad_y_l, pad_y_r), (pad_z_l, pad_z_r), (0, 0)),
                     'constant',
                     constant_values=(0, 0))
        return img

class RandomRotation3D(object):
    def __init__(self,max_angle):
        self.max_angle = max_angle
    def __call__(self, img):
        angle_2 = random.randint(-self.max_angle,self.max_angle)
        angle_1 = random.randint(-self.max_angle,self.max_angle)
        angle_3 = random.randint(-self.max_angle,self.max_angle)
        img = ndimage.rotate(img, angle_2, axes=(0, 1), reshape=True, order=0)
        img = ndimage.rotate(img, angle_1, axes=(0, 2), reshape=True, order=0)
        img = ndimage.rotate(img, angle_3, axes=(1, 2), reshape=True, order=0)
        return img

class RandomColorScale3D(object):
    def __init__(self,max_scale):
        self.max_scale = max_scale
    def __call__(self, img):

        scale_a = (2*self.max_scale * random.random()-self.max_scale)+1
        scale_b = (2*self.max_scale * random.random()-self.max_scale)+1
        scale = np.array([scale_a,scale_b,1.0]).astype('float32')
        scale = scale[None,None,None,:]
        img = img * scale
        return img

class RandomZoom3D(object):
    def __init__(self,max_scale):
        self.max_scale = max_scale
    def __call__(self, img):
        scale = (2*self.max_scale * random.random()-self.max_scale )+1
        img = ndimage.zoom(img, (scale, scale, scale, 1),order=0)
        return img

class RandomShift3D(object):
    def __init__(self,max_shift):
        self.max_shift = max_shift
    def __call__(self, img):
        shift_x = random.randint(-self.max_shift, self.max_shift)
        shift_y = random.randint(-self.max_shift, self.max_shift)
        shift_z = random.randint(-self.max_shift, self.max_shift)
        img = ndimage.shift(img, (shift_x, shift_y, shift_z, 0),order=0)
        return img

class RandomNoise3D(object):
    def __init__(self,sigma):
        self.sigma = sigma
    def __call__(self,img):
        img_shape = np.shape(img[:,:,:,0:2])

        noise = np.random.normal(0,self.sigma,img_shape).astype('float32')
        noise = np.reshape(noise,img_shape)
        noise = np.concatenate([noise,np.zeros_like(noise[:,:,:,0:1])],axis=-1)
        img = img + noise
        img = np.clip(img,0.0,1.0)
        return img

class RandomMask3D(object):
    def __init__(self,max_length,num,mask_prob):
        self.m_l = max_length
        self.n = num
        self.p = mask_prob
    def __call__(self,img):
        for i in range(self.n):
            if random.random()>1-self.p:
                continue
            x_r,y_r,z_r,_ = np.shape(img)
            z_ratio = z_r/x_r
            z_m_l=round(self.m_l*z_ratio+0.4)
            l_x = random.randint(0, self.m_l)
            l_y = random.randint(0, self.m_l)
            l_z = random.randint(0, z_m_l)
            l_p_x = random.randint(0, x_r - l_x)
            l_p_y = random.randint(0, y_r - l_y)
            l_p_z = random.randint(0, z_r - l_z)
            img[l_p_x:l_x+l_p_x,
                l_p_y:l_y+l_p_y,
                l_p_z:l_z+l_p_z,
                :]=0
            # print('1',z_m_l)
            # print('2',l_p_x, l_x + l_p_x)
            # print('3',l_p_y, l_y + l_p_y)
            # print('4',l_p_z, l_z + l_p_z)
        return img

class RandomColor2rdScale3D(object):
    def __init__(self,max_scale):
        self.max_scale = max_scale
    def __call__(self, img):
        img=np.clip(img,0.0,1.0)
        scale_a = (self.max_scale-1)*random.random()+1
        if random.random()>0.5:
            scale_a = 1/scale_a
        scale_b = (self.max_scale-1)*random.random()+1
        if random.random()>0.5:
            scale_b = 1/scale_b

        img[:,:,:,0]=np.power(img[:,:,:,0],scale_a)
        img[:,:,:,1]=np.power(img[:,:,:,1],scale_b)

        return img