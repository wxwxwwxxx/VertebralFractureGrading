import logging
import os
import time
import SimpleITK as sitk
import numpy as np
import torchvision.transforms

np.seterr(all="ignore")
import torch.utils.data as data
import yaml
from radiomics import featureextractor
from torchvision import transforms
import torch
import transform as custom_transform
import random
import torch.nn.functional as F
class Vertebrae_Dataset(data.Dataset):
    def __init__(self, dataset_path, file_list_yaml, transforms=None):

        self.dataset_path = dataset_path


        # self.data_list = os.listdir(os.path.join(self.dataset_path, 'img'))
        if isinstance(file_list_yaml,list):
            pass
        elif isinstance(file_list_yaml,str):
            file_list_yaml = [file_list_yaml]
        else:
            raise SystemExit("Unknown yaml,shutting down...")
        self.tag = file_list_yaml[0].split('.')[-2]
        print(self.tag)
        yaml_dump = ""
        for i in file_list_yaml:
            # Use first fn as tag
            # if i.split('.')[-2] != self.tag:
            #     print(f"Warning: Unconsistent dataset tag,{self.tag} and {i.split('.')[-2]}")
            with open(os.path.join(self.dataset_path, i)) as list_file:
                yaml_dump += list_file.read()
        yaml_cache = yaml.load(yaml_dump, yaml.Loader)
        self.data_dict = None
        self.data_list = None
        if isinstance(yaml_cache, dict):
            self.data_dict = yaml.load(yaml_dump, yaml.Loader)
            self.data_list = list(self.data_dict.keys())
        elif isinstance(yaml_cache, list):
            self.data_list = yaml.load(yaml_dump, yaml.Loader)
        else:
            raise SystemExit("Unknown yaml,shutting down...")

        self.transforms = transforms

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.get_img(index)
        elif isinstance(index, int):
            return self.get_img(self.data_list[index])

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.data_list)

    def get_img(self, fn):
        img_sitk, seg_sitk = self.load_sitk_img(fn)
        label, v_index = self.get_label(fn)
        output_img = self.get_img_with_preprocess(img_sitk, seg_sitk, v_index)
        return output_img, label, v_index

    def normalize(self, img_arr, ww, wl):
        half_ww = ww // 2
        a_max = wl + half_ww
        a_min = wl - half_ww
        img_arr = np.clip(img_arr, a_min=a_min, a_max=a_max)
        img_arr -= a_min
        img_arr = img_arr / (a_max - a_min)
        return img_arr

    def load_sitk_img(self, fn):
        img = sitk.ReadImage(os.path.join(self.dataset_path, "img", fn))
        seg = sitk.ReadImage(os.path.join(self.dataset_path, "seg", fn))
        return img, seg

    def get_label(self, fn):
        fn_name = fn.split('.')[0]

        v_tag = fn_name.split('_')[-2]

        g_level = fn_name.split('_')[-1]

        label = int(g_level)
        v_index = int(v_tag)
        return label, v_index

    def get_img_with_preprocess(self, img, seg, v_index):


        img_arr = sitk.GetArrayFromImage(img)
        img_arr = np.transpose(img_arr, [2, 1, 0])

        seg_arr = sitk.GetArrayFromImage(seg)
        seg_arr = np.transpose(seg_arr, [2, 1, 0])

        v_w = self.normalize(img_arr, 200, 40)
        v_w = v_w[:, :, :, None]

        b_w = self.normalize(img_arr, 1500, 400)
        b_w = b_w[:, :, :, None]

        seg_arr = seg_arr[:, :, :, None]

        output_img = np.concatenate([v_w, b_w, seg_arr], axis=-1)
        output_img = output_img.astype('float32')
        if isinstance(self.transforms,list):

            output_img1 = self.transforms[0](output_img)
            output_img2 = self.transforms[1](output_img)
            output_img = np.concatenate([output_img1,output_img2])
        elif self.transforms is not None:
            output_img = self.transforms(output_img)
        scale = np.array([1.0, 1.0, float(v_index) / 28.0]).astype('float32')
        scale = scale[None, None, None, :]
        output_img = output_img * scale
        output_img = output_img.transpose((3, 2, 0, 1))

        return output_img

class Radiomics_Dataset(Vertebrae_Dataset):
    def __init__(self, *args, **kwargs):
        # set level for all classes
        # logger = logging.getLogger("radiomics")
        # logger.setLevel(logging.ERROR)
        # ... or set level for specific class
        logger = logging.getLogger("radiomics.glcm")
        logger.setLevel(logging.ERROR)
        if 'radiomics_setting' in kwargs:
            self.setting = kwargs['radiomics_setting']
            del kwargs['radiomics_setting']
        else:
            self.setting = {}
        super().__init__(*args, **kwargs)
        self.extractor = None
        self.builtin_max = np.load(os.path.join( self.dataset_path,f"radiomics.{self.tag}", "r_max.npy"))
        self.builtin_min = np.load(os.path.join( self.dataset_path,f"radiomics.{self.tag}", "r_min.npy"))
        self.builtin_mask = np.load(os.path.join( self.dataset_path,f"radiomics.{self.tag}", "f_index_v2.npy"))
        self.builtin_mean_n = np.load(os.path.join( self.dataset_path,f"radiomics.{self.tag}", "r_mean_n.npy"))
        self.builtin_var_n = np.load(os.path.join( self.dataset_path,f"radiomics.{self.tag}", "r_var_n.npy"))

        print(f"Feature Mask Length: {len(self.builtin_mask)}")

    def feature_length(self):
        return len(self.builtin_mask)

    def radiomics_feature(self, img_sitk, seg_sitk):
        if self.extractor is None:
            self.extractor = featureextractor.RadiomicsFeatureExtractor(**self.setting)
            self.extractor.addProvenance(False)
            self.extractor.disableAllFeatures()
            feature_class_list = ["firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]
            img_type_list = ["Original", "Wavelet", "LoG", "Square", "SquareRoot", "Logarithm", "Exponential",
                             "Gradient", "LBP3D"]
            for f in feature_class_list:
                self.extractor.enableFeatureClassByName(f)
            for i in img_type_list:
                self.extractor.enableImageTypeByName(i)

        featureVector = self.extractor.execute(img_sitk, seg_sitk)
        featureVector = np.array(list(featureVector.values()), dtype=np.float32)
        return featureVector

    def radiomics_feature_from_cache(self, fn):
        featureVector = np.load(os.path.join(self.dataset_path, "radiomics", f"{fn}.npy"))
        return featureVector

    def get_img(self, fn):
        img_sitk, seg_sitk = self.load_sitk_img(fn)

        # use cached radiomics
        radiomics_feature = self.radiomics_feature_from_cache(fn)
        radiomics_feature = self.radiomics_normalize(radiomics_feature)
        radiomics_feature = self.radiomics_noise(radiomics_feature, 0.1)
        label, v_index = self.get_label(fn)
        output_img = self.get_img_with_preprocess(img_sitk, seg_sitk, v_index)

        return output_img, radiomics_feature, label, v_index

    def preprocess(self, save_path):
        # file_list = os.listdir(os.path.join(self.dataset_path, "img"))
        file_list = self.data_list
        os.makedirs(save_path, exist_ok=True)
        for i in file_list:
            img, seg = self.load_sitk_img(i)
            r_f = self.radiomics_feature(img, seg)
            print(i, np.shape(r_f))
            np.save(os.path.join(save_path, i), r_f)

    def radiomics_noise(self, radiomics_feature, ratio):
        r_std_noise = np.random.normal(0, 1, radiomics_feature.size)
        r_std_noise = r_std_noise * (np.sqrt(self.builtin_var_n[self.builtin_mask]) * ratio * 0.333) # 3 sigma
        radiomics_feature += r_std_noise
        return radiomics_feature

    def radiomics_standardize(self, radiomics_feature):
        r = (radiomics_feature - self.builtin_mean) / self.builtin_var
        r = r[self.builtin_mask]
        return r
    def radiomics_normalize(self, radiomics_feature):
        r = (radiomics_feature - self.builtin_min) / (self.builtin_max - self.builtin_min)
        r = r[self.builtin_mask]
        return r

class Radiomics_Dataset_Dumb_IMG(Radiomics_Dataset):
    def get_img_with_preprocess(self, img_sitk, seg_sitk, v_index):
        return 0

class Vertebrae_Dataset_with_fn(Vertebrae_Dataset):
    def get_img(self, fn):
        ret = super().get_img(fn=fn)
        l_ret = [i for i in ret]
        l_ret.append(fn)
        # if fn[0:3] == "sub":
        #     l_ret[1] += 4
        return l_ret

class CustomBatchSampler:
    def __init__(self, sampler, data):
        self.sampler = sampler

        self.data_list = data.data_list
        self.data_dict = data.data_dict

    def __iter__(self):
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch = self.data_dict[self.data_list[idx]]
            yield batch

    def __len__(self):
        return len(self.sampler)

class ContrastiveBatchSampler:
    def __init__(self,  data, batchsize_per_grade=3):
        #print("debug_sign_init")
        self.sampler_list = [1,2,3]
        self.data_list = data.data_list
        self.grade_list = [[],[],[],[]]
        self.len_list = []
        self.bs = batchsize_per_grade
        for i in self.data_list:
            g=data.get_label(i)[0]
            self.grade_list[g].append(i)
        for i in range(4):
            self.len_list.append(len(self.grade_list[i]))
    def __iter__(self):
        #print("debug_sign_iter")
        for i in range(4):
            random.shuffle(self.grade_list[i])
        for i in range(0,min(self.len_list),self.bs):
            batch = [x for l in self.grade_list for x in l[i:i+self.bs]]
            random.shuffle(batch)
            yield batch
    def __len__(self):
        return min(self.len_list)//self.bs

if __name__ == "__main__":

    # with open("/dataset/train_acc_dict.X6wnciif.yaml") as list_file:
    #     yaml_dump = list_file.read()
    # data_dict = yaml.load(yaml_dump, yaml.Loader)
    # data_list = list(data_dict.keys())
    #
    # print(len(data_dict[data_list[3]]))
    #
    # print(len(data_dict))
    # exit()
    t = transforms.Compose([
        custom_transform.RandomMask3D(10, 2, 0.5),
        custom_transform.RandomColorScale3D(0.1),
        custom_transform.RandomNoise3D(0.05),
        custom_transform.RandomRotation3D(10),
        custom_transform.RandomZoom3D(0.2),
        custom_transform.RandomShift3D(10),
        custom_transform.RandomAlign3D(128),
        custom_transform.RandomColor2rdScale3D(2),
        custom_transform.RandomMask3D(10, 2, 0.5)
    ])

    t = transforms.Compose([
        custom_transform.RandomMask3D(20, 2, 0.5),
        transforms.RandomApply([
        custom_transform.RandomColorScale3D(0.1),
        custom_transform.RandomNoise3D(0.05),
        custom_transform.RandomRotation3D(10),
        custom_transform.RandomZoom3D(0.2),
        custom_transform.RandomShift3D(10),
        ],
            p=0.7),
        custom_transform.RandomAlign3D(128),
        custom_transform.RandomMask3D(20, 2, 0.5)
    ])
    #train_data = Vertebrae_Dataset_with_fn("/dataset", "test_file_list.delx.yaml", transforms=t)

    # train_data.preprocess("/ckpt/radiomics_preprocess_xt3")
    # t = transforms.Compose([
    #     custom_transform.RandomColorScale3D(0.1),
    #     custom_transform.RandomRotation3D(10),
    #     custom_transform.RandomZoom3D(0.2),
    #     custom_transform.RandomShift3D(10),
    #     custom_transform.RandomAlign3D(128)
    # ])
    #


    train_data = Vertebrae_Dataset_with_fn("/dataset", "train_file_list.WA0aVG88.yaml", transforms=[t,t])
    #asampler = data.RandomSampler(train_data)
    custom_batch_sample = ContrastiveBatchSampler(train_data)
    trainloader = data.DataLoader(train_data, num_workers=1, batch_sampler=custom_batch_sample)

    from playground import SupConLoss
    from utils import img_plot
    c = SupConLoss()

    for z in range(10):

        for i, data in enumerate(trainloader, 0):
            print(i)
            i1,i2,i3,i4 = data

            features = torch.rand(12,2048)
            features = F.normalize(features, dim=1)
            print(features)
            f1, f2 = torch.split(i1,[128,128] , dim=3)

            i1 = torch.cat([f1, f2])
            i2 = torch.cat([i2, i2])
            i3 = torch.cat([i3, i3])

            print(i1.size())
            print(f1.size())
            print(f2.size())
            img_plot(f1)
            img_plot(f1[:,2:3,:,:,:])
            mask = f1[1,2,64,:,:].cpu().numpy()
            print(f1[1,2:3,0:64,0:64,0:64])
            #img_plot(f2)
            print(i2)
            print(i3)
            # print(c(features,i2[0:6]))


    #
    # test_data = Radiomics_Dataset_Dumb_IMG("/dataset", "test_acc_dict.WA0aVG88.yaml", transforms=t)
    # bsampler = data.SequentialSampler(test_data)
    # bcustom_batch_sample = CustomBatchSampler(bsampler, test_data)
    # testloader = data.DataLoader(test_data, num_workers=1, batch_sampler=bcustom_batch_sample)
    # mmax = 0
    # mmin = 100
    # for i in trainloader:
    #     i1,i2,i3,i4 = i
    #     s = i2.numpy()
    #
    #     #print(i1.size(),i2.size(),i3.size(),i4.size())
    #     # print(torch.min(i4).item(), torch.max(i4).item())
    #     print( torch.max(i4).item() - torch.min(i4).item() + 1 , i4.size()[0])
    #     if torch.max(i4).item()==28:
    #         print(i4)
    #     if torch.max(i4).item()-torch.min(i4).item()+1 != i4.size()[0]:
    #         print("warning:")
    #         print(i4)
    #     if mmax<torch.max(i4).item():
    #         mmax = torch.max(i4).item()
    #     if mmin>torch.min(i4).item():
    #         mmin = torch.min(i4).item()
    # for i in testloader:
    #     i1,i2,i3,i4 = i
    #     s = i2.numpy()
    #
    #     #print(i1.size(),i2.size(),i3.size(),i4.size())
    #     # print(torch.min(i4).item(), torch.max(i4).item())
    #     if torch.max(i4).item()==28:
    #         print(i4)
    #     if mmax < torch.max(i4).item():
    #         mmax = torch.max(i4).item()
    #     if mmin > torch.min(i4).item():
    #         mmin = torch.min(i4).item()
    # print(mmax, mmin)



