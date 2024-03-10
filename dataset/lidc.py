import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import glob
import h5py
import nibabel as nib
from PIL import Image
from dataset.transform_3d import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from skimage import measure
from xraysyn.networks.drr_projector_new import DRRProjector
from torch import nn

device = "cuda:0"
proj = DRRProjector(
    mode="forward", volume_shape=(128, 128, 128), detector_shape=(128, 128),
    pixel_size=(1.0, 1.0), interp="trilinear", source_to_detector_distance=1200).to(device)


class NormLayer(nn.Module):
  def __init__(self):
    super(NormLayer, self).__init__()

  def forward(self, inp):
    # print(inp.shape)
    inp = inp - inp.min()
    return inp / (inp.max() - inp.min())

def get_T(inp):
  param = np.asarray(inp)
  param = param * np.pi
  T = get_6dofs_transformation_matrix(param[3:], param[:3])
  T = torch.FloatTensor(T[np.newaxis, ...]).to('cuda')
  return torch.cat([T, T, T, T])


def get_6dofs_transformation_matrix(u, v):
  """ https://arxiv.org/pdf/1611.10336.pdf
  """
  x, y, z = u
  theta_x, theta_y, theta_z = v

  # rotate theta_z
  rotate_z = np.array([
    [np.cos(theta_z), -np.sin(theta_z), 0, 0],
    [np.sin(theta_z), np.cos(theta_z), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ])

  # rotate theta_y
  rotate_y = np.array([
    [np.cos(theta_y), 0, np.sin(theta_y), 0],
    [0, 1, 0, 0],
    [-np.sin(theta_y), 0, np.cos(theta_y), 0],
    [0, 0, 0, 1]
  ])

  # rotate theta_x and translate x, y, z
  rotate_x_translate_xyz = np.array([
    [1, 0, 0, x],
    [0, np.cos(theta_x), -np.sin(theta_x), y],
    [0, np.sin(theta_x), np.cos(theta_x), z],
    [0, 0, 0, 1]
  ])

  return rotate_x_translate_xyz.dot(rotate_y).dot(rotate_z)

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1   # 大于-320的为2，小于-320的为1
    labels = measure.label(binary_image)   # 连通域

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]

    # Fill the air around the person
    binary_image[background_label == labels] = 1

    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)

            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1

    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0
    print(binary_image.max())
    print(binary_image.min())

    return binary_image

def plot_3d(image, threshold=None):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)

    verts, faces,_,_ = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

class LIDCDataset(Dataset):
    def __init__(self, root_dir='E:/workspace/datasets/lidc_idri_true', augmentation=False):
        self.root_dir = root_dir
        self.file_names = glob.glob(os.path.join(
            root_dir, './*.nii'), recursive=True)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        img = nib.load(path)
        img = np.array(img.get_fdata())-1000
        segmented_lungs = segment_lung_mask(img, False)
        segmented_lungs_fill = segment_lung_mask(img, True)
        print(img.max())
        print(img.min())
        plt.hist(img.flatten(), bins=80, color='c')
        plt.xlabel("Hounsfield Units (HU)")
        plt.ylabel("Frequency")
        plt.show()
        plot_3d(img, 400)
        plot_3d(segmented_lungs, 0)
        img = (img-np.min(img))/(np.max(img) - np.min(img))
        if self.augmentation:
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img, 2)

        imageout = torch.from_numpy(img.copy()).float()
        imageout = imageout.permute(2,1,0)
        imageout = imageout.unsqueeze(0)

        return {'data': imageout}




class SingleDataGenerator(Dataset):
    def __init__(self, root_dir='E:/workspace/datasets', child_dir="LIDC", mode="train", cond=False, **kwargs):
        super(SingleDataGenerator, self).__init__()

        self.dataset = os.path.expanduser(os.path.join(root_dir, child_dir))
        self.mode = mode
        self.cond = cond

        if child_dir == "LIDC":
            self.anno_path = os.path.join('{}', "LIDC-HDF5-256", '{}', 'ct_xray_data.h5')
            if self.mode == "train":
                self.spilt_txt = "train"
            elif self.mode == "val":
                self.spilt_txt = "test"
            elif self.mode == "visual":
                self.spilt_txt = "visual"
            else:
                raise ValueError('Unkown self.mode')
        self.items = self.load_idx(self.spilt_txt)
        self.data_augmentation = CT_XRAY_Data_Augmentation()
        self.norm = NormLayer()


    def load_idx(self, spilt_txt):
        ids = []
        dataset = self.dataset
        txt_path = os.path.join(dataset, spilt_txt + '.txt')
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                ids += [(dataset, line.strip())]
        return ids

    def __getitem__(self, idx):
        img_idx = self.items[idx]
        img_item_path = self.anno_path.format(*img_idx)
        with h5py.File(img_item_path, "r") as f:
            ct = f['ct'][()]
            # ct = np.flip(ct, axis=0)
            # segmented_lungs = segment_lung_mask(ct, False)
            # segmented_lungs_fill = segment_lung_mask(ct, True)
            xray1 = f['xray1'][()]
            xray1 = np.expand_dims(xray1, axis=0)
            ct, xray1 = self.data_augmentation([ct, xray1])
            ct = ct.unsqueeze(dim=0)
            # T_in = get_T([0, 0, 0, 0, 0, 0])
            # ct = torch.unsqueeze(ct, dim=0)
            # ct = torch.unsqueeze(ct, dim=0).to('cuda')
            # ct = ct.transpose(4, 3)
            # ct = ct.contiguous()
            # xray = proj(ct, T_in)
            # xray = self.norm(xray)
            # ct = ct.squeeze(dim=0).cpu()
            #
            # from torchvision import transforms
            # tensor2img = transforms.ToPILImage()
            # xray = xray.squeeze(dim=0).cpu()
            # xray_test1 = torch.squeeze(xray).to("cpu")
            # print(xray_test1.shape)
            # im1 = tensor2img(xray_test1)
            # im1.show()

            if self.cond == True:
                return {"data": ct}
            else:
                return {'data': ct}

    def __len__(self):
        return len(self.items)

class CT_XRAY_Data_Augmentation(object):
  def __init__(self):
    self.augment = List_Compose([


      (Resize_image(size=(128, 128, 128)),
       Resize_image(size=(1, 64, 64))),

      (Limit_Min_Max_Threshold(0, 2500), None),


      (Normalization(0, 2500),
       Normalization(0, 255)),

      (Normalization_gaussian(0., 1.),
       Normalization_gaussian(0., 1.)),

      # (Get_Key_slice(opt.select_slice_num), None),

      (ToTensor(), ToTensor())

    ])



  def __call__(self, img_list):
    '''
    :param img: PIL image
    :param boxes: numpy.ndarray
    :param labels: numpy.ndarray
    :return:
    '''
    return self.augment(img_list)


if __name__ == '__main__':
    a = SingleDataGenerator()
    print(len(a))
    b = a[0]['data'][0]

    print(type(a[100]['data']))

    print(torch.min(b))
    toPIL = transforms.ToPILImage()
    print(b.shape)
    im = toPIL(b[30])
    # b = np.squeeze(a[0])
    # print(b)
    # im = Image.fromarray(a[0])
    im.show()
    # a[300].show()
