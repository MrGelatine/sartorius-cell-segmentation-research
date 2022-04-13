import matplotlib
matplotlib.use('Qt5Agg')
import collections
from losses import *

DATA_PATH = 'D:/CellPack'
SAMPLE_SUBMISSION = os.path.join(DATA_PATH, 'train')
TRAIN_CSV = os.path.join(DATA_PATH, 'train.csv')
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
TEST_PATH = os.path.join(DATA_PATH, 'true_test')
PCT_IMAGES_VALIDATION = 0.075
TEST = False
DEVICE = torch.device('cpu')

WIDTH = 704
HEIGHT = 520
#Decore OHE to array
def rle_decode(mask_rle, shape, color=1):

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    res = img.reshape(shape)
    return res
cell_type_dict = {"astro": 1, "cort": 2, "shsy5y": 3}
mask_threshold_dict = {1: 0.55, 2: 0.75, 3:  0.6}
min_score_dict = {1: 0.55, 2: 0.75, 3: 0.5}

#Show image,mask ant their intersetion
def display_masked_image(img, mask,pred_mask,final_mask,file_path,show):
    if len(img.shape) == 3:
        img = img[..., 0]
    dim = img.shape
    img = cv2.resize(img, (dim[0], dim[1]))
    #mask = cv2.resize(mask, (dim[0], dim[1]))
    #pred_mask = cv2.resize(pred_mask, (dim[0], dim[1]))

    plt.figure(figsize=(50, 50))
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(mask, cmap='plasma')
    plt.title('Ground-True Mask')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask, cmap='viridis')
    plt.title('Heat Map')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(final_mask, cmap='plasma')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.savefig(file_path)
    if(show):
        plt.show()
    else:
        plt.clf()

#Generate mask array from OHE
def build_masks(df_train, image_id, input_shape):
    height, width = input_shape
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
    mask = np.zeros((height, width))
    for label in labels:
        mask += rle_decode(label, shape=(height, width))
    mask = mask.clip(0, 1)
    return np.array(mask)

#Dataset implementation for UNet and FCN model
class CellDataset(Dataset):
    def __init__(self, df, train, count):
        self.IMAGE_RESIZE = (350, 350)
        self.RESNET_MEAN = (0.485, 0.456, 0.406)
        self.RESNET_STD = (0.229, 0.224, 0.225)
        self.df = df
        self.train = train
        if (train):
            self.base_path = TRAIN_PATH
        else:
            self.base_path = TEST_PATH
        self.gb = self.df.groupby('id')
        self.transforms = Compose([Resize(self.IMAGE_RESIZE[0], self.IMAGE_RESIZE[1]),
                                   Normalize(mean=self.RESNET_MEAN, std=self.RESNET_STD, p=1)])

        # Split train and val set
        all_image_ids = []
        for (dirpath, dirnames, filenames) in walk(self.base_path):
            for file in filenames:
                all_image_ids.append(file[:-4])
        all_image_ids = np.asarray(all_image_ids)
        iperm = np.random.permutation(len(all_image_ids))
        num_train_samples = count
        if(train):
            self.image_ids = all_image_ids[iperm[:num_train_samples]]
        else:
            self.image_ids = all_image_ids
        #for elem in train_image:
            #if(df[df.id == elem].iloc[0]['cell_type'] == "cort"):
                #self.train_image_ids.append(elem)
        #for elem in test_image:
            #if(df[df.id == elem].iloc[0]['cell_type'] == "cort"):
                #self.test_image_ids.append(elem)
        self.image_ids = np.asarray(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)

        # Read image
        image_path = os.path.join(self.base_path, image_id + ".png")
        image = cv2.imread(image_path)

        # Create the mask
        mask = build_masks(pd.read_csv(TRAIN_CSV), image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype('float32')
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return np.moveaxis(np.array(image), 2, 0), mask.reshape((1, self.IMAGE_RESIZE[0], self.IMAGE_RESIZE[1]))

    def __len__(self):
        return len(self.image_ids)

#Datset implementation for Mask R-CNN model

class CellDatasetMaskRCNN(Dataset):
    def __init__(self, image_dir, df, resize=False):
        self.IMAGE_RESIZE = (350, 350)
        self.RESNET_MEAN = (0.485, 0.456, 0.406)
        self.RESNET_STD = (0.229, 0.224, 0.225)

        self.transforms = Compose([Resize(self.IMAGE_RESIZE[0], self.IMAGE_RESIZE[1]),
                                   Normalize(mean=self.RESNET_MEAN, std=self.RESNET_STD, p=1)])
        self.image_dir = image_dir
        self.df = df

        self.should_resize = resize is not False
        # resize height and width of image
        if self.should_resize:
            self.height = int(HEIGHT * resize)
            self.width = int(WIDTH * resize)
        else:
            self.height = HEIGHT
            self.width = WIDTH

        # Creating a default dict - image_info
        # default dict can never raises key error
        # It provides a default value for the key that does not exists.
        self.image_info = collections.defaultdict(dict)
        # temp_df contain all annotations of particular image_id
        temp_df = self.df.groupby('id')['annotation'].agg(lambda x: list(x)).reset_index()

        # image_info dict will contain all info about particular image and its all annotations
        for index, row in temp_df.iterrows():
            self.image_info[index] = {
                'image_id': row['id'],
                'image_path': os.path.join(self.image_dir, row['id'] + '.png'),
                'annotations': row["annotation"]
            }

    def get_box(self, a_mask):
        ''' Get the bounding box of a given mask '''
        pos = np.where(a_mask)  # find out the position where a_mask=1
        xmin = np.min(pos[1])  # min pos will give min co-ordinate
        xmax = np.max(pos[1])  # max-position give max co-ordinate
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    def __getitem__(self, idx):
        ''' Get the image and the target'''

        img_path = self.image_info[idx]["image_path"]
        img = Image.open(img_path).convert("RGB")

        if self.should_resize:
            img = img.resize((self.width, self.height), resample=Image.BILINEAR)

        info = self.image_info[idx]
        n_objects = len(info['annotations'])  # no. of onjects present in an image
        # creating a masks of Zeros of shape(n_onjects,height,width)
        masks = np.zeros((len(info['annotations']), self.height, self.width), dtype=np.uint8)
        boxes = []

        # For each annotation create a mask image
        for i, annotation in enumerate(info['annotations']):
            a_mask = rle_decode(annotation, (HEIGHT, WIDTH))
            a_mask = Image.fromarray(a_mask)  # Creates an image memory from an object exporting the array interface

            # resizing the mask also
            if self.should_resize:
                a_mask = a_mask.resize((self.width, self.height), resample=Image.BILINEAR)

            a_mask = np.array(a_mask) > 0
            masks[i, :, :] = a_mask  # store the ith mask

            # finding the bounding box of respective mask for each annotation
            boxes.append(self.get_box(a_mask))

        # dummy labels
        labels = [1 for _ in range(n_objects)]

        # convert all into tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        # area=(xmax-xmin)*(ymax-ymin)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((n_objects,), dtype=torch.int64)

        # This is the required target for the Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_info)
