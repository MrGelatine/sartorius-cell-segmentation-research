from data_preporation import *
from losses import *
IMAGE_RESIZE = (350, 350)
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)
transforms = Compose([Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]),
                               Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1)])
#Model prediction
def predict(model,weight_path, image_path, df_train):
    model.load_model(weight_path)

    image = cv2.imread(image_path)
    mask = build_masks(df_train, image_path.split('\\')[-1][:-4], input_shape=(520, 704))
    augmented = transforms(image=image, mask=mask)
    image = augmented['image']
    mask = augmented['mask']

    prediction = torch.sigmoid(
        model(torch.as_tensor(np.moveaxis(np.array(image), 2, 0).reshape((1, 3, IMAGE_RESIZE[0], IMAGE_RESIZE[1])))))
    prediction = prediction.cpu().detach().numpy()
    prediction = np.asarray(prediction).reshape((IMAGE_RESIZE[0], IMAGE_RESIZE[1]))

    res = (np.asarray(prediction) > 0.85).astype(int)
    res = res.reshape((IMAGE_RESIZE[0], IMAGE_RESIZE[1]))
    return (image, mask, prediction, res)

#Prediction for examples of all cell types from dataset
def test_all_categories(net, weight_path, df_train, epoch, show):
    shsh5 = predict(net, weight_path, join(DATA_PATH, 'true_test\\5b8e5ee1ec61.png'), df_train)
    neuro_blastoma = predict(net, weight_path, join(DATA_PATH, 'true_test\\0a6ecc5fe78a.png'),
                             df_train)
    cort = predict(net, weight_path, join(DATA_PATH, 'true_test\\0cfdeeb0dded.png'), df_train)
    dim = cort[0].shape

    plt.figure(figsize=(50, 50))
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.resize(shsh5[0][..., 0], (dim[0], dim[1])), cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(shsh5[1], cmap='plasma')
    plt.title('Ground-True Mask')
    plt.axis('off')

    temp = shsh5[2]
    plt.subplot(3, 4, 3)
    plt.imshow(shsh5[2], cmap='plasma')
    plt.title('Heat Map')
    plt.axis('off')

    plt.subplot(3, 4, 4)
    plt.imshow(shsh5[3], cmap='plasma')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(3, 4, 5)
    plt.imshow(cv2.resize(neuro_blastoma[0][..., 0], (dim[0], dim[1])), cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(neuro_blastoma[1], cmap='plasma')
    plt.title('Ground-True Mask')
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(neuro_blastoma[2], cmap='plasma')
    plt.title('Heat Map')
    plt.axis('off')

    plt.subplot(3, 4, 8)
    plt.imshow(neuro_blastoma[3], cmap='plasma')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(3, 4, 9)
    plt.imshow(cv2.resize(cort[0][..., 0], (dim[0], dim[1])), cmap='gray')
    plt.title('Image')
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(cort[1], cmap='plasma')
    plt.title('Ground-True Mask')
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.imshow(cort[2], cmap='plasma')
    plt.title('Heat Map')
    plt.axis('off')

    plt.subplot(3, 4, 12)
    plt.imshow(cort[3], cmap='plasma')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.savefig(f"{epoch}.jpg")
    if (show):
        plt.show()
    else:
        plt.clf()

#Test model work on validation set
def net_test(net,df_train):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        test_dice = 0.0
        test_bce = 0.0
        test_multi = 0.0
        test_accuracy = 0.0
        test_precision = 0.0
        test_recall = 0.0
        test_f_score = 0.0

        dicer = DiceLoss()
        bcer = BCELoss()
        multer = DiceBCELoss()
        scorer = F_Score()

        test_dataset = CellDataset(df_train, train=False, count=75)
        dl_test = DataLoader(test_dataset, batch_size=5, num_workers=2, pin_memory=False, shuffle=False)
        test_batch_counter = test_dataset.__len__() / 5.0
        for batch_idx, batch in enumerate(dl_test):
            # transfer to torch + GPU
            images, masks = batch

            images = images.to(device=device)
            masks = masks.to(device=device)

            # transfer to torch + GPU
            # compute the loss
            y = torch.sigmoid(net(images))

            test_dice += dicer(y, masks)
            test_bce += bcer(y, masks)
            test_multi += multer(y, masks)
            temp = scorer(y, masks)
            test_accuracy += temp[0]
            test_precision += temp[1]
            test_recall += temp[2]
            test_f_score += temp[3]

        test_multi = test_multi / test_batch_counter
        test_dice = test_dice / test_batch_counter
        test_bce = test_bce / test_batch_counter
        test_accuracy = test_accuracy / test_batch_counter
        test_precision = test_precision / test_batch_counter
        test_recall = test_recall / test_batch_counter
        test_f_score = test_f_score / test_batch_counter

        return (test_multi, test_dice, test_bce, test_accuracy, test_precision, test_recall, test_f_score)

#Tunning decent threshold value
def threshold_fitting(model_path, data_loader):
    with torch.no_grad():
        kernel_size = 3
        nbase = [3, 32, 64, 128, 256]  # number of channels per layer
        nout = 1  # number of outputs
        model = UNet(n_channels=3, n_classes=1)
        model.load_model(model_path)
        final_threshold = []
        data_loader.train = False
        dl_train = DataLoader(data_loader, batch_size=50, num_workers=2, pin_memory=False, shuffle=False)
        conter = 0
        threshholds_losses = []
        for threshhold in range(30, 100, 10):
            print(threshhold)
            loss = 0.0
            for batch_idx, batch in enumerate(dl_train):
                images, masks = batch
                # transfer to torch + GPU
                # compute the loss
                y = torch.sigmoid(model(images)).cpu().detach().numpy()
                true_y = masks.cpu().detach().numpy()
                prediction = ()
                # prediction = np.argmax(y,axis=1)

                loss += np.absolute(
                    ((y > (threshhold * 0.01)).astype(int) - (true_y > (threshhold * 0.01)).astype(int))).sum()
            threshholds_losses.append(loss)
        print(threshholds_losses)
        print(30 + np.argmin(np.asarray(threshholds_losses)) * 10.0)