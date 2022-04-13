import matplotlib
matplotlib.use('Qt5Agg')
from albumentations import Normalize, HorizontalFlip, VerticalFlip, ShiftScaleRotate, ChannelShuffle, ColorJitter, GridDistortion, OpticalDistortion, ElasticTransform, CoarseDropout, Compose, OneOf, PadIfNeeded, CLAHE
from albumentations.pytorch.transforms import ToTensorV2
from test_funcs import *
from data_preporation import *
from maskrcnn import *
from fcn import *


#One picture = one type of cells
#Cell types: shsy5y[155], astro[131], cort[320]

def train(net, ds, n_epochs=20, batch_size=5, lr=1e-3, momentum=0.9, weight_decay=1e-5):

    dl_train = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device);  # remove semi-colon to see net structure

    # train the network
    # parameters related to training the network
    ### you will want to increase n_epochs!
    # number of times to cycle through all the data during training

    # where to save the network
    # make sure to clean these out every now and then, as you will run out of space
    now = datetime.now()
    timestamp = now.strftime('%Y%m%dT%H%M%S')

    # gradient descent flavor
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr,
                                weight_decay=weight_decay,
                                momentum=momentum)

    sheduler = np.linspace(0, lr, 10)
    sheduler = np.append(sheduler, lr * np.ones(n_epochs - 5))
    for i in range(5):
        sheduler = np.append(sheduler, sheduler[-1] / 2 * np.ones(10))

    criterion = DiceBCELoss()  ##DiceLoss()nn.BCELoss(weight=torch.as_tensor(9.271177944862155))#MixedLoss(10.0, 2.0)
    dicer = DiceLoss()
    scorer = F_Score()
    bcer = BCELoss()
    multer = DiceBCELoss()

    # Statistic
    epoch_train_mix_losses = np.zeros(n_epochs)
    epoch_train_mix_losses[:] = np.nan
    epoch_test_mix_losses = np.zeros(n_epochs)
    epoch_test_mix_losses[:] = np.nan
    # epoch_train_mix_losses = np.loadtxt(r"metrics/values/epoch_train_mix_losses.txt")
    # epoch_test_mix_losses = np.loadtxt(r"metrics/values/epoch_test_mix_losses.txt")

    epoch_train_BCE_losses = np.zeros(n_epochs)
    epoch_train_BCE_losses[:] = np.nan
    epoch_test_BCE_losses = np.zeros(n_epochs)
    epoch_test_BCE_losses[:] = np.nan
    # epoch_train_BCE_losses = np.loadtxt(r"metrics/values/epoch_train_BCE_losses.txt")
    # epoch_test_BCE_losses = np.loadtxt(r"metrics/values/epoch_test_BCE_losses.txt")

    epoch_train_dice_losses = np.zeros(n_epochs)
    epoch_train_dice_losses[:] = np.nan
    epoch_test_dice_losses = np.zeros(n_epochs)
    epoch_test_dice_losses[:] = np.nan
    # epoch_train_dice_losses = np.loadtxt(r"metrics/values/epoch_train_dice_losses.txt")
    # epoch_test_dice_losses = np.loadtxt(r"metrics/values/epoch_test_dice_losses.txt")

    epoch_train_accuracy = np.zeros(n_epochs)
    epoch_train_accuracy[:] = np.nan
    epoch_test_accuracy = np.zeros(n_epochs)
    epoch_test_accuracy[:] = np.nan
    # epoch_train_accuracy = np.loadtxt(r"metrics/values/epoch_train_accuracy.txt")
    # epoch_test_accuracy = np.loadtxt(r"metrics/values/epoch_test_accuracy.txt")

    epoch_train_precision = np.zeros(n_epochs)
    epoch_train_precision[:] = np.nan
    epoch_test_precision = np.zeros(n_epochs)
    epoch_test_precision[:] = np.nan
    # epoch_train_precision = np.loadtxt(r"metrics/values/epoch_train_precision.txt")
    # epoch_test_precision = np.loadtxt(r"metrics/values/epoch_test_precision.txt")

    epoch_train_recall = np.zeros(n_epochs)
    epoch_train_recall[:] = np.nan
    epoch_test_recall = np.zeros(n_epochs)
    epoch_test_recall[:] = np.nan
    # epoch_train_recall = np.loadtxt("D:\PycharmProjects\pythonProject\epoch_train_recall.txt")
    # epoch_test_recall = np.loadtxt("D:\PycharmProjects\pythonProject\epoch_test_recall.txt")

    epoch_train_f_score = np.zeros(n_epochs)
    epoch_train_f_score[:] = np.nan
    epoch_test_f_score = np.zeros(n_epochs)
    epoch_test_f_score[:] = np.nan
    # epoch_train_f_score = np.loadtxt(r"metrics/values/epoch_train_f_score.txt")
    # epoch_test_f_score = np.loadtxt(r"metrics/values/epoch_test_f_score.txt")

    start_epoch = 0
    batch_count = ds_train.__len__() / batch_size
    # Training Loop
    for epoch in range(start_epoch, 20):
        train_dice_losses = 0.0
        train_BCE_losses = 0.0
        train_mix_losses = 0.0
        train_accuracy = 0.0
        train_precision = 0.0
        train_recall = 0.0
        train_f_score = 0.0

        net.train()
        epoch_loss = 0
        iters = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = sheduler[epoch]
        with tqdm(total=ds_train.__len__(), desc=f"Epoch {epoch + 1}/{n_epochs}", unit='img') as pbar:
            for batch_idx, batch in enumerate(dl_train):
                images, masks = batch
                images = images.to(device=device)
                masks = masks.to(device=device)

                y = torch.sigmoid(net(images))
                train_dice = dicer(y, masks)
                train_bce = bcer(y, masks)
                loss = criterion(y, masks)

                train_dice_losses += train_dice.item()
                train_BCE_losses += train_bce.item()
                train_mix_losses += multer(y, masks)
                temp = scorer(y, masks)
                train_accuracy += temp[0]
                train_precision += temp[1]
                train_recall += temp[2]
                train_f_score += temp[3]

                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iters += 1
                pbar.update(masks.shape[0])

            epoch_train_dice_losses[epoch] = train_dice_losses / batch_count
            epoch_train_BCE_losses[epoch] = train_BCE_losses / batch_count
            epoch_train_mix_losses[epoch] = train_mix_losses / batch_count
            epoch_train_accuracy[epoch] = train_accuracy / batch_count
            epoch_train_precision[epoch] = train_precision / batch_count
            epoch_train_recall[epoch] = train_recall / batch_count
            epoch_train_f_score[epoch] = train_f_score / batch_count
            epoch_test_mix_losses[epoch], epoch_test_dice_losses[epoch], epoch_test_BCE_losses[epoch], \
            epoch_test_accuracy[epoch], \
            epoch_test_precision[epoch], epoch_test_recall[epoch], epoch_test_f_score[epoch] = net_test(net, df_train)

            pbar.set_postfix(**{'loss (epoch)': epoch_loss})

        plt.figure(figsize=(15, 15))

        plt.plot(epoch_train_mix_losses)
        plt.plot(epoch_test_mix_losses)
        plt.title('Mixed Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(r"./metrics/images/mixed_loss.jpg")
        plt.clf()

        plt.plot(epoch_train_dice_losses)
        plt.plot(epoch_test_dice_losses)
        plt.title('Dice Loss')
        plt.ylabel('coef')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(r"metrics/images/dice_loss.jpg")
        plt.clf()

        plt.plot(epoch_train_BCE_losses)
        plt.plot(epoch_test_BCE_losses)
        plt.title('Binary Cross-Entropy Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(r"metrics/images/cross_entropy_loss.jpg")
        plt.clf()

        plt.plot(epoch_train_accuracy)
        plt.plot(epoch_test_accuracy)
        plt.title('Accuracy')
        plt.ylabel('Coeficient')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(r"metrics/images/Accuracy.jpg")
        plt.clf()

        plt.plot(epoch_train_precision)
        plt.plot(epoch_test_precision)
        plt.title('Precision')
        plt.ylabel('Coeficient')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(r"metrics/images/Precision.jpg")
        plt.clf()

        plt.plot(epoch_train_recall)
        plt.plot(epoch_test_recall)
        plt.title('Recall')
        plt.ylabel('Coeficient')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(r"metrics/images/Recall.jpg")
        plt.clf()

        plt.plot(epoch_train_f_score)
        plt.plot(epoch_test_f_score)
        plt.title('F-Score')
        plt.ylabel('Coeficient')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(r"metrics/images/f_score.jpg")
        plt.clf()

        np.savetxt(r"metrics/values/epoch_train_mix_losses.txt", epoch_train_mix_losses, delimiter=',')
        np.savetxt(r"metrics/values/epoch_test_mix_losses.txt", epoch_test_mix_losses, delimiter=',')

        np.savetxt(r"metrics/values/epoch_train_dice_losses.txt", epoch_train_dice_losses, delimiter=',')
        np.savetxt(r"metrics/values/epoch_test_dice_losses.txt", epoch_test_dice_losses, delimiter=',')

        np.savetxt(r"metrics/values/epoch_train_BCE_losses.txt", epoch_train_BCE_losses, delimiter=',')
        np.savetxt(r"metrics/values/epoch_test_BCE_losses.txt", epoch_test_BCE_losses, delimiter=',')

        np.savetxt(r"metrics/values/epoch_train_accuracy.txt", epoch_train_accuracy, delimiter=',')
        np.savetxt(r"metrics/values/epoch_test_accuracy.txt", epoch_test_accuracy, delimiter=',')

        np.savetxt(r"metrics/values/epoch_train_precision.txt", epoch_train_precision, delimiter=',')
        np.savetxt(r"metrics/values/epoch_test_precision.txt", epoch_test_precision, delimiter=',')

        np.savetxt(r"metrics/values/epoch_train_recall.txt", epoch_train_recall, delimiter=',')
        np.savetxt(r"metrics/values/epoch_test_recall.txt", epoch_test_recall, delimiter=',')

        np.savetxt(r"metrics/values/epoch_train_f_score.txt", epoch_train_f_score, delimiter=',')
        np.savetxt(r"metrics/values/epoch_test_f_score.txt", epoch_test_f_score, delimiter=',')

        print(f"\nSaving network state at epoch {epoch + 1}")
        net.save_model(rf"epoch/{net.name}_epoch{epoch + 1}.pth")
        test_all_categories(net, rf"epoch/{net.name}_epoch{epoch + 1}.pth", df_train, epoch+1, show=False)

if __name__ == '__main__':
    binbucket = np.zeros(1)
    binsum = 0.0
    weights = np.asarray([ 29.9,0.28136258])
    weights = torch.tensor(weights, dtype=torch.float)

    df_train = pd.read_csv(TRAIN_CSV)
    ds_train = CellDataset(df_train, train=True, count=75)

    #net = UNet(n_channels=3, n_classes=1)
    net = UNet(n_channels=3,n_classes=1)

    # threshhold_fitting('unet_epoch27.pth',ds_train)
    #predict(net, rf'epoch/fcn_epoch1.pth', join(DATA_PATH, 'true_test\\2cab2cb161a4.png'),df_train,"predict.jpg", show=True)
    # predict(net,'unet_epoch14.pth', join(DATA_PATH, 'true_test\\2cab2cb161a4.png'), df_train,"predict.jpg",show=True)
    train(net=net,ds=ds_train)

    #net.load_model('unet_epoch16.pth')

