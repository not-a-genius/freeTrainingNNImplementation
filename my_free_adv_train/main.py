EPOCHS = 90
BATCH_SIZE = 256
IMG_SIZE = 256
CROP_SIZE = 224


 # Initiate data loaders
    traindir = os.path.join(configs.data, 'train')
    valdir = os.path.join(configs.data, 'val')

train_dataset = datasets.ImageFolder(
        traindir,

        #TODO see changes
        transforms.Compose([
            transforms.RandomResizedCrop(configs.DATA.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=configs.DATA.batch_size, shuffle=True,
    num_workers=configs.DATA.workers, pin_memory=True, sampler=None)


normalize = transforms.Normalize(mean=configs.TRAIN.mean,
                                std=configs.TRAIN.std)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(configs.DATA.img_size),
        transforms.CenterCrop(configs.DATA.crop_size),
        transforms.ToTensor(),
    ])),
    batch_size=configs.DATA.batch_size, shuffle=False,
    num_workers=configs.DATA.workers, pin_memory=True)
