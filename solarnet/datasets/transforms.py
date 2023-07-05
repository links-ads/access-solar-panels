import albumentations as alb
from albumentations.pytorch import ToTensorV2
from solarnet.utils.transforms import SSLTransform


def adapt_channels(mean: tuple, std: tuple, in_channels: int = 3):
    assert mean is not None and std is not None, "Non-null means and stds required"
    if in_channels > len(mean):
        mean += (mean[0],)
        std += (std[0],)
    return mean, std


def classif_train_transforms(mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)) -> alb.Compose:
    return alb.Compose([
        alb.Flip(p=0.5),
        alb.ShiftScaleRotate(shift_limit=0.2, scale_limit=(0, 0.4), rotate_limit=90, p=0.5),
        alb.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
        alb.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        alb.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def classif_test_transforms(mean: tuple = (0.485, 0.456, 0.406), std: tuple = (0.229, 0.224, 0.225)) -> alb.Compose:
    return alb.Compose([alb.Normalize(mean=mean, std=std), ToTensorV2()])


def segment_train_transforms(
        image_size: int = 256,
        in_channels: int = 3,
        mean: tuple = (0.485, 0.456, 0.406),
        std: tuple = (0.229, 0.224, 0.225),
) -> alb.Compose:
    # if input channels are 4 and mean and std are for RGB only, copy red for IR
    mean, std = adapt_channels(mean, std, in_channels=in_channels)
    return alb.Compose([
        alb.RandomSizedCrop(min_max_height=(128, 256), height=image_size, width=image_size, p=0.5),
        alb.Flip(p=0.5),
        alb.RandomRotate90(p=0.5),
        alb.OneOf([
            alb.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
            alb.GridDistortion(distort_limit=0.3, p=0.5),
            alb.OpticalDistortion(distort_limit=0.3, shift_limit=0.2, p=0.5)
        ],
                  p=0.5),
        alb.OneOf([alb.GaussNoise(var_limit=(20, 60), p=0.5),
                   alb.Blur(blur_limit=(1, 3), p=0.5)], p=0.6),
        alb.RandomBrightnessContrast(p=0.5),
        alb.RandomGamma(p=0.5),
        alb.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


def segment_test_transforms(in_channels: int = 3,
                            mean: tuple = (0.485, 0.456, 0.406),
                            std: tuple = (0.229, 0.224, 0.225)) -> alb.Compose:
    mean, std = adapt_channels(mean, std, in_channels=in_channels)
    return alb.Compose([alb.Normalize(mean=mean, std=std), ToTensorV2()])


def ssl_train_transforms():
    return SSLTransform(alb.ReplayCompose([alb.RandomRotate90(always_apply=True),
                                           ToTensorV2()]),
                        track_params={0: "factor"})
