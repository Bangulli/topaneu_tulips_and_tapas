from pprint import pprint
from pathlib import Path
import os, datetime, torch, shutil
from tqdm import tqdm
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader, TopAneu_TnTs2_DS_for_vessel_pt
from TnT.utils.transforms import DropCoordinates, ClipCtaIntensities, RandomFlipLaterality, RandomDropCoordinates, LabelEncoder, NoisyCoordinates, DecodeAneu, DecodeVessel, Resample, RandomResample, RandomNonCorrespondingMask, RandomNonCorrespondingMorph, RandomMask, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2 import TnTS2, TnTS2Loss, TnTS2_ViT
from TnT.trainer.accelerated import AccelTrainer
from TnT.trainer.base import BasicTrainer
from monai.transforms import (
    RandAffined,
    RandAffine,
    RandFlip,
    RandRotate90,
    RandSpatialCrop,
    RandGaussianNoise,
    RandAdjustContrast,
    RandGaussianSmooth,
    RandScaleIntensity,
    RandShiftIntensity,
    RandBiasField,
    RandHistogramShift,
    NormalizeIntensity,
)

if __name__ == '__main__':
    PATCH_SIZE_VX = 128 # resizing in transforms
    PATCH_SIZE_MM = 70 # extracting from image
    PATCH_PER_VLOC = 8 # 4*36 = 144 samples per available image !! yields hella training samples
    EARLY_STOP_PATCHING = 2
    BATCH_SIZE = 6
    
    # ## Do splits
    # ds = TopAneu_TnTs2_DS("/home/20260926/datasets/TopAneu-26", patch_size_mm=PATCH_SIZE_MM)
    # folds = ds.split('0.8-0.1-0.1', 42)
    
    # for id, fold in zip(['train', 'test', 'val'], folds):
    #     fold.save(id)

    ## Prep trans
        ## Prep trans
    transforms = Compose([
        LabelEncoder(),
        MaybeToTensor(dtype=torch.float16),
        ClipCtaIntensities(),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        AdaNorm.make(True),
    ])
    
    train_transforms = Compose([
        LabelEncoder(),
        MaybeToTensor(),
        ClipCtaIntensities(),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        
        # ---- Spatial stuff ----
        RandomFlipLaterality(0.5),
        NoisyCoordinates(0, 0.05, 0.05, 0.8),

        # ---- Intensity-only transforms: image channel exclusively ----
        ImageTransformWrapper(
            RandGaussianNoise(prob=0.8, mean=0.0, std=100),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandGaussianSmooth(prob=0.15, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandAdjustContrast(prob=0.2, gamma=(0.7, 1.5)),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandScaleIntensity(prob=0.2, factors=0.1),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandShiftIntensity(prob=0.2, offsets=50),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandBiasField(prob=0.1, coeff_range=(0.0, 0.3)),
            apply_to=['image']
        ),
        ImageTransformWrapper(
            RandHistogramShift(prob=0.1, num_control_points=(3, 5)),
            apply_to=['image']
        ),
        AdaNorm.make(True),
    ])
    
    train_scratch = "pre-ves-train"
    val_scratch = 'pre-ves-val'
        
    ####################################### Experiment 1: train+val for train, test for val, no test
    if os.path.exists(train_scratch): shutil.rmtree(train_scratch)
    if os.path.exists(val_scratch): shutil.rmtree(val_scratch)
    
    if not os.path.exists('pre-ves-val.json'):
        val = TopAneu_TnTs2_DS_for_vessel_pt.load('val.json', transforms)
        val.preprocess(patches_per_vloc=1, max_items=1 if EARLY_STOP_PATCHING else -1)
        val.save('pre-ves-val.json')
    else: val = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-val.json', transforms)
    val.wdir = val_scratch
        
    if not os.path.exists('pre-ves-train.json'):
        train = TopAneu_TnTs2_DS_for_vessel_pt.load('train.json', train_transforms)
        train.preprocess(patches_per_vloc=PATCH_PER_VLOC, max_items=1 if EARLY_STOP_PATCHING else -1)
        train.save('pre-ves-train.json')
    else: train = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-train.json', train_transforms)
    train.wdir = train_scratch
    
    if not os.path.exists('pre-ves-test.json'):
        test = TopAneu_TnTs2_DS_for_vessel_pt.load('test.json', transforms)
        test.preprocess(patches_per_vloc=1, max_items=1 if EARLY_STOP_PATCHING else -1)
        test.save('pre-ves-test.json')
    else: test = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-test.json', transforms)
    
    train.patch_size_mm=PATCH_SIZE_MM
    val.patch_size_mm=PATCH_SIZE_MM
    test.patch_size_mm=PATCH_SIZE_MM
    train.patch_size_vx=PATCH_SIZE_VX
    val.patch_size_vx=PATCH_SIZE_VX
    test.patch_size_vx=PATCH_SIZE_VX
    
    for i in tqdm(range(len(train)), desc="Caching train samples"):
        temp = train[i]
    for i in tqdm(range(len(val)), desc="Caching val samples"):
        temp = val[i]
    
    ## PrEP DL
    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate, num_workers=6)
    val_dl = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate, num_workers=6)
    test_dl = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, collate_fn=TnTs2_collate, num_workers=6)
    
    ## setup objs
    trainer = BasicTrainer(lr=1e-3, prec="half")
    model = TnTS2()
    loss = TnTS2Loss(model.n_locs_v, model.n_locs_a)
    
    ## train or load
    model = trainer.train(model, train_dl, val_dl, 200, 10, wdir=Path(datetime.datetime.now().strftime(r'TnTS2_pretraining_from-%H:%M:%S-%d.%m.%y')), loss=loss)

    ## QnD test
    # print('#'*20, 'Training ACC', '#'*20)
    # train = TopAneu_TnTs2_DS_for_vessel_pt.load('pre-ves-train.json', transforms)
    # train_dl = DataLoader(train, batch_size=4, shuffle=True)
    # acc = trainer.test(model, train_dl, decoder=DecodeTargetForVessels())
    # print('#'*20, 'Validation ACC', '#'*20)
    # acc = trainer.test(model, val_dl, decoder=DecodeTargetForVessels())
    # print('#'*20, 'Testing ACC for Aneu', '#'*20)
    # acc = trainer.test(model, test_dl, decoder=DecodeAneu(), target='aneu')
    print('#'*20, 'Testing ACC for Vessel', '#'*20)
    acc = trainer.test(model, test_dl, decoder=DecodeVessel(), target='vessel')