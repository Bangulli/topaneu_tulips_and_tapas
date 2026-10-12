from pprint import pprint
from pathlib import Path
import os, datetime, torch, json
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader, TopAneuDS
from TnT.utils.transforms import get_train_test_transforms, DecodeAneu, DecodeVessel, Resample, RandomResample, RandomNonCorrespondingMask, RandomNonCorrespondingMorph, RandomMask, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2 import TnTS2
from TnT.model.stage1 import get_s1
from TnT.trainer.base import BasicTrainer
from TnT.trainer.n_fold import NFoldTrainer
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
    PATCH_SIZE_VX = 64 # to avoid oom error on local
    BATCH_SIZE = 4
    EARLY_STOP_PATCHING = False
    
    ## Prep trans
    train_transforms, transforms = get_train_test_transforms(PATCH_SIZE_VX, False)
    
    ## load splits
    if not os.path.exists('tuning-train.json'):
        train = TopAneu_TnTs2_DS.load('train.json', train_transforms)
        #train.s1_predictor = get_s1('/home/tue20260926/Models/TopAneu-26/Stage1/nnUNetTrainer_single_encoder_mixed__nnUNetPlans__3d_fullres', 'cuda')
        # with open('syn_data_recipe.json', 'r') as f:
        #     syn = json.load(f)
        train.preprocess(include_bg=False, include_s1_pred=False, syn_samples=False, max_items=1 if EARLY_STOP_PATCHING else -1)
        train.save('tuning-train.json')
    else: train = TopAneu_TnTs2_DS.load('tuning-train.json', train_transforms); train.s1_predictor = get_s1('/home/tue20260926/Models/TopAneu-26/Stage1/nnUNetTrainer_single_encoder_mixed__nnUNetPlans__3d_fullres', 'cuda')
    train.wdir = 'tuning-train'
    
    if not os.path.exists('tuning-val.json'):
        val = TopAneu_TnTs2_DS.load('val.json', transforms)
        val.preprocess(max_items=1 if EARLY_STOP_PATCHING else -1)
        val.save('tuning-val.json')
    else: val = TopAneu_TnTs2_DS.load('tuning-val.json', transforms)
    val.wdir = 'tuning-val'   
    
    ## PrEP DL
    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate)
    val_dl = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate)
    
    ## setup objs
    trainer = BasicTrainer(lr=3e-4)
    model = TnTS2()
    
    ## train or load
    model = trainer.train(model=model, train_dl=train_dl, val_dl=val_dl, epochs=20, early_stop=5, use_aneu_class_balancing=False)#(model=model, ds=train, train_trans=train_transforms, val_trans=transforms, epochs=20, early_stop=5)

    ## QnD test
    test = TopAneuDS.load('test.json')
    print('#'*20, 'Testing TopAneu performance', '#'*20)
    acc = trainer.test_TopAneu(model, test)
    if not os.path.exists('tuning-test.json'):
            test = TopAneu_TnTs2_DS.load('test.json', transforms)
            test.preprocess(max_items=1 if EARLY_STOP_PATCHING else -1)
            test.save('tuning-test.json')
    else: test = TopAneu_TnTs2_DS.load('tuning-test.json', transforms)
    acc = trainer.test(model, DataLoader(test, batch_size=1, shuffle=False, collate_fn=TnTs2_collate))