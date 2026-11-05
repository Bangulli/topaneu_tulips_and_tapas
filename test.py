from pprint import pprint
from pathlib import Path
import os, datetime, torch, json, shutil, tqdm
from TnT.utils.dataloader import TopAneu_TnTs2_DS, TnTs2_collate, DataLoader, TopAneuDS
from TnT.utils.transforms import ClipCtaIntensities, RandomFlipLaterality, RandomDropCoordinates, LabelEncoder, NoisyCoordinates, DecodeAneu, DecodeVessel, Resample, RandomResample, RandomNonCorrespondingMask, RandomNonCorrespondingMorph, RandomMask, AdaNorm, Compose, MaybeToTensor, MaybeResize, BinarizeAneuChannel, BinarizeVesselChannel, ImageTransformWrapper
from TnT.model.stage2 import TnTS2, TnTS2_ViT
from TnT.model.stage1 import get_s1
from TnT.trainer.base import BasicTrainer
from TnT.trainer.n_fold import NFoldTrainer
from TnT.pipeline.inference import InferencePipeline
from TnT.evaluation.topaneu26 import TopAneu26LikeEvaluator
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
    PATCH_SIZE_VX = 64
    PATCH_SIZE_MM= 35
    BATCH_SIZE = 4
    EARLY_STOP_PATCHING = False
    pt_source = '/home/20260926/TnT/TnTS2_pretraining_from-12:53:24-27.10.26/best_val_loss'
    s1_model = "/home/20260926/TnT/stage1_models/nnUNetTrainer_single_encoder_baseline__nnUNetPlans__3d_fullres"
    
    ## Prep trans
    val_transforms = Compose([
        LabelEncoder(),
        MaybeToTensor(),
        ClipCtaIntensities(),
        MaybeResize(size=PATCH_SIZE_VX),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        AdaNorm.make(True),
    ])
    test_transforms = Compose([
            MaybeToTensor(),
            ClipCtaIntensities(),
            MaybeResize(size=PATCH_SIZE_VX),
            BinarizeAneuChannel(),
            BinarizeVesselChannel(),
            AdaNorm.make(True),
        ])
    train_transforms = Compose([
        LabelEncoder(),
        MaybeToTensor(),
        ClipCtaIntensities(),
        MaybeResize(PATCH_SIZE_VX),
        BinarizeAneuChannel(),
        BinarizeVesselChannel(),
        
        # ---- Spatial stuff ----
        RandomFlipLaterality(0.5),
        
        # ---- Coordinate stuff ----
        # NoisyCoordinates(0, 0.05, 0.05, 0.8),
        # RandomDropCoordinates(0.25),
        
        # # ---- Custom stuff ---- reduces perf
        # RandomMask(0.2),
        # RandomNonCorrespondingMask(0.2),
        # RandomNonCorrespondingMorph(0.2),

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
    
    train_scratch = 'tuning-train'
    val_scratch = 'tuning-val'
        
    ####################################### Experiment 1: train+val for train, test for val, no test
    if os.path.exists(train_scratch): shutil.rmtree(train_scratch)
    if os.path.exists(val_scratch): shutil.rmtree(val_scratch)
    ## load splits
    train = TopAneu_TnTs2_DS.load('train.json', train_transforms)
    val = TopAneu_TnTs2_DS.load('val.json', train_transforms)
    train = TopAneu_TnTs2_DS.join([train, val])
    train.preprocess(include_bg=False, include_s1_pred=False, syn_samples=False, max_items=1 if EARLY_STOP_PATCHING else -1)
    train.wdir = train_scratch
    # train.s1_predictor = get_s1(s1_model, 'cuda', use_mirroring=False)
    

    val = TopAneu_TnTs2_DS.load('test.json', val_transforms)
    val.preprocess(include_bg=False, include_s1_pred=False, syn_samples=False, max_items=1 if EARLY_STOP_PATCHING else -1)
    val.wdir = val_scratch
    
    for i in tqdm.tqdm(range(len(train)), desc="Caching train samples"):
        temp = train[i]
    for i in tqdm.tqdm(range(len(val)), desc="Caching val samples"):
        temp = val[i]
        
    train.s1_predictor = None
    val.s1_predictor = None
    
    ## PrEP DL
    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate, num_workers=6)
    val_dl = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, collate_fn=TnTs2_collate, num_workers=6)
    
    ## setup objs
    trainer = BasicTrainer(lr=5e-6)
    model = TnTS2(PATCH_SIZE_MM, PATCH_SIZE_VX)
    
    ## train or load
    wdir = Path(datetime.datetime.now().strftime(r'TnTS2_tuning_from-%H:%M:%S-%d.%m.%y'))
    model = trainer.train(model=model, train_dl=train_dl, val_dl=val_dl, epochs=1000, early_stop=100, use_aneu_class_balancing=False, wdir=wdir)#(model=model, ds=train, train_trans=train_transforms, val_trans=transforms, epochs=20, early_stop=5)

    ## E2E test
    ## Prep pipeline
    pl = InferencePipeline(s1_model, model, PATCH_SIZE_VX, PATCH_SIZE_MM, use_tta=True, use_mirroring=False, transforms=test_transforms)
    
    ## Prep data
    test = TopAneuDS.load('test.json')
    ## Prep evaluator
    outdir = Path(wdir)/'validation_test'
    os.makedirs(outdir, exist_ok=True)
    
    ev = TopAneu26LikeEvaluator(pl, outdir)
    res, agg, avg, disc = ev.eval_list(test.src, test.cases)
    with open(f'{outdir}/classification_failure.json', 'w') as f:
            json.dump(disc, f, indent=4)
    
    ev.plot(agg, f'{outdir}/per_clas')
    ev.re_eval_by_modality(res, outdir)
    with open(f'{outdir}/per_cls.json', 'w') as f:
        json.dump(avg, f, indent=4)