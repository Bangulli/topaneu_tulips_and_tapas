from TnT.model.stage2 import TnTS2
from TnT.utils.transforms import *
from TnT.utils.dataloader import TopAneuDS, TnTs2_collate, TopAneu_TnTs2_DS
from TnT.evaluation.topaneu26 import TopAneu26LikeEvaluator
from TnT.pipeline.inference import InferencePipeline
import json, os
import SimpleITK as sitk
from pathlib import Path
if __name__ == '__main__':    
    S1_path = '/home/tue20260926/Models/TopAneu-26/Stage1/nnUNetTrainer_single_encoder_baseline__nnUNetPlans__3d_fullres'
    S2_path = '/home/tue20260926/Repos/topaneu_tulips_and_tapas/cfg2 + pre + s1 TnTS2_training_from-13:48:40-04.09.26/best_val_loss'
    
    ## Prep pipeline
    pl = InferencePipeline(S1_path, S2_path, use_tta=True, use_mirroring=False)
    
    ## Prep data
    test = TopAneuDS.load('test.json')
    ## Prep evaluator
    outdir = Path(S2_path).parent/'s1_test'
    ev = TopAneu26LikeEvaluator(pl, outdir, precomp_predictions='/home/tue20260926/Data/TNT/s1/infersTs_altTrainer_best')
    res, agg, avg, disc = ev.eval_list(test.src, test.cases)
    with open(f'{outdir}/classification_failure.json', 'w') as f:
            json.dump(disc, f, indent=4)
    
    os.makedirs(outdir, exist_ok=True)
    ev.plot(agg, f'{outdir}/per_clas')
    ev.re_eval_by_modality(res, outdir)
    with open(f'{outdir}/per_cls.json', 'w') as f:
        json.dump(avg, f, indent=4)
