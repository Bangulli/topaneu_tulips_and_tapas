from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import cross_entropy, binary_cross_entropy, softmax
from pathlib import Path
import os, tqdm, torch, numpy as np, shutil, datetime, json
from TnT.utils.transforms import DecodeAneu
import matplotlib.pyplot as plt
from TnT.trainer.metrics import loc_lat_cls_acc, LossHistory
import TnT.trainer.class_weigths as cw
from TnT.pipeline.inference import InferencePipeline
from TnT.evaluation.topaneu26 import TopAneu26LikeEvaluator

class BasicTrainer():
    def __init__(self, lr=1e-4, optim = Adam, sched = CosineAnnealingLR, device='cuda', prec='half', wandb_cfg=None):
        self.lr = lr
        self.optim = optim
        self.sched = sched
        self.device = device
        self.wdir=None
        self.prec = torch.float16 if prec=='half' else torch.float32
        self.wandb_cfg = wandb_cfg # kwargs for wandb.init, e.g. {'project': 'TnT', 'group': 'modspec', 'tags': ['CT']}; None disables wandb
        self.run = None
        
    def _init_wandb(self, model, train_dl, val_dl, epochs, early_stop, wdir, use_aneu_class_balancing, loss):
        if self.wandb_cfg is None: return
        import wandb
        self.finish()
        cfg = {
            'model': type(model).__name__,
            'model_parent': str(model.parent),
            'epochs': epochs,
            'early_stop': early_stop,
            'lr': self.lr,
            'optimizer': type(self.optim).__name__,
            'scheduler': type(self.sched).__name__,
            'loss': 'model.loss' if loss is None else type(loss).__name__,
            'device': self.device,
            'precision': str(self.prec),
            'aneu_class_balancing': use_aneu_class_balancing,
            'batch_size': train_dl.batch_size,
            'n_train': len(train_dl.dataset),
            'n_val': len(val_dl.dataset),
            'train_transforms': str(train_dl.dataset.transforms),
            'val_transforms': str(val_dl.dataset.transforms),
            'wdir': str(wdir),
        }
        kwargs = {'name': str(wdir), 'dir': wdir, **self.wandb_cfg}
        kwargs['config'] = {**cfg, **kwargs.get('config', {})}
        self.run = wandb.init(**kwargs)
        self.run.define_metric('epoch')
        self.run.define_metric('epoch/*', step_metric='epoch')
    
    def finish(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
        
    def _save_train_cfg(self, model, train_dl, val_dl, epochs, early_stop, wdir, use_aneu_class_balancing):
        with open(wdir/'train_cfg.txt', 'w') as f:
            f.write(f"Model: {model}\n")
            f.write(f"Model parent: {model.parent}\n")
            f.write(f"Epochs: {epochs}\n")
            if early_stop is not None: f.write(f"Early stopping after {early_stop} epochs of no improvement\n")
            else: f.write(f"Early stopping disabled\n")
            f.write(f"Learningrate: {self.lr}\n")
            f.write(f"Learningrate Scheduler: {self.sched}\n")
            f.write(f"Optimizer: {self.optim}\n")
            f.write(f"Loss: model.loss function\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Using Aneurysm Class weight balancing: {use_aneu_class_balancing}\n")
            f.write(f"Train Transforms: {train_dl.dataset.transforms}\n")
            f.write(f"Val Transforms: {val_dl.dataset.transforms}\n")
        
    def train(self, model, train_dl, val_dl, epochs=1, early_stop=5, wdir=Path(datetime.datetime.now().strftime(r'TnTS2_training_from-%H:%M:%S-%d.%m.%y')), loss=None, use_aneu_class_balancing = False):
        os.makedirs(wdir)
        self.wdir=wdir
        self._save_train_cfg(model, train_dl, val_dl, epochs, early_stop, wdir, use_aneu_class_balancing)
        if use_aneu_class_balancing:
            ac_weights = torch.tensor(cw.ANEURYSM).to(self.device)
        else:
            ac_weights = None
        model.to(self.prec)
        model.to(self.device)
        self.optim = self.optim(model.parameters(), self.lr)
        self.sched = self.sched(self.optim, epochs)
        self._init_wandb(model, train_dl, val_dl, epochs, early_stop, wdir, use_aneu_class_balancing, loss)
        loss_history = LossHistory(len(train_dl), len(val_dl))
        
        fmt = f"0{len(str(epochs))}d"

        for e in range(1, epochs+1):
            print(f'------------------- Epoch {e:{fmt}}/{epochs} -------------------')
            ## Training step
            model.train()
            with torch.autocast(self.device, self.prec):
                for i, batch in enumerate(tqdm.tqdm(train_dl, desc='Training batches')):
                    self.optim.zero_grad()
                    if loss is None: l = model.loss(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'], batch['location_v'].to(self.device), batch['location_a'].to(self.device), batch['laterality'].to(self.device), weights=ac_weights)
                    else: 
                        lat, loc_v, loc_a = model(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'])
                        l = loss(lat, loc_v, loc_a, batch['location_v'].to(self.device), batch['location_a'].to(self.device), batch['laterality'].to(self.device))
                    assert not torch.isnan(l), f"Loss in training batch {i} for epoch {e} is NaN, terminating. \n The corresponding loaded image batch may contain NaN: {torch.isnan(batch['image']).any()}"
                    l.backward()
                    self.optim.step()
                    loss_history.add_train(l)
                    if self.run is not None: self.run.log({'batch/train_loss': l.item()})
            
            ## Validation step
            model.eval()
            with torch.no_grad(), torch.autocast(self.device, self.prec):
                for i, batch in enumerate(tqdm.tqdm(val_dl, desc='Validating batches')):
                    if loss is None: l = model.loss(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'], batch['location_v'].to(self.device), batch['location_a'].to(self.device), batch['laterality'].to(self.device), weights=ac_weights)
                    else: 
                        lat, loc_v, loc_a = model(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'])
                        l = loss(lat, loc_v, loc_a, batch['location_v'].to(self.device), batch['location_a'].to(self.device), batch['laterality'].to(self.device))
                    assert not torch.isnan(l), f"Loss in validation batch {i} for epoch {e} is NaN, terminating. \n The corresponding loaded image batch may contain NaN: {torch.isnan(batch['image']).any()}"
                    loss_history.add_val(l)
            
            ## scheduling
            lr = self.sched.get_last_lr()[0]
            self.sched.step()
            loss_history.fin_epoch()
            loss_history.plot_progress(wdir)
            if self.run is not None:
                self.run.log({'epoch': e, 'epoch/train_loss': loss_history.train_loss[-1], 'epoch/val_loss': loss_history.val_loss[-1], 'epoch/lr': lr})
            
            ## saving
            model.save(wdir/f'latest_epoch', overwrite=True)
            
            ## saving if best
            if all([l==b for l, b in zip(loss_history.latest(), loss_history.min())]):
                model.save(wdir/'best_val_loss', overwrite=True)
                best_epoch, best_loss = loss_history.min()
                with open(wdir/f'best_val_loss'/'note.txt', 'w') as f:
                    f.write(f'Convergence achieved after {best_epoch} epochs with validation loss {best_loss}')
                if self.run is not None: self.run.summary.update({'best_epoch': best_epoch, 'best_val_loss': best_loss})
                
            
            ## early stopping
            if loss_history.has_converged(early_stop):
                break
        
        else:
            best_epoch, best_loss = loss_history.min()
            print(f'No convergence achieved after {epochs} epochs. Best loss is {best_loss} at epoch {best_epoch}')
            shutil.copytree(wdir/f'latest_epoch', wdir/f'best_val_loss', dirs_exist_ok=True)
            with open(wdir/f'best_val_loss'/'note.txt', 'w') as f:
                f.write(f'No convergence achieved after {epochs} epochs. Best loss is {best_loss} at epoch {best_epoch}')
            if self.run is not None: self.run.summary['converged'] = False
                
        model.load(wdir/f'best_val_loss')
        return model
                
    def test(self, model, test_dl, best_model_dir=None, decoder=DecodeAneu(), target='aneu'): 
        if best_model_dir is not None:
            model.load(best_model_dir)
        model.to(self.device)
        model.eval()
        preds = []
        gts = []
        ids = []
        for batch in tqdm.tqdm(test_dl, desc='Testing batches'):
            ids += batch['id']
            pred_lat, pred_loc = model.classify(batch['image'].to(self.device), batch['coords'].to(self.device), batch['modality'], target=target)
            pred_lat=pred_lat.detach().to('cpu')
            pred_loc=pred_loc.detach().to('cpu')
            preds.append(torch.concat([pred_loc, pred_lat], dim=-1))
            if target == 'vessel': gts.append(torch.concat([batch['location_v'], batch['laterality']], dim=-1))
            else:  gts.append(torch.concat([batch['location_a'], batch['laterality']], dim=-1))
        
        preds = torch.concat(preds, dim=0).to(torch.uint8) 
        preds_dec = decoder(preds)
        gts = torch.concat(gts, dim=0).to(torch.uint8) 
        gts_dec = decoder(gts)
        
        # for id, g, p, g_vec, p_vec in zip(ids, gts_dec, preds_dec, gts, preds):
        #     print(f'Image {id} with GT: loc={g[0]} lat={g[1]} cls={g[2]} got PREDS: loc={p[0]} lat={p[1]} cls={p[2]}')
        #     print(f'    target vector: {g_vec.tolist()}')
        #     print(f'    softmax pred vector: {p_vec.tolist()}')
        
        acc = loc_lat_cls_acc(gts_dec, preds_dec)
        
        print(f"Model achieved an accuracy of {acc}")
        with open(self.wdir/'acc.txt', 'w') as f:
            f.write(f"Model achieved an accuracy of {acc}")
        if self.run is not None: self.run.summary.update({f'test_{target}/{k}': float(v) for k, v in acc.items()})
        return acc
    
    def test_TopAneu(self, model, testds, test_trans, s1, precomp):
        print('----------- With Perfect Segmentations ----------')
        pl = InferencePipeline(None, model, 64, 35, use_tta=True, transforms=test_trans)
        outdir = self.wdir/'eval'/'perfect'
        ev = TopAneu26LikeEvaluator(pl, outdir, precomp_predictions=None)
        res, agg, avg, disc = ev.eval_ds(testds)#ev.eval_list(test.src, test.cases)
        with open(f'{outdir}/classification_failure.json', 'w') as f:
                json.dump(disc, f, indent=4)
        os.makedirs(outdir, exist_ok=True)
        ev.plot(agg, f'{outdir}/per_cls')
        ev.re_eval_by_modality(res, outdir)
        with open(f'{outdir}/per_cls.json', 'w') as f:
            json.dump(avg, f, indent=4)
        if self.run is not None: self.run.summary['topaneu_perfect'] = avg
        
        print('----------- With Stage 1 Segmentations ----------')
        pl = InferencePipeline(s1, model, 64, 35, use_tta=True)
        outdir = self.wdir/'eval'/'with_s1'
        ev = TopAneu26LikeEvaluator(pl, outdir, precomp_predictions=precomp)
        res, agg, avg, disc = ev.eval_ds(testds)#ev.eval_list(test.src, test.cases)
        with open(f'{outdir}/classification_failure.json', 'w') as f:
                json.dump(disc, f, indent=4)
        os.makedirs(outdir, exist_ok=True)
        ev.plot(agg, f'{outdir}/per_cls')
        ev.re_eval_by_modality(res, outdir)
        with open(f'{outdir}/per_cls.json', 'w') as f:
            json.dump(avg, f, indent=4)
        if self.run is not None: self.run.summary['topaneu_with_s1'] = avg