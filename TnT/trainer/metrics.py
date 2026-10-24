from sklearn.metrics import accuracy_score
import numpy as np  
import matplotlib.pyplot as plt 
import json

def loc_lat_cls_acc(gts, preds):
    # print('----- EVALUATING -----')
    # for gt, pred in zip(gts, preds):
    #     print(f'    > GT: loc:{gt[0]} - lat:{gt[1]} - lbl:{gt[2]}')
    #     print(f'    > PD: loc:{pred[0]} - lat:{pred[1]} - lbl:{pred[2]}')
    
    loc_acc = sum([gt[0]==pred[0] for gt, pred in zip(gts, preds)])/len(gts)
    lat_acc = sum([gt[1]==pred[1] for gt, pred in zip(gts, preds)])/len(gts)
    cls_acc = sum([gt[2]==pred[2] for gt, pred in zip(gts, preds)])/len(gts)
    return {'location acc': loc_acc, 'laterality acc': lat_acc, 'topaneu cls acc': cls_acc}

class LossHistory():
    def __init__(self, nsteps_tr, nsteps_vl):
        self.train_loss = []
        self.val_loss = []
        self.cur_val = 0
        self.cur_train = 0
        self.nval = nsteps_vl
        self.ntra = nsteps_tr
    
    def add_train(self, l):
        self.cur_train += (l.item()/self.ntra)
        
    def add_val(self, l):
        self.cur_val += (l.item()/self.nval)
        
    def has_converged(self, patience, tol=1e-4):
        if patience is None: return False
        if len(self.val_loss)<patience: return False
        within_patience = self.val_loss[-patience:]
        return not min(within_patience)<within_patience[0]-tol
    
    def min(self):
        e, v = min(enumerate(self.val_loss), key=lambda x: x[1])
        return e+1, v
    
    def latest(self):
        e, v = len(self.val_loss), self.val_loss[-1]
        return e, v
    
    def fin_epoch(self):
        self.train_loss.append(self.cur_train)
        self.cur_train = 0
        self.val_loss.append(self.cur_val)
        self.cur_val = 0
        
    def plot_progress(self, wdir):
        plt.plot(self.train_loss, label='training loss')
        plt.plot(self.val_loss, label='validation loss')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.legend()
        plt.title(f'Training/Validation loss per epoch')
        plt.savefig(wdir/'training_losses.png')
        plt.close()
        plt.clf()
        
        with open(wdir/'losses.json', 'w') as f:
            json.dump(
                {
                    'train': self.train_loss,
                    'val': self.val_loss
                },
                f, 
                indent=4
            )
