import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib as pl
import datetime
import os

from monai.networks.nets import resnet50, vit#, resnet18

class TnTS2(nn.Module):
    def __init__(self, n_locs_v=21, n_locs_a=29, n_lats=2):
        super().__init__()
        self.n_locs_v, self.n_locs_a, self.n_lats = n_locs_v, n_locs_a, n_lats
        self.bb = resnet50(spatial_dims=3, n_input_channels=3) # outputs [B, 400]
        self.laterality = nn.Linear(404, self.n_lats)
        self.location_vessel = nn.Linear(404, self.n_locs_v)
        self.location_aneu = nn.Linear(404+self.n_locs_v, self.n_locs_a) # in pretraining is the vessel classes
        self.parent = None
        
    def forward(self, patch, coords, modalities):
        unbatched = isinstance(modalities, str)
        if unbatched:
            modalities = [modalities]
            patch = patch.unsqueeze(0)
            coords = coords.unsqueeze(0)

        x = self.bb(patch)
        modality_flag = torch.tensor([m == 'MRA' for m in modalities], dtype=torch.uint8, device=coords.device).unsqueeze(1)
        x = torch.concat([x, coords, modality_flag], dim=-1)
        lat = self.laterality(x)
        loc_v = self.location_vessel(x)
        
        loc_a = self.location_aneu(torch.concat([x, loc_v.detach()], dim=-1))

        if unbatched:
            lat = lat.squeeze(0)
            loc_v = loc_v.squeeze(0)
            loc_a = loc_a.squeeze(0)

        return lat, loc_v, loc_a
    
    def classify(self, patch, coords, modalities, target='aneu'):
        unbatched = isinstance(modalities, str)
        if unbatched: raise RuntimeError("Unbatched data is not supported")
        if target=='vessel': lat, loc, _ = self.forward(patch, coords, modalities)
        else: lat, _, loc = self.forward(patch, coords, modalities)
        lat_sigmoid, loc_sigmoid = F.sigmoid(lat), F.sigmoid(loc)
        
        assigned_lat = torch.zeros_like(lat_sigmoid, dtype=torch.uint8)
        assigned_loc = torch.zeros_like(loc_sigmoid, dtype=torch.uint8)
        for b_item in range(lat_sigmoid.shape[0]):
            loc = torch.argmax(loc_sigmoid[b_item, :]).item()
            lat = torch.argmax(lat_sigmoid[b_item, :]).item()
            assigned_lat[b_item, lat]=1
            assigned_loc[b_item, loc]=1
        
        return assigned_lat, assigned_loc
    
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        if os.path.exists(pth) and not overwrite:
            override = datetime.datetime.now().strftime(r'TnTS2_from-%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        if not os.path.exists(pth): os.makedirs(pth)
        torch.save(self.bb.state_dict(), pth/'bb.pth')
        torch.save(self.location_vessel.state_dict(), pth/'vessel.pth')
        torch.save(self.location_aneu.state_dict(), pth/'aneu.pth')
        torch.save(self.laterality.state_dict(), pth/'laterality.pth')
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.bb.load_state_dict(torch.load(pth/'bb.pth'))
        self.location_vessel.load_state_dict(torch.load(pth/'vessel.pth'))
        self.location_aneu.load_state_dict(torch.load(pth/'aneu.pth'))
        self.laterality.load_state_dict(torch.load(pth/'laterality.pth'))    
        self.parent=pth
    
    @staticmethod
    def from_pretrained(pth):
        pth = pl.Path(pth)
        model = TnTS2()
        model.parent=pth
        model.bb.load_state_dict(torch.load(pth/'bb.pth'))
        model.location_vessel.load_state_dict(torch.load(pth/'location.pth'))
        model.laterality.load_state_dict(torch.load(pth/'laterality.pth'))
        return model
        
    def loss(self, patch, coords, modalities, targets_v, targets_a, targets_lat, weights):
        unbatched = isinstance(modalities, str)
        if unbatched:
            modalities = [modalities]
            patch = patch.unsqueeze(0)
            coords = coords.unsqueeze(0)

        lat, loc_v, loc_a = self.forward(patch, coords, modalities)
        
        loc_a_loss = F.cross_entropy(loc_a, targets_a, weight=weights)
        loc_v_loss = F.cross_entropy(loc_v, targets_v)
        lat_loss = F.cross_entropy(lat, targets_lat)
        return loc_a_loss+loc_v_loss+lat_loss
    
class TnTS2_ViT(nn.Module):
    def __init__(self, n_locs_v=21, n_locs_a=29, n_lats=2, img_size=64, patch_size=8):
        super().__init__()
        self.n_locs_v, self.n_locs_a, self.n_lats = n_locs_v, n_locs_a, n_lats
        self.bb = vit.ViT(in_channels=3, img_size=(img_size,img_size,img_size), patch_size=(patch_size,patch_size,patch_size), dropout_rate=0.1) # outputs [B, 768]
        self.laterality = nn.Linear(772, self.n_lats)
        self.location_vessel = nn.Linear(772, self.n_locs_v)
        self.location_aneu = nn.Linear(772+self.n_locs_v, self.n_locs_a) # in pretraining is the vessel classes
        self.parent = None
        
    def forward(self, patch, coords, modalities):
        unbatched = isinstance(modalities, str)
        if unbatched:
            modalities = [modalities]
            patch = patch.unsqueeze(0)
            coords = coords.unsqueeze(0)

        x, _ = self.bb(patch)
        x = torch.mean(x, dim=1)
        modality_flag = torch.tensor([m == 'MRA' for m in modalities], dtype=torch.uint8, device=coords.device).unsqueeze(1)
        x = torch.concat([x, coords, modality_flag], dim=-1)
        lat = self.laterality(x)
        loc_v = self.location_vessel(x)
        
        loc_a = self.location_aneu(torch.concat([x, loc_v.detach()], dim=-1))

        if unbatched:
            lat = lat.squeeze(0)
            loc_v = loc_v.squeeze(0)
            loc_a = loc_a.squeeze(0)

        return lat, loc_v, loc_a
    
    def classify(self, patch, coords, modalities, target='aneu'):
        unbatched = isinstance(modalities, str)
        if unbatched: raise RuntimeError("Unbatched data is not supported")
        if target=='vessel': lat, loc, _ = self.forward(patch, coords, modalities)
        else: lat, _, loc = self.forward(patch, coords, modalities)
        lat_sigmoid, loc_sigmoid = F.sigmoid(lat), F.sigmoid(loc)
        
        assigned_lat = torch.zeros_like(lat_sigmoid, dtype=torch.uint8)
        assigned_loc = torch.zeros_like(loc_sigmoid, dtype=torch.uint8)
        for b_item in range(lat_sigmoid.shape[0]):
            loc = torch.argmax(loc_sigmoid[b_item, :]).item()
            lat = torch.argmax(lat_sigmoid[b_item, :]).item()
            assigned_lat[b_item, lat]=1
            assigned_loc[b_item, loc]=1
        
        return assigned_lat, assigned_loc
    
    def save(self, pth, overwrite=False):
        pth = pl.Path(pth)
        if os.path.exists(pth) and not overwrite:
            override = datetime.datetime.now().strftime(r'TnTS2_from-%H:%M:%S-%d.%m.%y')
            print(f"INFO: {pth} already exists, using {pth.parent/override} instead")
            pth=pth.parent/override
        if not os.path.exists(pth): os.makedirs(pth)
        torch.save(self.bb.state_dict(), pth/'bb.pth')
        torch.save(self.location_vessel.state_dict(), pth/'vessel.pth')
        torch.save(self.location_aneu.state_dict(), pth/'aneu.pth')
        torch.save(self.laterality.state_dict(), pth/'laterality.pth')
        
    def load(self, pth):
        pth=pl.Path(pth)
        self.bb.load_state_dict(torch.load(pth/'bb.pth'))
        self.location_vessel.load_state_dict(torch.load(pth/'vessel.pth'))
        self.location_aneu.load_state_dict(torch.load(pth/'aneu.pth'))
        self.laterality.load_state_dict(torch.load(pth/'laterality.pth'))    
        self.parent=pth
    
    @staticmethod
    def from_pretrained(pth):
        pth = pl.Path(pth)
        model = TnTS2()
        model.parent=pth
        model.bb.load_state_dict(torch.load(pth/'bb.pth'))
        model.location_vessel.load_state_dict(torch.load(pth/'location.pth'))
        model.laterality.load_state_dict(torch.load(pth/'laterality.pth'))
        return model
        
    def loss(self, patch, coords, modalities, targets_v, targets_a, targets_lat, weights):
        unbatched = isinstance(modalities, str)
        if unbatched:
            modalities = [modalities]
            patch = patch.unsqueeze(0)
            coords = coords.unsqueeze(0)

        lat, loc_v, loc_a = self.forward(patch, coords, modalities)
        
        loc_a_loss = F.cross_entropy(loc_a, targets_a, weight=weights)
        loc_v_loss = F.cross_entropy(loc_v, targets_v)
        lat_loss = F.cross_entropy(lat, targets_lat)
        return loc_a_loss+loc_v_loss+lat_loss

class TnTS2Loss(nn.Module): ## intended for use with the accelerator so here the BCE can be used no problem
    def __init__(self, n_locs_v, n_locs_a):
        super().__init__()
        self.n_locs_v = n_locs_v
        self.n_locs_a = n_locs_a
        
    def forward(self, lat, loc_v, loc_a, targets_v, targets_a, targets_lat):   
        # print(targets_v.shape, targets_v)
        # print(loc_v.shape, loc_v)     
        loc_v_loss = F.cross_entropy(loc_v, targets_v)
        lat_loss = F.cross_entropy(lat, targets_lat)
        loc_a_loss = F.binary_cross_entropy_with_logits(loc_a, targets_a)
        return loc_v_loss+lat_loss+loc_a_loss