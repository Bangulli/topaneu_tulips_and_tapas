from TnT.utils.transforms import get_inference_transforms, DecodeAneu
from TnT.utils.dataloader import TopAneu_TnTs2_DS
from TnT.model.stage2 import TnTS2, TnTS2_ViT
from TnT.model.stage1 import get_s1
from TnT.model.modality_specific import TnTS2_Specific
import SimpleITK as sitk, numpy as np, torch
from scipy.ndimage import label, generate_binary_structure
from pathlib import Path
import torch.nn.functional as F
from pathlib import Path

class InferencePipeline():
    def __init__(self, s1_model_path, s2_model_path, patch_size_vx=64, patch_size_mm=35, device='cuda', use_tta=False, transforms=None, use_mirroring=True):
        
        self.patch_size_mm = patch_size_mm
        self.s2_transforms = transforms
        self.decoder = DecodeAneu()
        self.device = device
        self.tta = use_tta
        
        ## the models
        if s1_model_path is not None: self.s1_model = self._make_s1(s1_model_path, use_mirroring=use_mirroring)
        if s2_model_path is not None: 
            if isinstance(s2_model_path, TnTS2) or isinstance(s2_model_path, TnTS2_Specific) or isinstance(s2_model_path, TnTS2_ViT): 
                self.s2_model = s2_model_path
                self.s2_model.to(self.device)
                self.s2_model.eval()
            else: self.s2_model = self._make_s2(s2_model_path)

    
    @torch.no_grad()  
    def __call__(self, sample: sitk.Image, modality: str, base=None):
        if isinstance(sample, sitk.Image):
            image, vmask, lmask = self._stage1(sample)
            spacing = sample.GetSpacing()
        elif isinstance(sample, dict):
            image=sample['image']
            vmask=sample['vessel_mask']!=0
            lmask=sample['location_mask']!=0
            spacing = sample['spacing']
        elif isinstance(sample, str):
            sample = Path(sample)
            image = sitk.ReadImage(sample)
            msk = sitk.GetArrayFromImage(sitk.ReadImage(base/sample.name.replace('_0000', '')))
            spacing = image.GetSpacing()
            image = sitk.GetArrayFromImage(image)
            vmask = msk==1
            lmask = msk==2
        else: raise RuntimeError(f'unknown input type {type(sample)}')
        
        if np.any(lmask):
            s2ds = CaseDL(image, vmask, lmask, modality, self.s2_transforms, spacing, self.patch_size_mm)
            for i in range(len(s2ds)):
                smp = s2ds[i]
                if self.tta: pred = self._stage2_TTA(smp)
                else: pred = self._stage2(smp)
                s2ds.add_label(i, pred)
            output = s2ds.make_mask()
        else: output = np.zeros_like(vmask)
        
        if isinstance(sample, sitk.Image): 
            outimg = sitk.GetImageFromArray(output.astype(np.uint8))
            outimg.CopyInformation(sample)
            return outimg
        else: return output.astype(np.uint8)
        
    def _stage1(self, image: sitk.Image):
        # nnU-Net expects numpy arrays shaped (channels, z, y, x)
        img_array = sitk.GetArrayFromImage(image).astype(np.float32)  # (z, y, x)
        img_array = img_array[np.newaxis, ...]  # -> (1, z, y, x) H, D, W
    
        # sitk spacing is (x, y, z); nnU-Net wants (z, y, x)W, D, H
        nnunet_spacing = list(image.GetSpacing())[::-1]
        props = {"spacing": nnunet_spacing}
    
        segmentation = self.s1_model.predict_single_npy_array(
            img_array, props, None, None, False
        ).astype(np.uint8)
        
        vmask_arr = segmentation==1
        lmask_arr = segmentation==2
        
        spacing = image.GetSpacing()
        mm3_per_voxel = spacing[0]*spacing[1]*spacing[2]
        cc, n = label(lmask_arr, structure=generate_binary_structure(3, 3))
        for i in range(1, n+1):
            object_size = np.sum(cc==i)*mm3_per_voxel
            if object_size < 2:
                lmask_arr[cc==i]=0 
        
        return img_array.squeeze(0), vmask_arr, lmask_arr
        
    
    def _stage2(self, smp: dict) -> int:
        pred_lat, pred_loc = self.s2_model.classify(smp['image'].unsqueeze(0).to(self.device), smp['coords'].unsqueeze(0).to(self.device), [smp['modality']])
        pred_lat=pred_lat.detach().to('cpu')
        pred_loc=pred_loc.detach().to('cpu')
        preds = torch.concat([pred_loc, pred_lat], dim=-1)
        assert len(preds.shape)==2, 'only implemented for batched data'
        decoded_label = self.decoder(preds)[0][2]
        return decoded_label
    
    def _stage2_TTA(self, smp:dict) -> int:
        ttas = [ ## for more ttas the model should be trained with more flip augmentations
            (False, False, False), 
            (False, False, True),
        ]
        
        loc_probas = []
        lat_probas = []
        for tta in ttas: 
            img = smp['image']
            crd = smp['coords']
            
            flip_lat = False
            for dim, trig in enumerate(tta):## lat dim is 2 in unchanneled data
                if trig:
                    img = torch.flip(img, [dim+1]) # +1 to offset channel dim
                    crd[dim] = 1 - crd[dim]
                    if dim == 2: flip_lat = True
                
            pred_lat, _, pred_loc = self.s2_model.forward(img.unsqueeze(0).to(self.device), crd.unsqueeze(0).to(self.device), [smp['modality']])
            pred_lat=pred_lat.detach().to('cpu')
            pred_loc=pred_loc.detach().to('cpu')
            if flip_lat: 
                tmp = pred_lat[0, -2].clone()
                pred_lat[0, -2] = pred_lat[0, -1]
                pred_lat[0, -1] = tmp
            loc_probas.append(F.sigmoid(pred_loc))
            lat_probas.append(F.sigmoid(pred_lat))
            
        loc_probas = torch.mean(torch.concat(loc_probas, dim=0), dim=0).unsqueeze(0)
        assert len(loc_probas.shape)==2
        lat_probas = torch.mean(torch.concat(lat_probas, dim=0), dim=0).unsqueeze(0)
        assert len(lat_probas.shape)==2
            
        assigned_lat = torch.zeros_like(lat_probas, dtype=torch.uint8)
        assigned_loc = torch.zeros_like(loc_probas, dtype=torch.uint8)
        for b_item in range(lat_probas.shape[0]):
            loc = torch.argmax(loc_probas[b_item, :]).item()
            lat = torch.argmax(lat_probas[b_item, :]).item()
            assigned_lat[b_item, lat]=1
            assigned_loc[b_item, loc]=1
        
        preds = torch.concat([assigned_loc, assigned_lat], dim=-1)
        assert len(preds.shape)==2, 'only implemented for batched data'
        decoded_label = self.decoder(preds)[0][2]
        return decoded_label
            
    
    def _make_s1(self, path, verbose=True, use_mirroring=True):
        return get_s1(path, self.device, verbose=verbose, use_mirroring=use_mirroring)
    
    def _make_s2(self, path):
        if isinstance(path, dict):
            predictor = TnTS2_Specific(path['mr'], path['ct'])
            predictor.to(self.device)
            predictor.eval()
        elif isinstance(path, str) or isinstance(path, Path):
            try:
                predictor = TnTS2()
                predictor.load(path)
                predictor.to(self.device)
                predictor.eval()
            except:
                predictor = TnTS2_ViT()
                predictor.load(path)
                predictor.to(self.device)
                predictor.eval()
        else:
            raise ValueError(f'Cannot build S2 model from input type {type(path)}, needs to be path, str or dict of paths/strs for modality speficif modeling')
        return predictor
class CaseDL(TopAneu_TnTs2_DS):
    def __init__(self, image, vmask, lmask, modality, transforms, spacing, patch_size_mm):
        self.image = image
        self.vmask = vmask
        self.lmask = lmask
        self.modality = 'CTA' if 'ct' in modality.lower() else 'MRA'
        self.transforms = transforms
        self.cc, self.n = label(self.lmask, structure=generate_binary_structure(3, 3))
        self.spacing = spacing
        self.patch_size_mm = patch_size_mm
        vbb_coords = np.argwhere(self.vmask).T # VBB = Vessel Bounding Box
        vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
        vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
        vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
        self.vbb = [vbb_d, vbb_h, vbb_w]
        self.vbb_shape = [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])]
        
        self.assigned_labels = {}
        
    def __getitem__(self, idx):     
        coords = np.mean(np.argwhere(self.cc==idx+1), axis=0).tolist()   
        multichannel_img = np.stack( # 4D array: [C, H, D, W]
            [
                self._center_crop(self.image, coords, self.spacing, self.patch_size_mm),
                self._center_crop(self.lmask, coords, self.spacing, self.patch_size_mm),
                self._center_crop(self.vmask, coords, self.spacing, self.patch_size_mm)
            ], axis=0
        )
        coords_in_vbb = [
            coords[0]-self.vbb[0][0],
            coords[1]-self.vbb[1][0],
            coords[2]-self.vbb[2][0]
        ]
        
        dct = {
            'image': multichannel_img,
            'coords': np.array(coords_in_vbb, dtype=int)/np.array(self.vbb_shape, dtype=int), # relative
            'modality': self.modality, # string
            'spacing': self.spacing
        }
        
        if self.transforms:
            dct = self.transforms(dct)
        
        return dct

    def __len__(self):
        return self.n

    def add_label(self, idx, lbl):
        assert isinstance(lbl, int)
        self.assigned_labels[idx]=lbl
        assert len(self.assigned_labels)<=len(self)
        
    def make_mask(self):
        output = np.zeros_like(self.cc, dtype=np.uint8)
        for idx, lbl in self.assigned_labels.items():
            output[self.cc==idx+1]=lbl
        return output