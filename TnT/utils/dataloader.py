from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import os, tqdm, SimpleITK as sitk, json, numpy as np, random, torch, copy
from scipy.ndimage import label, binary_erosion, binary_dilation, generate_binary_structure
from pprint import pprint
from TnT.model.stage1 import get_s1

class TopAneuDS(Dataset):
    def __init__(self, source, transforms=None, load_type_mask=False, cases=None):
        self.src = Path(source)
        self._check_complete(cases)
        self.transforms = transforms
        self.load_tm = load_type_mask
        
    def _check_complete(self, cases=None):
        assert os.path.exists(self.src/'images') and os.path.exists(self.src/'location_jsons') and os.path.exists(self.src/'location_masks') and os.path.exists(self.src/'type_masks') and os.path.exists(self.src/'vessel_masks'), 'Could not find all required diretories'
        if cases is None: self.cases = sorted(os.listdir(self.src/'images')) # sorted for consistency
        else: self.cases = cases
        for img in self.cases:
            fn = img.replace('_0000.', '.')
            fn_json = fn.replace('.nii.gz', '.json')
            assert os.path.exists(self.src/'images'/img) and os.path.exists(self.src/'location_jsons'/fn_json) and os.path.exists(self.src/'location_masks'/fn) and os.path.exists(self.src/'type_masks'/fn) and os.path.exists(self.src/'vessel_masks'/fn), f'Could not find all required images for case {fn} -> Image: {os.path.exists(self.src/'images'/img)}, Json: {os.path.exists(self.src/'location_jsons'/fn_json)}, Location Mask: {os.path.exists(self.src/'location_masks'/fn)}, Type Mask: {os.path.exists(self.src/'type_masks'/fn)}, Vessel Mask: {os.path.exists(self.src/'vessel_masks'/fn)}'
        
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        img = self.cases[idx]
        fn = img.replace('_0000.', '.')
        fn_json = fn.replace('.nii.gz', '.json')
        
        sitk_im = sitk.ReadImage(self.src/'images'/img)
        vol = sitk.GetArrayFromImage(sitk_im)
        l_msk = sitk.GetArrayFromImage(sitk.ReadImage(self.src/'location_masks'/fn))
        v_msk = sitk.GetArrayFromImage(sitk.ReadImage(self.src/'vessel_masks'/fn))
        if self.load_tm: t_msk = sitk.GetArrayFromImage(sitk.ReadImage(self.src/'type_masks'/fn))
        with open(self.src/'location_jsons'/fn_json, 'r') as f:
            l_json = json.load(f)
            
        dct = {
            'image': vol,
            'location_mask': l_msk,
            'type_mask': t_msk if self.load_tm else None,
            'vessel_mask': v_msk,
            'location': l_json['locations'],
            'modality': 'MRA' if '_mr_' in fn else 'CTA',
            'id': fn.replace('.nii.gz', ''),
            'spacing': sitk_im.GetSpacing()
        }
        
        if self.transforms:
            dct = self.transforms(dct)
            
        return dct
    
    @staticmethod
    def load(path, transforms=None):
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'r') as file:
            loaded = json.load(file)
        ds = TopAneuDS(source=loaded['source'], transforms=transforms, cases=loaded['cases'])
        return ds
    
class TopAneu_TnTs2_DS(Dataset):
    ########################### builtins
    def __init__(self, source, transforms=None, cases=None, patch_size_mm=35, patch_size_vx=64 ,wdir=None, s1_path=None):
        self.image_ds = TopAneuDS(source, transforms=None, load_type_mask=False, cases=cases)
        self.aneus = []
        self.transforms = transforms
        self.patch_size_mm = patch_size_mm
        self.patch_size_vx = patch_size_vx
        self.wdir = wdir
        self.s1_predictor = get_s1(s1_path) if s1_path is not None else None
        
    def __len__(self):
        return len(self.aneus)
    
    def __getitem__(self, idx):
        if self.wdir is not None: 
            self.wdir = Path(self.wdir)
            os.makedirs(self.wdir, exist_ok=True)
        if not self.is_patched: self.preprocess()
        smp = self.aneus[idx]
        multichannel_img = None
        
        if self.wdir is not None:
            if os.path.exists(self.wdir/f"{idx}.npy"):
                multichannel_img = np.load(self.wdir/f"{idx}.npy")
                with open(self.wdir/f"{idx}.json", 'r') as f:
                    img_smp = json.load(f)
        
        if multichannel_img is None:
            img_smp = self.image_ds[smp['idx']]
            
            
            multichannel_img = np.stack( # 4D array: [C, H, D, W]
                [
                    self._center_crop(img_smp['image'], smp['coords'], img_smp['spacing'], self.patch_size_mm),
                    self._center_crop(img_smp['location_mask'], smp['coords'], img_smp['spacing'], self.patch_size_mm),
                    self._center_crop(img_smp['vessel_mask'], smp['coords'], img_smp['spacing'], self.patch_size_mm)
                ], axis=0
            )
            
            # get the associated vloc
            if np.any(multichannel_img[2]!=0):
                msk = np.bitwise_and(binary_dilation(multichannel_img[2]!=0), binary_dilation(multichannel_img[1]!=0))
                if np.any(msk): vessel_id = np.median(multichannel_img[2][msk].astype(np.uint8))
                else: vessel_id = np.median(multichannel_img[2][multichannel_img[2]!=0].astype(np.uint8))
            else: vessel_id = 0
            img_smp['vloc']= int(vessel_id)
            
            # make random sphere for synthetic patches
            if smp['make_syn_aneu_msk']: 
                multichannel_img = self._put_random_sphere_as_aneu(multichannel_img, img_smp["spacing"])
                
            # use s1 predictor to make s1 realistic masks
            if smp['trigger_s1']:
                nnunet_spacing = img_smp['spacing'][::-1]
                props = {"spacing": nnunet_spacing}
                pred = self.s1_predictor.predict_single_npy_array(multichannel_img[0][None], props, None, None, False).squeeze().astype(np.uint8)
                multichannel_img[2]=pred==1
                multichannel_img[1]=pred==2
                
            if self.wdir is not None:
                np.save(self.wdir/f"{idx}.npy", multichannel_img)
                with open(self.wdir/f"{idx}.json", 'w') as f:
                    json.dump({'id': img_smp['id'], 'spacing': img_smp['spacing'], 'vloc': vessel_id}, f, indent=4)
        
        coords_in_vbb = [
            smp['coords'][0]-smp['vbb'][0][0],
            smp['coords'][1]-smp['vbb'][1][0],
            smp['coords'][2]-smp['vbb'][2][0]
        ]
        
        dct = {
            'image': multichannel_img,
            'location_v': [img_smp['vloc']],
            'location_a': smp['location'], # multihot
            'coords': np.array(coords_in_vbb, dtype=int)/np.array(smp['vbb.shape'], dtype=int), # relative
            'modality': smp['modality'], # string
            'id': img_smp['id'],
            'spacing': img_smp['spacing']
        }
        
        if self.transforms:
            dct = self.transforms(dct)
        
        return dct
    ###########################
    ########################### properties
    @property
    def is_patched(self):
        return len(self.aneus)>0
    ########################### 
    ########################### privates
    def _center_crop(self, img, coords, spacing, size_mm):
            sz_mm = round(size_mm/2)
            sz = [round(sz_mm/sp) for sp in spacing]
            mx = max(sz)
            shape = img.shape
            
            coords_lower = []
            coords_upper = []
            for c_ax, sh_ax, sz_ax in zip(coords, shape, sz):
    
                upper = c_ax+sz_ax
                lower = c_ax-sz_ax
                
                coords_lower.append(lower+mx)
                coords_upper.append(upper+mx)
            
            img = np.pad(img, mx, 'reflect')
            
            crop = img[
                int(coords_lower[0]):int(coords_upper[0]),
                int(coords_lower[1]):int(coords_upper[1]),
                int(coords_lower[2]):int(coords_upper[2])
            ]  
            return crop
    
    def _center_mask_out(self, img, coords, spacing, size_mm):
        sz_mm = round(size_mm/2)
        sz = [round(sz_mm/sp) for sp in spacing]
        shape = img.shape
        
        coords_lower = []
        coords_upper = []
        for c_ax, sh_ax, sz_ax in zip(coords, shape, sz):

            upper = c_ax+sz_ax
            lower = c_ax-sz_ax
            
            if upper<=sh_ax and lower >=0:
                coords_lower.append(lower)
                coords_upper.append(upper)
            elif upper > sh_ax:
                diff = upper-sh_ax
                coords_upper.append(sh_ax)
                coords_lower.append(lower-diff)
            elif lower < 0:
                diff = lower
                coords_lower.append(0)
                coords_upper.append(upper+abs(diff))
        
        img[
            int(coords_lower[0]):int(coords_upper[0]),
            int(coords_lower[1]):int(coords_upper[1]),
            int(coords_lower[2]):int(coords_upper[2])
        ] = 0      
        return img
    
    def _make_bg_patches(self, n):
        bg_patches = []
        for i in tqdm.tqdm(range(n), desc='Generating random bg patches'):
            rndm_img_idx = random.choice(range(len(self)))
            cur_smp = self.aneus[rndm_img_idx]
            img_smp = self.image_ds[cur_smp['idx']]
            masked = self._center_mask_out(copy.deepcopy(img_smp['vessel_mask']), cur_smp['coords'], img_smp['spacing'], self.patch_size_mm)
            
            possible_seeds = np.argwhere(masked)
            
            seed = possible_seeds[random.choice(range(possible_seeds.shape[0])), :]
            
            vbb_coords = np.argwhere(img_smp['vessel_mask']).T # VBB = Vessel Bounding Box
            vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
            vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
            vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
            
            smp = {
                    'idx': i, # the base image idx in the base dataset
                    'coords': seed.tolist(), # the centroid
                    'location': 0,
                    'modality': img_smp['modality'],
                    'vbb': [vbb_d, vbb_h, vbb_w],
                    'vbb.shape': [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])],
                    'id': img_smp['id'],
                    'trigger_s1': False,
                    'make_syn_aneu_msk':False,
                }
            
            bg_patches.append(smp)
        return bg_patches
            
    def _put_random_sphere_as_aneu(self, img, spacing):
        diameter_in_mm = random.choice(np.arange(2, 10.5, 0.25)) # generates a random aneurysm diameter in mms
        diameter = max(spacing)*diameter_in_mm # converts into voxels bit lossy though since the spacing is non isometric, yielding basically an odd shape
        diameter = min(min(img.shape if len(img.shape)<4 else img.shape[1:]), round(diameter))
        diameter = max(diameter, 1)
        
        obj = np.zeros((diameter, diameter, diameter), dtype=bool)
        radius = diameter / 2
        center = (diameter - 1) / 2  # e.g. for diameter=3: center=1.0

        z, y, x = np.ogrid[:diameter, :diameter, :diameter]
        dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2

        obj[dist_sq <= radius**2] = True
        obj = obj.astype(np.uint8)
        size_sph = obj.shape
        center = [s//2 for s in img.shape[1:]]
        lowers = [max(0, min(c - s // 2, img.shape[1:][i] - s))
              for i, (c, s) in enumerate(zip(center, size_sph))]

        img[1,
            lowers[0]:lowers[0]+size_sph[0],
            lowers[1]:lowers[1]+size_sph[1],
            lowers[2]:lowers[2]+size_sph[2]] = obj
        return img
    
    ###########################
    ########################### publics
    def preprocess(self, include_bg=False, include_s1_pred=False, syn_samples=False, max_items=-1):
        if self.is_patched: return
        self.aneus = []
        for i in tqdm.tqdm(range(len(self.image_ds)), desc='Patching'):
            sample = self.image_ds[i]
            if not any(sample['location']): continue
            
            cc, n = label(sample['location_mask'], generate_binary_structure(3,3))
            
            vbb_coords = np.argwhere(sample['vessel_mask']).T # VBB = Vessel Bounding Box
            vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
            vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
            vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
            
            for obj in range(1, n+1):
                smp = {
                    'idx': i, # the base image idx in the base dataset
                    'coords': np.mean(np.argwhere(cc==obj), axis=0).tolist(), # the centroid
                    'location': int(np.median(sample['location_mask'][cc==obj])),
                    'modality': sample['modality'],
                    'vbb': [vbb_d, vbb_h, vbb_w],
                    'vbb.shape': [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])],
                    'id': sample['id'],
                    'trigger_s1': False,
                    'make_syn_aneu_msk':False
                }
                self.aneus.append(smp)
            if i == max_items:break
        
        n_real_patches = len(self.aneus) # store number of actual patches.
        
        extra_patches = []
        if include_bg:
            n_bg = round(n_real_patches*include_bg)
            extra_patches += self._make_bg_patches(n_bg)
        if syn_samples:
            for smp in tqdm.tqdm(syn_samples, desc='Generating Synthetic Samples'):
                match_idx = [i for i, c in enumerate(self.image_ds.cases) if c==(smp["id"]+"_0000.nii.gz")]
                if len(match_idx)!=1: print(f"could not find {smp["id"]} in this fold"); continue
                sample = self.image_ds[match_idx[0]]
                vbb_coords = np.argwhere(sample['vessel_mask']).T # VBB = Vessel Bounding Box
                vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
                vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
                vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
                a = {
                    'idx': match_idx[0], # the base image idx in the base dataset
                    'coords': smp['coords'], # the centroid
                    'location': smp['location'],
                    'modality': sample['modality'],
                    'vbb': [vbb_d, vbb_h, vbb_w],
                    'vbb.shape': [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])],
                    'id': smp['id'],
                    'trigger_s1': False,
                    'make_syn_aneu_msk': True,
                }
                extra_patches.append(a)
        if include_s1_pred:
            for a in self.aneus:
                cpy = copy.deepcopy(a)
                cpy['trigger_s1']=True
                extra_patches.append(cpy)
            
        self.aneus+=extra_patches
        
    def split(self, folds='0.8-0.2', random_seed=42):
        folds = [float(val) for val in folds.split('-')]
        assert sum(folds)==1, 'Sum of folds must be equal to 1'
        random.seed(random_seed)
        rndm_cases = self.image_ds.cases.copy()
        random.shuffle(rndm_cases)
        n_cases = len(self.image_ds)
        fold_ds = []
        subsets = []
        lower = 0
        for fold in folds:
            upper = round(n_cases*fold)+lower
            upper = upper if upper <= n_cases else n_cases
            subset = rndm_cases[lower:upper]
            fold_ds.append(TopAneu_TnTs2_DS(self.image_ds.src, transforms=None, cases=subset, patch_size_mm=self.patch_size_mm))
            subsets.append(subset)
            lower=upper
        
        for i, sub in enumerate(subsets):
            other = []
            for j, ss in enumerate(subsets):
                if j == i: continue
                other += ss
            assert not any([s in other for s in sub]), 'Found leakage between sets.'
            
        if self.is_patched:
            for fold in fold_ds:
                for aneu in self.aneus:
                    if aneu['id']+'_0000.nii.gz' in fold.image_ds.cases:
                        aneu['idx'] = [i for i, c in enumerate(fold.image_ds.cases) if c == aneu['id']+'_0000.nii.gz'][0]
                        fold.aneus.append(aneu)
        
        return fold_ds
    
    def separate_by_modality(self):
        #source, transforms=None, cases=None, patch_size_mm=50, wdir=None
        ct = TopAneu_TnTs2_DS(self.image_ds.src, self.transforms, self.image_ds.cases, self.patch_size_mm, str(self.wdir)+'_ct' if self.wdir is not None else None)
        
        mr = TopAneu_TnTs2_DS(self.image_ds.src, self.transforms, self.image_ds.cases, self.patch_size_mm, str(self.wdir)+'_mr' if self.wdir is not None else None)
        
        for a in self.aneus:
            if a['modality']=='MRA': mr.aneus.append(a)
            else: ct.aneus.append(a)
            
        return ct, mr
    
    def save(self, path):
        saveable = {
            'cases': self.image_ds.cases,
            'source': str(self.image_ds.src),
            'aneus': self.aneus,
            'patch_size_mm': self.patch_size_mm,
            'patch_size_vx': self.patch_size_vx
        }
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'w') as file:
            json.dump(saveable, file, indent=4)
    
    def append(self, to_add):
        if to_add.image_ds.src != self.image_ds.src: raise RuntimeError('Sources dont match between datasets, cant append')
        self.aneus += to_add.aneus
        self.image_ds.cases += to_add.image_ds.cases
        
    @staticmethod
    def load(path, transforms=None):
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'r') as file:
            loaded = json.load(file)
        ds = TopAneu_TnTs2_DS(source=loaded['source'], transforms=transforms, cases=loaded['cases'])
        ds.aneus = loaded['aneus'] if loaded['aneus'] is not None else []
        ds.patch_size_mm = loaded['patch_size_mm']
        ds.patch_size_vx = loaded['patch_size_vx']
        return ds
    
    @staticmethod
    def join(folds):
        if len(folds) == 1: return folds[0]
        new_ds = folds[0]
        for i in range(1, len(folds)):
            new_ds.append(folds[i])
        return new_ds
    ###########################
    
class TopAneu_TnTs2_DS_for_vessel_pt(Dataset):
    ########################### builtins
    def __init__(self, source, transforms=None, cases=None, patch_size_mm=35, patch_size_vx=64, wdir=None):
        self.image_ds = TopAneuDS(source, transforms=None, load_type_mask=False, cases=cases)
        self.aneus = []
        self.transforms = transforms
        self.patch_size_mm = patch_size_mm
        self.patch_size_vx = patch_size_vx
        self.wdir = wdir
        
    def __len__(self):
        return len(self.aneus)
    
    def __getitem__(self, idx):
        if self.wdir is not None: 
            self.wdir = Path(self.wdir)
            os.makedirs(self.wdir, exist_ok=True)
        if not self.is_patched: self.preprocess()
        smp = self.aneus[idx]
        
        if os.path.exists(self.wdir/f"{idx}.npy") and self.wdir is not None:
            multichannel_img = np.load(self.wdir/f"{idx}.npy")
            with open(self.wdir/f"{idx}.json", 'r') as f:
                img_smp = json.load(f)
        
        else:
            img_smp = self.image_ds[smp['idx']]
            
            multichannel_img = np.stack( # 4D array: [C, H, D, W]
                [
                    self._center_crop(img_smp['image'], smp['coords'], img_smp['spacing'], self.patch_size_mm),
                    self._center_crop(img_smp['location_mask'], smp['coords'], img_smp['spacing'], self.patch_size_mm),
                    self._center_crop(img_smp['vessel_mask'], smp['coords'], img_smp['spacing'], self.patch_size_mm)
                ], axis=0
            )
            
            multichannel_img = self._put_random_sphere_as_aneu(multichannel_img, img_smp["spacing"])
            
            if self.wdir is not None:
                np.save(self.wdir/f"{idx}.npy", multichannel_img)
                with open(self.wdir/f"{idx}.json", 'w') as f:
                    json.dump({'id': img_smp['id'], 'spacing': img_smp['spacing']}, f, indent=4)
        
        coords_in_vbb = [
            smp['coords'][0]-smp['vbb'][0][0],
            smp['coords'][1]-smp['vbb'][1][0],
            smp['coords'][2]-smp['vbb'][2][0]
        ]
        
        dct = {
            'image': multichannel_img,
            'location_v': [smp['location']], # multihot
            'location_a': None,
            'coords': np.array(coords_in_vbb, dtype=int)/np.array(smp['vbb.shape'], dtype=int), # relative
            'modality': smp['modality'], # string
            'id': img_smp['id'],
            'spacing': img_smp['spacing']
        }
        
        if self.transforms:
            dct = self.transforms(dct)
        
        return dct
    ###########################
    ########################### properties
    @property
    def is_patched(self):
        return len(self.aneus)>0
    ########################### 
    ########################### privates
    def _center_crop(self, img, coords, spacing, size_mm):
        sz_mm = round(size_mm/2)
        sz = [round(sz_mm/sp) for sp in spacing]
        mx = max(sz)
        shape = img.shape
        
        coords_lower = []
        coords_upper = []
        for c_ax, sh_ax, sz_ax in zip(coords, shape, sz):

            upper = c_ax+sz_ax
            lower = c_ax-sz_ax
            
            coords_lower.append(lower+mx)
            coords_upper.append(upper+mx)
        
        img = np.pad(img, mx, 'reflect')
        
        crop = img[
            int(coords_lower[0]):int(coords_upper[0]),
            int(coords_lower[1]):int(coords_upper[1]),
            int(coords_lower[2]):int(coords_upper[2])
        ]  
        return crop
    
    def _center_mask_out(self, img, coords, spacing, size_mm):
        sz_mm = round(size_mm/2)
        sz = [round(sz_mm/sp) for sp in spacing]
        shape = img.shape
        
        coords_lower = []
        coords_upper = []
        for c_ax, sh_ax, sz_ax in zip(coords, shape, sz):

            upper = c_ax+sz_ax
            lower = c_ax-sz_ax
            
            if upper<=sh_ax and lower >=0:
                coords_lower.append(lower)
                coords_upper.append(upper)
            elif upper > sh_ax:
                diff = upper-sh_ax
                coords_upper.append(sh_ax)
                coords_lower.append(lower-diff)
            elif lower < 0:
                diff = lower
                coords_lower.append(0)
                coords_upper.append(upper+abs(diff))
        
        img[
            int(coords_lower[0]):int(coords_upper[0]),
            int(coords_lower[1]):int(coords_upper[1]),
            int(coords_lower[2]):int(coords_upper[2])
        ] = 0      
        return img
            
    def _put_random_sphere_as_aneu(self, img, spacing):
        diameter_in_mm = random.choice(np.arange(2, 10.5, 0.25)) # generates a random aneurysm diameter in mms
        diameter = max(spacing)*diameter_in_mm # converts into voxels bit lossy though since the spacing is non isometric, yielding basically an odd shape
        diameter = min(min(img.shape if len(img.shape)<4 else img.shape[1:]), round(diameter))
        diameter = max(diameter, 1)
        
        obj = np.zeros((diameter, diameter, diameter), dtype=bool)
        radius = diameter / 2
        center = (diameter - 1) / 2  # e.g. for diameter=3: center=1.0

        z, y, x = np.ogrid[:diameter, :diameter, :diameter]
        dist_sq = (x - center)**2 + (y - center)**2 + (z - center)**2

        obj[dist_sq <= radius**2] = True
        obj = obj.astype(np.uint8)
        size_sph = obj.shape
        center = [s//2 for s in img.shape[1:]]
        lowers = [max(0, min(c - s // 2, img.shape[1:][i] - s))
              for i, (c, s) in enumerate(zip(center, size_sph))]

        img[1,
            lowers[0]:lowers[0]+size_sph[0],
            lowers[1]:lowers[1]+size_sph[1],
            lowers[2]:lowers[2]+size_sph[2]] = obj
        return img
    
    ###########################
    ########################### publics
    def preprocess(self, patches_per_vloc=42, max_items=-1):
        if self.is_patched: return
        for i in tqdm.tqdm(range(len(self.image_ds)), desc='Making Patches from Vessels'):
            sample = self.image_ds[i]
            
            vessel_mask = sample['vessel_mask']
            
            vbb_coords = np.argwhere(vessel_mask).T # VBB = Vessel Bounding Box
            vbb_d = [int(np.min(vbb_coords[0])), int(np.max(vbb_coords[0]))]
            vbb_h = [int(np.min(vbb_coords[1])), int(np.max(vbb_coords[1]))]
            vbb_w = [int(np.min(vbb_coords[2])), int(np.max(vbb_coords[2]))]
            
            vessel_cls = np.unique(vessel_mask).tolist()
            
            for vc in vessel_cls:
                if vc==0:continue # skip bg
                binarized = vessel_mask==vc
                surface = binarized & ~binary_erosion(binarized)
                for obj in range(patches_per_vloc):
                    possible_seeds = np.argwhere(surface)
                    seed = possible_seeds[random.choice(range(possible_seeds.shape[0])), :]
                    smp = {
                        'idx': i, # the base image idx in the base dataset
                        'coords': seed.tolist(), # the centroid
                        'location': int(vc),
                        'modality': sample['modality'],
                        'vbb': [vbb_d, vbb_h, vbb_w],
                        'vbb.shape': [int(vbb_d[1]-vbb_d[0]), int(vbb_h[1]-vbb_h[0]), int(vbb_w[1]-vbb_w[0])],
                        'id': sample['id']
                    }
                    self.aneus.append(smp)
                
            if i == max_items:break
        
    def split(self, folds='0.8-0.2', random_seed=42):
        folds = [float(val) for val in folds.split('-')]
        assert sum(folds)==1, 'Sum of folds must be equal to 1'
        random.seed(random_seed)
        rndm_cases = self.image_ds.cases.copy()
        random.shuffle(rndm_cases)
        n_cases = len(self.image_ds)
        fold_ds = []
        subsets = []
        lower = 0
        for fold in folds:
            upper = round(n_cases*fold)+lower
            upper = upper if upper <= n_cases else n_cases
            subset = rndm_cases[lower:upper]
            fold_ds.append(TopAneu_TnTs2_DS_for_vessel_pt(self.image_ds.src, transforms=None, cases=subset, patch_size_mm=self.patch_size_mm))
            subsets.append(subset)
            lower=upper
        
        for i, sub in enumerate(subsets):
            other = []
            for j, ss in enumerate(subsets):
                if j == i: continue
                other += ss
            assert not any([s in other for s in sub]), 'Found leakage between sets.'
            
        if self.is_patched:
            for fold in fold_ds:
                for aneu in self.aneus:
                    if aneu['id']+'_0000.nii.gz' in fold.image_ds.cases:
                        fold.aneus.append(aneu)
        
        return fold_ds
    
    def save(self, path):
        saveable = {
            'cases': self.image_ds.cases,
            'source': str(self.image_ds.src),
            'aneus': self.aneus,
            'patch_size_mm': self.patch_size_mm,
            'patch_size_vx': self.patch_size_vx
        }
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'w') as file:
            json.dump(saveable, file, indent=4)
        
    @staticmethod
    def load(path, transforms=None):
        path = str(path)+'.json' if not str(path).endswith('.json') else str(path)
        with open(path, 'r') as file:
            loaded = json.load(file)
        ds = TopAneu_TnTs2_DS_for_vessel_pt(source=loaded['source'], transforms=transforms, cases=loaded['cases'])
        ds.aneus = loaded['aneus'] if loaded['aneus'] is not None else []
        ds.patch_size_mm = loaded['patch_size_mm']
        ds.patch_size_vx = loaded['patch_size_vx']
        return ds
    ###########################
    
def TnTs2_collate(batch):
    # Assumes all data is already a tensor, if not will attempt to cast to tensor
    images = []
    coords = []
    locations_a = []
    locations_v = []
    lats = []
    modalities = []
    ids = []
    vlocs = []
    for sample in batch:
        images.append(sample['image'])
        coords.append(sample['coords'])
        locations_a.append(sample['location_a'])
        locations_v.append(sample['location_v'])
        lats.append(sample['laterality'])
        modalities.append(sample['modality'])
        ids.append(sample['id'])
        if "vloc" in sample.keys(): vlocs.append(sample["vloc"])
        
    return {
        'image': torch.stack(images, dim=0),
        'coords': torch.stack(coords, dim=0),
        'location_a': torch.stack(locations_a, dim=0),
        'location_v': torch.stack(locations_v, dim=0),
        'laterality': torch.stack(lats, dim=0),
        'modality': modalities, # just a basic list
        'id': ids, # just a basic list
        "vloc": torch.stack(vlocs, dim=0) if any(vlocs) else None
    }
