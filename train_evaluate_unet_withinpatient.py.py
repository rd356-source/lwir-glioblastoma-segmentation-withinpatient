#!/usr/bin/env python
# coding: utf-8

# In[20]:


import torch


# In[21]:


import os, re, json, random, math, time
import numpy as np
import cv2
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


# In[22]:


FRAMES_DIR = ""
MASKS_DIR = ""
IMG_SIZE = 256
BASE_CHANNELS = 32
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 150
PATIENCE = 30
THRESHOLD = 0.5


# In[23]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# In[24]:


def create_splits():
    random.seed(42)
    frames = sorted(f for f in os.listdir(FRAMES_DIR))
    random.shuffle(frames)
    n = len(frames)
    n_test, n_val = max(1,int(n*0.10)), max(1,int(n*0.20))
    n_train = n - n_test - n_val
    
    result = {"fold0": {
        "train": list(frames[:n_train]),
        "val":   list(frames[n_train:n_train+n_val]),
        "test":  list(frames[n_train+n_val:])}}
    print(f"Split: {n_train} train | {n_val} val | {n_test} test")
    
    json.dump(result, open("splits.json", "w"), indent=2)
    print("Saved -> splits.json")


# In[25]:


def aug_fn(img,mask):
    if random.random() < 0.3:
        img = np.fliplr(img).copy()
        mask = np.fliplr(mask).copy()
    return img, mask


# In[26]:


def get_number(f):
    nums = re.findall(r'\d+', f)
    return int(nums[0]) if nums else -1


# In[27]:


class LWIRDataset(Dataset):
    def __init__(self, split, file_list):
        self.split = split
        self.frames = sorted([os.path.join(FRAMES_DIR, f) for f in file_list],
                             key=lambda p: get_number(os.path.basename(p)))
        self.masks = {}
        if os.path.exists(MASKS_DIR):
            for f in os.listdir(MASKS_DIR):
                if f.endswith(".png"):
                    self.masks[get_number(f)] = os.path.join(MASKS_DIR, f)
        print(f"{split}: {len(self.frames)} frames | {len(self.masks)} masks")
        
    def __len__(self): return len(self.frames)
    
    def __getitem__(self, idx):
        img = np.array(Image.open(self.frames[idx]).resize((IMG_SIZE,IMG_SIZE)), dtype=np.float32)
        mp = self.masks[get_number(os.path.basename(self.frames[idx]))]
        mask = (np.array(Image.open(mp).convert("L").resize((IMG_SIZE,IMG_SIZE),Image.NEAREST))>127).astype(bool)
        
        if self.split == "train":
            img,mask = aug_fn(img,mask)
            
        img = (img-img.mean())/(img.std()+1e-8)
        img = np.nan_to_num(img)
        return torch.tensor(img).unsqueeze(0), torch.tensor(mask.astype(np.float32)).unsqueeze(0)


# In[28]:


def dconv(i, o):
    return nn.Sequential(
    nn.Conv2d(i,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True),
    nn.Conv2d(o,o,3,padding=1,bias=False),nn.BatchNorm2d(o),nn.ReLU(inplace=True))


# In[29]:


class Down(nn.Module):
    def __init__(self,i,o):
        super().__init__(); self.c=dconv(i,o); self.p=nn.MaxPool2d(2)
    def forward(self, x):
        s=self.c(x); return s, self.p(s)


# In[30]:


class Up(nn.Module):
    def __init__(self,i,o):
        super().__init__();
        self.u=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True);
        self.c=dconv(i,o)
    def forward(self,x,s):
        return self.c(torch.cat([self.u(x),s],dim=1))


# In[31]:


class UNet(nn.Module):
    def __init__(self,base=32):
        super().__init__(); b=base
        self.d1,self.d2,self.d3,self.d4 = Down(1,b),Down(b,b*2),Down(b*2,b*4),Down(b*4,b*8)
        self.bn = dconv(b*8,b*16)
        self.u4,self.u3,self.u2,self.u1 = Up(b*16+b*8,b*8),Up(b*8+b*4,b*4),Up(b*4+b*2,b*2),Up(b*2+b,b)
        self.out = nn.Conv2d(b,1,1)
        
    def forward(self,x):
        s1,x=self.d1(x); s2,x=self.d2(x); s3,x=self.d3(x); s4,x=self.d4(x)
        x = self.bn(x)
        x = self.u4(x,s4); x=self.u3(x,s3); x = self.u2(x,s2); x=self.u1(x,s1)
        return self.out(x)


# In[32]:


def compute_pos_weight():
    total_pos, total_neg = 0.0, 0.0
    for f in os.listdir(MASKS_DIR):
        if not f.endswith(".png"): continue
        m = cv2.imread(os.path.join(MASKS_DIR, f), 0)
        if m is not None:
            b = (m > 127).astype(np.float32)
            total_pos += b.sum(); total_neg += (1-b).sum()
    weight = float(min(total_neg/(total_pos + 1e-6), 10.0))
    print(f"Pos weight: {weight:.2f}x (tumour = {total_pos+total_neg}*100:.1f)% of pixels)")
    return torch.tensor(weight).to(device)


# In[33]:


def tversky_loss(p, t, alpha=0.7, beta=0.3, s=1e-6):
    p = torch.sigmoid(p)
    tp = (p*t).sum()
    fp = (p*(1-t)).sum()
    fn = ((1-p)*t).sum()
    return 1-(tp+s)/(tp + alpha*fn + beta*fp + s)

def total_loss(p,t,pos_weight):
    bce= nn.BCEWithLogitsLoss(pos_weight=pos_weight)(p,t)
    tvk = tversky_loss(p,t)
    return bce + tvk

def dice_score(p,t,s=1e-6):
    p=(torch.sigmoid(p)>THRESHOLD).float()
    return ((2*(p*t).sum()+s)/(p.sum()+t.sum()+s)).item()


# In[34]:


def plot_training_curve(train_losses, val_losses, val_dices, fold):
    epochs = range(1, len(train_losses) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.plot(epochs, train_losses, color="#0D9488", label="Training loss", linewidth=2)
    ax1.plot(epochs, val_losses, color="#94A3B8", label="Val loss", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", color="#0D9488", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="#0D9488")
    
    ax2 = ax1.twinx()
    ax2.plot(epochs, val_dices, color="#10B981", label="Val Dice")
    ax2.tick_params(axis="y", labelcolor="#10B981")
    ax2.set_ylim(0, 1)
    
    best_epoch = val_dices.index(max(val_dices)) + 1
    ax2.axvline(x=best_epoch, color = "#F59E0B", linestyle=":", linewidth=1.5, label= f"Best Dice (epoch {best_epoch})")
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="center right", fontsize=10)
    
    plt.title(f"Training Curve - LWIR Tumour Segmentation ({fold})", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"training_curve_{fold}.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved -> training_curve_{fold}.png")


# In[35]:


def train(pos_weight):
    print(f"Training on: {device}")
    
    splits = json.load(open("splits.json"))
    fold = list(splits.keys())[0]
    to_list = lambda x: [x] if isinstance(x, str) else x
    train_dl = DataLoader(LWIRDataset("train", to_list(splits[fold]["train"])), BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(LWIRDataset("val", to_list(splits[fold]["val"])), BATCH_SIZE)
    model = UNet(BASE_CHANNELS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=7, factor =0.5)
    best, no_imp = 0,0
    
    train_losses, val_losses, val_dices = [], [], []
    
    for epoch in range(1, EPOCHS+1):
        model.train(); tl=0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = total_loss(model(x), y, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            opt.step()
            tl += loss.item()
        tl /= len(train_dl)
    
        
        model.eval(); vl,vd=0,0
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(device),y.to(device)
                out = model(x)
                vl+=total_loss(out,y, pos_weight).item()
                vd+=dice_score(out,y)
        vl/=len(val_dl); vd/=len(val_dl)
                
        print(f"Epoch {epoch:3d} | train {tl:.4f} | val {vl:.4f} | dice {vd:.4f}")
        
        train_losses.append(round(tl, 4))
        val_losses.append(round(vl, 4))
        val_dices.append(round(vd, 4))
        
                
        if vd>best:
            best=vd; torch.save(model.state_dict(),f"best_{fold}.pth")
            print(" -> saved (dice {best:.4f})")
            no_imp=0
            
        else:
            no_imp+=1
            
        sched.step(vl)
        if no_imp>=PATIENCE: print("Early stopping"); break
            
    print(f"Training done. Best dice: {best:.4f}")
    
    json.dump({"train_loss": train_losses, "val_loss": val_losses, "val_dice": val_dices},
              open(f"training_log_{fold}.json", "w"), indent=2)
    plot_training_curve(train_losses, val_losses, val_dices, fold)
    return model


# In[36]:


def evaluate(model, device):
    fold = list(json.load(open("splits.json")).keys())[0]
    to_list = lambda x: [x] if isinstance(x, str) else x
    test_ds = LWIRDataset("test", to_list(json.load(open("splits.json"))[fold]["test"]))
    model.load_state_dict(torch.load(f"best_{fold}.pth", map_location=device))
    model.eval()
    ds, as_ = [], []
    mt, e2e = [], []
    
    with torch.no_grad():
        for idx in range (len(test_ds)):
            t0 = time.perf_counter()
            img, mask = test_ds[idx]
            inp = img.unsqueeze(0).to(device)
            
            tl = time.perf_counter()
            out = model(inp)
            mt.append((time.perf_counter()-tl)*1000)
            
            pred =  (out>THRESHOLD).float()
            e2e.append((time.perf_counter()-t0)*1000)
            
            m=mask.unsqueeze(0).to(device)
            s=1e-6
            ds.append(((2*(pred*m).sum()+s)/(pred.sum()+m.sum()+s)).item())
            
            p = pred.squeeze().cpu().numpy().astype(bool)
            g = mask.squeeze().numpy().astype(bool)
            if p.any() and g.any():
                as_.append(float((distance_transform_edt(~g)[p].mean()+distance_transform_edt(~p)[g].mean())/2))
            else:
                as_.append(0.0)
                
        print(f"\nTest Results:")
        print(f" Dice: {np.mean(ds):.4f}")
        print(f" ASSD: {np.mean(as_):.2f} px")
        print(f" Model only mean: {np.mean(mt):.1f} ms p95 {np.percentile(mt,95):.1f} ms")
        print(f" End to end mean: {np.mean(e2e):.1f} ms p95 {np.percentile(e2e,95):.1f} ms")
                


# In[38]:


def overlay(model, device):
    fold = list(json.load(open("splits.json")).keys())[0]
    to_list = lambda x: [x] if isinstance(x, str) else x
    test_ds = LWIRDataset("test", to_list(json.load(open("splits.json"))[fold]["test"]))
    model.eval()
    os.makedirs("overlays", exist_ok=True); frames=[]
    
    with torch.no_grad():
        for i in range(len(test_ds)):
            img_t, mask_t = test_ds[i]
            inp = img_t.unsqueeze(0).to(device)
            pred = (torch.sigmoid(model(inp)) > THRESHOLD).squeeze().cpu().numpy().astype(bool)
            
            frame = img_t.squeeze().numpy()
            frame_disp = ((frame-frame.min())/(frame.max()-frame.min())*255).astype(np.uint8)
            bgr = cv2.cvtColor(frame_disp, cv2.COLOR_GRAY2BGR)
            
            mask_disp = (mask_t.squeeze().numpy().astype(np.uint8))*255
            
            pred_disp = (pred.astype(np.uint8))*255
            
            ov = bgr.copy(); ov[pred] = (0, 200, 0)
            out = cv2.addWeighted(bgr, 0.6, ov, 0.4, 0)
            
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            ax[0].imshow(frame_disp, cmap='gray'); ax[0].set_title("Preprocessed LWIR"); ax[0].axis('off')
            ax[1].imshow(mask_disp, cmap='gray'); ax[1].set_title("Expert Annotated Mask"); ax[1].axis('off')
            ax[2].imshow(pred_disp, cmap='gray'); ax[2].set_title("Predicted Mask"); ax[2].axis('off')
            ax[3].imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); ax[3].set_title("Overlay"); ax[3].axis('off')
            
            plt.tight_layout(pad=1.5)
            plt.subplots_adjust(top=0.8)
            
            fig.canvas.draw()
            panel_img = np.asarray(fig.canvas.buffer_rgba())
            panel_img = cv2.cvtColor(panel_img, cv2.COLOR_RGB2BGR)
            plt.close(fig)
            
            cv2.imwrite(f"overlays/overlay_{i:05d}.png", panel_img)
            frames.append(panel_img)
            
    h, w, _ = frames[0].shape
    vid = cv2.VideoWriter("overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 1, (w, h))
    for f in frames: vid.write(f)
    vid.release()
                       
                       
if __name__ == "__main__":
    create_splits()
    pos_weight = compute_pos_weight()
    model = train(pos_weight)
    evaluate(model, device)
    overlay(model, device)


# In[37]:

# In[ ]:





# In[ ]:




