#!/usr/bin/env python
# coding: utf-8

# In[31]:


import cv2, os

cap = cv2.VideoCapture("")
os.makedirs("", exist_ok = True)

n, saved = 0,0
while True:
    ret, frame = cap.read()
    if not ret: break
   
    cv2.imwrite(f"case/frame_{n:06d}.png", frame)
    saved += 1
    n += 1
    
cap.release()
print(f"Saved {saved} frames")


# In[33]:


import tifffile, cv2, os, numpy as np

frames = tifffile.imread("")
os.makedirs("", exist_ok = True)

for i, frame in enumerate(frames):
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(f"case/frame_{i:06d}.png", frame)
    
print(f"Saved {len(frames)} frames")

