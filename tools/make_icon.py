# Draws platform/macos/AppIcon.icns (black background, glowing blue ring). Run from the repo root: python3 tools/make_icon.py
import numpy as np
from PIL import Image, ImageFilter
S=2048; c=S/2
y,x=np.mgrid[0:S,0:S].astype(np.float32)+0.5
# macOS-style rounded square (824/1024 content, radius 185/1024)
half=824/1024*S/2; rad=185/1024*S
dx=np.maximum(np.abs(x-c)-(half-rad),0); dy=np.maximum(np.abs(y-c)-(half-rad),0)
inside=np.clip((rad-np.sqrt(dx*dx+dy*dy))*1.0+0.5,0,1)*((np.abs(x-c)<=half)&(np.abs(y-c)<=half))
# background: near-black with a faint blue centre
r=np.sqrt((x-c)**2+(y-c)**2)/half
bg=np.zeros((S,S,3),np.float32)
bg[...,0]=0.010+0.020*np.clip(1-r,0,1)**2
bg[...,1]=0.014+0.035*np.clip(1-r,0,1)**2
bg[...,2]=0.022+0.080*np.clip(1-r,0,1)**2
# faint stars
rng=np.random.default_rng(7)
for _ in range(90):
    sx,sy=rng.uniform(c-half*0.92,c+half*0.92,2); b=rng.uniform(0.15,0.55)**2; s=rng.uniform(1.2,2.8)
    m=np.exp(-((x-sx)**2+(y-sy)**2)/(2*s*s))*b
    bg+=m[...,None]*np.array([0.8,0.9,1.0],np.float32)
# ring: radius R, band half-width h, viewed at tilt phi, rotated by rot
R=0.74*half; h=0.085*half; phi=np.radians(68); rot=np.radians(-18)
cr,sr=np.cos(rot),np.sin(rot)
X=(x-c)*cr+(y-c)*sr; Y=-(x-c)*sr+(y-c)*cr
cphi,sphi=np.cos(phi),np.sin(phi)
col=np.zeros((S,S,3),np.float32); alpha=np.zeros((S,S),np.float32)
ct=np.clip(X/R,-1,1); valid=np.abs(X)<=R
def band(sign):
    st=sign*np.sqrt(np.maximum(1-ct*ct,0))
    yc=R*st*cphi                     # centre line of the band
    half_w=h*sphi                    # band's vertical extent on screen
    d=np.abs(Y-yc)
    a=np.clip((half_w-d)*1.2+0.5,0,1)*valid
    # edge fade by |x| near R to keep anti-aliasing at the tips
    a*=np.clip((R-np.abs(X))*0.6+0.2,0,1)
    t=(Y-yc)/half_w                  # -1..1 across the band
    theta=np.arctan2(st,ct)
    light=0.55+0.45*np.cos(theta-np.radians(-130))   # lit from upper left
    edge=1-0.35*t*t
    return a,light*edge,t
# back (far) half first: inner surface visible, a bit darker
for sign,(base,hi) in ((-1,(np.array([0.05,0.30,0.85]),np.array([0.45,0.80,1.0]))),
                       (1,(np.array([0.08,0.42,1.0]),np.array([0.70,0.93,1.0])))):
    a,l,t=band(sign)
    shade=np.clip(l,0,1)[...,None]
    cc=base*(0.35+0.65*shade)+hi*np.clip(shade-0.55,0,1)*1.6
    # thin bright rim lines on both edges
    rim=np.exp(-((np.abs(t)-0.85)/0.07)**2)[...,None]*0.5*shade
    cc=np.clip(cc+rim*np.array([0.6,0.85,1.0]),0,1.2)
    k=1.0 if sign>0 else 0.85
    col=col*(1-a[...,None])+cc*k*a[...,None]; alpha=np.maximum(alpha,a)
# glow
gm=Image.fromarray((alpha*255).astype(np.uint8))
g1=np.asarray(gm.filter(ImageFilter.GaussianBlur(S*0.012)),np.float32)/255
g2=np.asarray(gm.filter(ImageFilter.GaussianBlur(S*0.045)),np.float32)/255
glow=(g1*0.55+g2*0.75)[...,None]*np.array([0.10,0.45,1.0],np.float32)
img=bg+glow
img=img*(1-alpha[...,None])+col*alpha[...,None]+glow*0.25
img=1-np.exp(-img*1.4)                # soft tone map
img=np.clip(img/(1-np.exp(-1.4)),0,1)
rgba=np.dstack([img,inside]).astype(np.float32)
out=Image.fromarray((rgba*255+0.5).astype(np.uint8),'RGBA').resize((1024,1024),Image.LANCZOS)
out.save('build/icon_1024.png') if __import__('os').path.isdir('build') else None
out.save('platform/macos/AppIcon.icns',sizes=[(16,16),(32,32),(64,64),(128,128),(256,256),(512,512),(1024,1024)])
print('ok')
