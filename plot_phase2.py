#!/usr/bin/env python
# coding: utf-8

# In[ ]:




import numpy as np
np.set_printoptions(linewidth=500)
np.set_printoptions(precision=8, floatmode='fixed')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd


#大きさを返す
def norm(r):
    return(np.sqrt(np.real(np.dot(r.conjugate(),r))))

#スピン１、スピン２のスピノルを作る（規格化込み）
def spin1(spin):
    a=np.array(spin)
    a=a/norm(a)
    return(a)
def spin2(spin2):
    a=np.array([r2,r1,r0,rm1,rm2])
    a=a/norm(a)
    return(a)


#スピン演算子
rt2=np.sqrt(2.0)
F3x=np.array([[0.0,rt2,0.0],
              [rt2,0.0,rt2],
              [0.0,rt2,0.0]
             ])/2.0
F3y=np.array([[0.0,rt2,0.0],
              [-rt2,0.0,rt2],
              [0.0,-rt2,0.0]
             ])/2.0j
F3z=np.array([[1.0,0.0,0.0],
              [0.0,0.0,0.0],
              [0.0,0.0,-1.0]
             ])

rt6=np.sqrt(6.0)
F5x=(np.array([[0.0,2.0,0.0,0.0,0.0],
               [2.0,0.0,rt6,0.0,0.0],
               [0.0,rt6,0.0,rt6,0.0],
               [0.0,0.0,rt6,0.0,2.0],
               [0.0,0.0,0.0,2.0,0.0]
              ]))/2.0
F5y=(np.array([[0.0,2.0,0.0,0.0,0.0],
                [-2.0,0.0,rt6,0.0,0.0],
                [0.0,-rt6,0.0,rt6,0.0],
                [0.0,0.0,-rt6,0.0,2.0],
                [0.0,0.0,0.0,-2.0,0.0]
               ]))/2.0j
F5z=np.array([[2.0,0.0,0.0,0.0,0.0],
              [0.0,1.0,0.0,0.0,0.0],
              [0.0,0.0,0.0,0.0,0.0],
              [0.0,0.0,0.0,-1.0,0.0],
              [0.0,0.0,0.0,0.0,-2.0]
             ])

#回転行列
l3x,P3x=np.linalg.eig(F3x)
l5x,P5x=np.linalg.eig(F5x)
l3y,P3y=np.linalg.eig(F3y)
l5y,P5y=np.linalg.eig(F5y)
l3z,P3z=np.linalg.eig(F3z)
l5z,P5z=np.linalg.eig(F5z)
def R3x(t,s):
    A=np.diag(np.exp(-1.0j*l3x*t))
    R=np.dot(P3x,np.dot(A,np.conjugate(P3x.T)))
    ans=np.dot(R,s)
    return(ans)

def R5x(t,s):
    A=np.diag(np.exp(-1.0j*l5x*t))
    R=np.dot(P5x,np.dot(A,np.conjugate(P5x.T)))
    ans=np.dot(R,s)
    return(ans)

def R3y(t,s):
    A=np.diag(np.exp(-1.0j*l3y*t))
    R=np.dot(P3y,np.dot(A,np.conjugate(P3y.T)))
    ans=np.dot(R,s)
    return(ans)

def R5y(t,s):
    A=np.diag(np.exp(-1.0j*l5y*t))
    R=np.dot(P5y,np.dot(A,np.conjugate(P5y.T)))
    ans=np.dot(R,s)
    return(ans)

def R3z(t,s):
    A=np.diag(np.exp(-1.0j*l3z*t))
    R=np.dot(P3z,np.dot(A,np.conjugate(P3z.T)))
    ans=np.dot(R,s)
    return(ans)

def R5z(t,s):
    A=np.diag(np.exp(-1.0j*l5z*t))
    R=np.dot(P5z,np.dot(A,np.conjugate(P5z.T)))
    ans=np.dot(R,s)
    return(ans)

#A_0^2
P02=np.array([[0.0,0.0,0.0,0.0,1.0],
              [0.0,0.0,0.0,-1.0,0.0],
              [0.0,0.0,1.0,0.0,0.0],
              [0.0,-1.0,0.0,0.0,0.0],
              [1.0,0.0,0.0,0.0,0.0]
             ])/np.sqrt(5.0)

def A0(r):
    F=np.dot(r,np.dot(P02,r))
    Fans=np.dot(F.conjugate(),F).real
    return(Fans)


#F_(m)^(f)
def F1(s):
    Fx=np.dot(s.conjugate(),np.dot(F3x,s))
    Fy=np.dot(s.conjugate(),np.dot(F3y,s))
    Fz=np.dot(s.conjugate(),np.dot(F3z,s))
    return(np.array([Fx,Fy,Fz]).real)

def F2(r):
    Fx=np.dot(r.conjugate(),np.dot(F5x,r))
    Fy=np.dot(r.conjugate(),np.dot(F5y,r))
    Fz=np.dot(r.conjugate(),np.dot(F5z,r))
    return(np.array([Fx,Fy,Fz]).real)

def F11(s):
    F=np.dot(F1(s),F1(s))
    return(F)

def F22(r):
    F=np.dot(F2(r),F2(r))
    return(F)

def F12(s,r):
    F=np.dot(F1(s),F2(r))
    return(F)


#P_1^12論文のベタ打ち
r01=np.sqrt(1.0/10.0)
r04=np.sqrt(2.0/5.0)
r03=np.sqrt(3.0/10.0)
r06=np.sqrt(3.0/5.0)
def P112beta(s,r):
    A11=(s[0]*r[2]*r01-s[1]*r[1]*r03+s[2]*r[0]*r06)
    A10=(s[0]*r[3]*r03-s[1]*r[2]*r04+s[2]*r[1]*r03)
    A1m1=(s[0]*r[4]*r06-s[1]*r[3]*r03+s[2]*r[2]*r01)
    ans=np.dot(A11.conjugate(),A11)+np.dot(A10.conjugate(),A10)+np.dot(A1m1.conjugate(),A1m1)
    return(ans.real)
def zrad(s,r):
    S=F1(s)
    Sx,Sy,Sz=F1(s)
    a=np.arctan2(Sy,Sx)
    ans1=R3z(-a,s)
    ans2=R5z(-a,r)
    return(a,ans1,ans2)


def yrad(s,r):
    S=F1(s)
    Sx,Sy,Sz=F1(s)
    a=np.arctan2(Sx,Sz)
    ans=R3y(-a,s)
    ans2=R5y(-a,r)
    return(a,ans,ans2)


def RotFz(s1,s2):
    tau,rs1,rs2=zrad(s1,s2)
    beta,rrs1,rrs2=yrad(rs1,rs2)
    return(rrs1,rrs2)


def P12(s1,s2):
    rs1,rs2=RotFz(s1,s2)
    return(P112beta(rs1,rs2))


#状態を決めて色指定
def CheckState(s1,s2):
    F_1=norm(F1(s1))
    F_2=norm(F2(s2))
    F12dot=F12(s1,s2)
    F12cross=norm(np.cross(F1(s1),F2(s2)))
    A_0=A0(s2)
    P_12=P12(s1,s2)
    if abs(F_1-1)<1e-3:
        if abs(F_2-2)<1e-3:
            if F12dot>0:
                return("crimson")#FF+
            else:
                return("deeppink")#FF-
        elif abs(F_2-1)<1e-3:
            if F12dot>0:
                return("yellow")#FF'+
            else:
                return("khaki")#FF'-
        elif abs(F_2)<1e-3:
            if abs(P_12-.1)<1e-3:
                return("greenyellow")#FU
            elif abs(P_12-.2)<1e-3:
                return("brown")#FC
            elif abs(P_12-.3)<1e-3:
                return("olive")#FB
            else:
                return("black")#error
            
        else:
            if abs(F12cross)<1e-3:
                if abs(A_0)<1e-3:
                    if F12dot>0:
                        return("chocolate")#d+
                    else:
                        return("peru")#d-
                else:
                    if F12dot>0:
                        return("darkgreen")#c+ 
                    else:
                        return("limegreen")#c-
            else:
                return("black")#error
    elif abs(F_1)<1e-3:
        if abs(F_2-2)<1e-3:
            return("green")#PF
        elif abs(F_2-1)<1e-3:
            return("lightgreen")#PF'
        elif abs(F_2)<1e-3:
            if abs(A_0-.2)<1e-3:

                if abs(P_12-.4)<1e-3:
                    return("navy")#PU
                elif abs(P_12)<1e-3:
                    return("fuchsia")#PB

            elif abs(A_0)<1e-3:
                return("orange")#PC
            else:
                return("plum")#e
        else:
            return("black")#error
    else:
        #a,bの判定
        if abs(F_2)>1e-3 and abs(A_0)>1e-3:
            if abs(F12cross)<1e-3:
                if F12dot>0:
                    return("dodgerblue")#a+
                else:
                    return("skyblue")#a-
            else:
                if F12dot>0:
                    return("orangered")#b+
                else:
                    return("coral")#b-
        else:
            return("black")#error


fontsize=18
plt.rcParams["font.size"] = fontsize
plt.figure(figsize=(5,5))
plt.xlabel('$c_1^{(1)},c_1^{(2)}$')
plt.ylabel('$c_2^{(2)}$')

data = pd.read_table('data_mixed.txt', header=None, delim_whitespace=True)
print(data)

for i in range(len(data)):
    data_tmp = data.iloc[i]
    c = data_tmp[0:5]  # c = [c11,c12,c22,c112,c212]
    s1 = np.array([complex(data_tmp[k],data_tmp[k+1]) for k in [5,7,9]])
    s2 = np.array([complex(data_tmp[k],data_tmp[k+1]) for k in [11,13,15,17,19]])
    plt.plot(c[0],c[2],".",color=CheckState(s1,s2))

plt.tight_layout()
plt.savefig("phasediagram4.png")




