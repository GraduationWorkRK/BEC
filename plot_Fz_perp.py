#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd

import random

import sys

np.set_printoptions(linewidth=500)
np.set_printoptions(precision=4, floatmode='fixed')


#大きさを返す
def norm(r):
    return(np.sqrt(np.real(np.dot(r.conjugate(),r))))


#ランダムな複素数を作る
def randC():
    return(complex(random.uniform(-1,1),random.uniform(-1,1)))
#スピン１、スピン２のスピノルを作る（規格化込み）
def spin1(s1,s0,sm1):
    a=np.array([s1,s0,sm1])
    a=a/norm(a)
    return(a)
def spin2(r2,r1,r0,rm1,rm2):
    a=np.array([r2,r1,r0,rm1,rm2])
    a=a/norm(a)
    return(a)
def rands1():
    a=[random.uniform(-1,1) for i in range(6)]
    return(spin1(complex(a[0],a[1]),complex(a[2],a[3]),complex(a[4],a[5])))
def rands2():
    a=[random.uniform(-1,1) for i in range(10)]
    return(spin2(complex(a[0],a[1]),complex(a[2],a[3]),complex(a[4],a[5]),complex(a[6],a[7]),complex(a[8],a[9])))


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
#    vecplot(F1(s),F2(r))
    return(F)


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

#逆行列はnp.linalg.inv(A)で出る


#成分を実数にするように位相変換
def tau_0real(s):
    tau=np.arctan(-s[0].imag/s[0].real)
    ans=np.exp(1.0j*tau)*s
    return(ans)
def tau_1real(s):
    tau=np.arctan(-s[1].imag/s[1].real)
    ans=np.exp(1.0j*tau)*s
    return(ans)
def tau_2real(s):
    tau=np.arctan(-s[2].imag/s[2].real)
    ans=np.exp(1.0j*tau)*s
    return(ans)


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


def Fxsize(s1,s2):
    rs1,rs2=RotFz(s1,s2)
    f1x=np.dot(rs1.conjugate(),np.dot(F3x,rs1)).real
    f2x=np.dot(rs2.conjugate(),np.dot(F5x,rs2)).real
    return(f1x,f2x)
def Fysize(s1,s2):
    rs1,rs2=RotFz(s1,s2)
    f1y=np.dot(rs1.conjugate(),np.dot(F3y,rs1)).real
    f2y=np.dot(rs2.conjugate(),np.dot(F5y,rs2)).real
    return(f1y,f2y)
def Fzsize(s1,s2):
    rs1,rs2=RotFz(s1,s2)
    f1z=np.dot(rs1.conjugate(),np.dot(F3z,rs1)).real
    f2z=np.dot(rs2.conjugate(),np.dot(F5z,rs2)).real
    return(f1z,f2z)



c22=-0.5

c11=[]
f2ps=[]
#c1=np.linspace(-1.,1.,101)

data = pd.read_table('data_mixed.txt', header=None, delim_whitespace=True)
print(data)

data_c22_fix = data[data[2]==c22]
print(data_c22_fix)

for i in range(len(data_c22_fix)):
    data_tmp = data_c22_fix.iloc[i]
    c = data_tmp[0:5]
    c11.append(c[0])
    s1 = np.array([complex(data_tmp[k],data_tmp[k+1]) for k in [5,7,9]])
    s2 = np.array([complex(data_tmp[k],data_tmp[k+1]) for k in [11,13,15,17,19]])

    f1x,f2x=Fxsize(s1,s2)
    f1y,f2y=Fysize(s1,s2)
    f2p=np.sqrt(f2x**2+f2y**2)
    f2ps.append(f2p)

fontsize = 18
plt.rcParams["font.size"] = fontsize
plt.plot(c11,f2ps)
plt.ylim(0,1)
plt.xlabel('$c_1^{(1)},c_1^{(2)}$')
plt.ylabel(r'$F_\perp^{(2)}$')

plt.tight_layout()
plt.savefig("Fpc22_-05")
