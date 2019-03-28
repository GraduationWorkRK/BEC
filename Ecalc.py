#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
np.set_printoptions(linewidth=500)
np.set_printoptions(precision=8)

import random


# In[ ]:


#大きさを返す
def norm(r):
    return(np.sqrt(np.real(np.dot(r.conjugate(),r))))


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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


# In[ ]:


#エネルギーを計算
def Espin(c11,c12,c22,c112,c212,s,r):
    ans=(c11*F11(s)+c12*F22(r)+c22*A0(r))/2.0+c112*F12(s,r)+c212*P112beta(s,r)
    return(np.real(ans))


# In[ ]:


#適当な移動
def Move(s1,move):
    test=s1+move
    return test/norm(test)


# In[ ]:


#maxstep回動かなくなるまで刻みhでエネルギー低い方に移動
def Eoptimize(c11,c12,c22,c112,c212,s1,s2,maxstep,h):
    Einit=10000.
    ikeep=0
    while True:
        E=Espin(c11,c12,c22,c112,c212,s1,s2)
        test1=Move(s1,rands1()*h)
        test2=Move(s2,rands2()*h)
        Etest=Espin(c11,c12,c22,c112,c212,test1,test2)
        if E>Etest:
            s1=test1
            s2=test2
            E=Etest
            ikeep=0
        else:
            ikeep+=1
        if ikeep==maxstep:
            return(E,s1,s2)


# In[ ]:


#刻み幅変えてより低いエネルギーへ
def Eopt(c11,c12,c22,c112,c212):
    s1test=rands1()
    s2test=rands2()
    Etest,s1test,s2test=Eoptimize(c11,c12,c22,c112,c212,s1test,s2test,100,1)
    Etest,s1test,s2test=Eoptimize(c11,c12,c22,c112,c212,s1test,s2test,100,.1)
    Etest,s1test,s2test=Eoptimize(c11,c12,c22,c112,c212,s1test,s2test,100,.01)
    Etest,s1test,s2test=Eoptimize(c11,c12,c22,c112,c212,s1test,s2test,100,.001)

    return(Etest,s1test,s2test)


# In[ ]:


#初期値変えて何度も試す
def Emin(c11,c12,c22,c112,c212):
    E=10000
    tall=50
    keep=0
    while True:
        Etest,s1test,s2test=Eopt(c11,c12,c22,c112,c212)      
        if E>Etest:
            E=Etest
            sans1=s1test
            sans2=s2test
            keep=0
        else:
            keep+=1
        if keep==tall:
            return(E,sans1,sans2)


# In[ ]:


#状態を書き出す
RUN = 50
c11= RUN/50.
c12= RUN/50.
c112, c212=0.5, .0
for c22 in np.arange(-1.,1.02,.02):
    E,s1,s2=Emin(c11,c12,c22,c112,c212)
    ans=[c11,c12,c22,c112,c212,s1,s2]
    a_str=[str(a) for a in ans ]
    with open("data2_{:+06.3f}_{:+06.3f}.txt".format(c11,c22), mode="a") as f:
        s = "{c[0]:6.3f} {c[1]:6.3f} {c[2]:6.3f} {c[3]:6.3f} {c[4]:6.3f}"         " {s1[0].real:15.8E} {s1[0].imag:15.8E}"         " {s1[1].real:15.8E} {s1[1].imag:15.8E}"         " {s1[2].real:15.8E} {s1[2].imag:15.8E}"         " {s2[0].real:15.8E} {s2[0].imag:15.8E}"         " {s2[1].real:15.8E} {s2[1].imag:15.8E}"         " {s2[2].real:15.8E} {s2[2].imag:15.8E}"         " {s2[3].real:15.8E} {s2[3].imag:15.8E}"         " {s2[4].real:15.8E} {s2[4].imag:15.8E}\n"
        f.write(s.format(c=[c11,c12,c22,c112,c212],s1=s1,s2=s2))

