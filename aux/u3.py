import numpy as np

def expi(alpha):
  real=np.cos(alpha)
  imag=np.sin(alpha)
  return real,imag,"{0}+i{1}".format(real,imag)

def u1(thet):
  a00=1
  a01=0
  a10=0
  a11=exp(i(thet/2))

  return

def u3(thet,phi,lam):
  a00=cos(thet/2)
  a01=-exp(i*lam)sin(thet/2)
  a10=exp(i*phi)sin(thet/2)
  a11=exp(i(lam+phi))cos(thet/2)
  prunt
