import numpy as np
import copy, subprocess, os, yaml
import matplotlib.pyplot as plt
from random import randint
import matplotlib.animation as animation
from numpy.lib.type_check import nan_to_num
import initial, boundary, cfxx, rhs, hcal, newgrd, mkzero, cip1d
from matplotlib._version import get_versions as mplv
from matplotlib.animation import PillowWriter

os.system("del /Q .\png\*.png")

# Open Config File
with open('config.yml','r', encoding='utf-8') as yml:   
    config = yaml.load(yml)
 
xl=float(config['xl']); nx=int(config['nx'])
j_channel=int(config['j_channel'])
slope=float(config['slope'])
x_slope=float(config['x_slope'])
slope1=float(config['slope1']); slope2=float(config['slope2'])
xb1=float(config['xb1']); xb2=float(config['xb2']); xb3=float(config['xb3'])
dbed=float(config['dbed'])
xw1=float(config['xw1']); xw2=float(config['xw2']); xw3=float(config['xw3'])
w0=float(config['w0']);w1=float(config['w1']);w2=float(config['w2'])
w3=float(config['w3']);w4=float(config['w4'])

qp=float(config['qp']); g=float(config['g']); snm=float(config['snm'])
alh=float(config['alh']); lmax=int(config['lmax']); errmax=float(config['errmax'])
hmin=float(config['hmin'])
j_upstm=int(config['j_upstm']); j_dwstm=int(config['j_dwstm'])
etime=float(config['etime']); dt=float(config['dt']); tuk=float(config['tuk'])
alpha_up=float(config['alpha_up']); alpha_dw=float(config['alpha_dw'])

nx1=nx+1; nx2=nx+2
dx=xl/nx; xct=xl/2.
x=np.linspace(0,xl,nx+1)
x_cell=np.zeros([nx2])
x_cell=initial.x_cell_init(x_cell,x,dx,nx)
it_out=int(tuk/dt)

# Make Array
hs=np.zeros([nx2]); hs_up=np.zeros([nx2]); fr=np.zeros([nx2])
h=np.zeros([nx2]); hn=np.zeros([nx2]); h_up=np.zeros([nx2])
u=np.zeros([nx2]); un=np.zeros([nx2])
eta=np.zeros([nx2]); eta0=np.zeros([nx2]); eta_up=np.zeros([nx2]);eta_up0=np.zeros([nx2])
cfx=np.zeros([nx2]); qu=np.zeros([nx2])
gux=np.zeros([nx2]); gux_n=np.zeros([nx2])
fn=np.zeros([nx2]);gxn=np.zeros([nx2])
h0_up=np.zeros([nx2]); hc_up=np.zeros([nx2]); hs0_up=np.zeros([nx2])
bslope=np.zeros([nx2]); bslope_up=np.zeros([nx2])
b=np.zeros([nx2]); b_up=np.zeros([nx2]); u0_up=np.zeros([nx2])


# Geometric Condition
if j_channel==1:
    eta,eta0,eta_up,eta_up0=initial.eta_init \
        (eta,eta0,eta_up,eta_up0,nx,dx, \
            slope,xl,xb1,xb2,xb3,dbed)
else:
    eta,eta0,eta_up,eta_up0=initial.eta_init_2 \
        (eta,eta0,eta_up,eta_up0,nx,dx, \
        xl,x_slope,slope1,slope2)

b,b_up=initial.width_init(b,b_up,nx,dx,xw1,xw2,xw3,xl, \
    w0,w1,w2,w3,w4)
bslope,bslope_up=initial.bslope_cal(nx,dx,bslope,bslope_up,eta,eta_up)

# Uniform Flow Depth and Critial Depth
hs0_up,h0_up,u0_up=initial.h0_cal(nx,dx,qp,snm,eta_up, \
    hs0_up,h0_up,b_up,bslope_up,u0_up)    
hc_up=initial.hc_cal(hc_up,qp,nx,b_up,g)

# Initial Depth and Water Surface Elevation
hs_upstm=hs0_up[0]*alpha_up 
hs_dwstm=hs0_up[nx]*alpha_dw

h,hs,h_up,hs_up=initial.h_init \
        (eta,eta0,eta_up,eta_up0,h,hs,h_up,hs_up,hs0_up, \
        hs_upstm,hs_dwstm,nx,dx,xl)

# Hydraulic and Phisical Parameters

u_upstm=qp/(b_up[0]*hs_upstm)
u_dwstm=qp/(b_up[nx]*hs_dwstm)

h,hs=boundary.h_bound(h,hs,eta,hs_upstm,hs_dwstm,nx,j_upstm,j_dwstm)
hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
h_up=boundary.h_up_cal(hs_up,eta_up,h_up,nx)
hn=copy.copy(h)

u,fr=initial.u_init(g,qp,u,qu,hs_up,b_up,fr,nx)
u=boundary.u_bound(u,hs_up,qp,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
un=copy.copy(u)

y_h0=np.zeros([nx+1]); y_hc=np.zeros([nx+1])

for i in np.arange(0,nx+1):
    y_h0[i]=h0_up[i]; y_hc[i]=eta_up[i]+hc_up[i]

# Seting for Plot

fig=plt.figure(figsize=(30,40))
ims=[]
flag_legend=True

# Upper Panel Left:Elevation Right:Velocity, Right: Width
ax1= fig.add_subplot(3,1,1)
im1= ax1.set_title("1D Open Channel Flow",fontsize=40)
im1= ax1.set_xlabel("x(m)",fontsize=20)
zmax=np.amax(h0_up)*1.2
im1= ax1.set_ylim(0, zmax)
im1= ax1.set_ylabel("Elevation(m)",fontsize=20)
ax1r=ax1.twinx() # Right Hand Vertial Axis
bmax=np.amax(b_up)*1.5
bmin=0.
im1r=ax1r.set_ylim(bmin,bmax)
im1r=ax1r.set_ylabel("Width",fontsize=20)

# Mid Pannel: Velocity
ax2=fig.add_subplot(3,1,2)
im2= ax2.set_xlabel("x(m)",fontsize=20)
umax=np.amax(u)*1.5
im2= ax2.set_ylim(0, umax)
im2= ax2.set_ylabel("Velocity(m/s)",fontsize=20)

#Lower Panel Left:Discharge Right:Froude Number
ax3= fig.add_subplot(3,1,3)
im3= ax3.set_xlabel("x(m)",fontsize=20)
qmax=np.amax(qp)*1.2
im3= ax3.set_ylim(0, qmax)
im3= ax3.set_ylabel("Discharge(m2/s)",fontsize=20)
ax4=ax3.twinx() # Right Vertical Axis
frmax=np.amax(fr)*2.5
frmin=np.amin(fr)*1.2
im4= ax4.set_ylim(frmin, frmax)
im4= ax4.set_ylabel("Froude Number",fontsize=20)

time=0.; err=0.; icount=0; nfile=0; l=0
################ Main #####################

while time<= etime:
    if icount%it_out==0:
        print('time=',np.round(time,3),l)

# Plot Calculated Values
        hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
        h_up=boundary.h_up_cal(hs_up,eta_up,h_up,nx)
        y=np.zeros([nx+1]); y1=np.zeros([nx+1]); y2=np.zeros([nx+1])
        y3=np.zeros([nx+1]); y4=np.zeros([nx+1]); yb=np.zeros([nx+1])
#        print(u[nx],hs_up[nx],qu[nx])
        for i in np.arange(0,nx+1):
            y[i]=eta_up[i]; yb[i]=b_up[i]
            y1[i]=h_up[i]
            y2[i]=u[i]
            y3[i]=qu[i]
            y4[i]=u[i]/np.sqrt(g*hs_up[i])

        
        im1= ax1.plot(x,y,'magenta',label='Bed',linewidth=5)
        im1r=ax1r.plot(x,yb,'green',label='Width',linewidth=5)
#        im11= ax1.plot(x,y1,linestyle = "dashed", color='blue',label="WSE",linewidth=2) 
        im1= ax1.plot(x_cell,h,'blue',label="WSE",linewidth=5) 
        if np.abs(dbed)<0.001:
            im_h0=ax1.plot(x,y_h0,linestyle = "dashed",color='green',label='h0')
            im0=ax1.text(x[nx],y_h0[nx],'h0',size='25')
        else:
            im_h0=""
            im0=""

        im_hc=ax1.plot(x,y_hc,linestyle = "dashed",color='black',label='hc')
        imc=ax1.text(x[nx],y_hc[nx],'hc',size='25')
        
        im2= ax2.plot(x,y2,'red',label='Velocity',linewidth=5)
        text1= ax1.text(0.,0.,"Time="+str(np.round(time,3))+"sec",size=30)
        lg0=ax1.text(0.,y[0],'Bed',size=30)
        lg0r=ax1r.text(0.,yb[0],'Width',size=30)

        lg1=ax1.text(0.,y1[0],'Water',size=30)
        text2= ax2.text(0.,0.,"Time="+str(np.round(time,3))+"sec",size=30)
        lg2=ax2.text(0.,y2[0],'Velocity',size=30)


        im3= ax3.plot(x,y3,'green',label='Dicharge',linewidth=5)
        im4= ax4.plot(x,y4,'black',label='Froude Number',linewidth=5)
        text3= ax3.text(0.,0.,"Time="+str(np.round(time,3))+"sec",size=30)
        lg3=ax3.text(0.,y3[0],'Discharge',size=30)
        lg4=ax4.text(0.,y4[0],'Fr',size=30)

# 
# 
# lag_legend:         
#            ax1.legend(bbox_to_anchor=(0, 1.0), loc='upper left', borderaxespad=0, fontsize=18)
#            ax2.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right', borderaxespad=0, fontsize=18)
#            plt.legend()
#            flag_legend=False   
        
#        nfile=nfile+1
#        fname="./png/" + 'f%04d' % nfile + '.png'
#        print(fname)
#        plt.savefig(fname)
        if np.abs(dbed)<0.001:
            itot=im_h0+im_hc+im1+im2+im3+im4+[text1]+[text2]+[text3]+ \
            [lg1]+[lg2]+[lg3]+[lg4]+[im0]+[imc]+im1r+[lg0r]
        else:
            itot=im_hc+im1+im2+im3+im4+[text1]+[text2]+[text3]+ \
            [lg1]+[lg2]+[lg3]+[lg4]+[imc]+im1r+[lg0r]

        ims.append(itot)
        
    #        exit()

# Non-Advection Phase
    l=0
    while l<lmax:
        hs_up=boundary.hs_up_cal(hs,hs_up,nx,hs_upstm,hs_dwstm,j_upstm,j_dwstm)
        cfx=cfxx.cfx_cal(cfx,nx,un,hs_up,g,snm)
        un=rhs.un_cal(un,u,nx,dx,cfx,hn,g,dt)
        un=boundary.u_bound(un,hs_up,qp,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
        qu=rhs.qu_cal(qu,un,hs_up,b_up,nx)
        hn,hs,err=hcal.hh(hn,h,hs,eta,qu,b,alh,hmin,dx,nx,dt,err)
        hn,hs=boundary.h_bound(hn,hs,eta,hs_upstm,hs_dwstm,nx,j_upstm,j_dwstm)
#        print(time,h[nx+1],hn[nx+1])


        if err<errmax:
            break
        l=l+1



#Differentials in Non Advection Phase
    gux=newgrd.ng_u(gux,u,un,nx,dx)
    gux=boundary.gbound_u(gux,nx)

# Advection Phase
    fn,gxn=mkzero.z0(fn,gxn,nx)
    fn,gxn=cip1d.u_cal1(un,gux,u,fn,gxn,nx,dx,dt)
    un,gux=cip1d.u_cal2(fn,gxn,u,un,gux,nx,dx,dt)
    un=boundary.u_bound(un,hs_up,qp,nx,j_upstm,j_dwstm,u_upstm,u_dwstm)
    gux=boundary.gbound_u(gux,nx)


# Update u and h
    h=copy.copy(hn); u=copy.copy(un)
    

#Time Step Update
    time=time+dt
    icount=icount+1

    


ani = animation.ArtistAnimation(fig, ims, interval=10)
#plt.show()
ani.save('htwidth.gif',writer='imagemagick')
#ani.save('cip.mp4',writer='ffmpeg')
#subprocess.call('ffmpeg -an -i cip.mp4 png\%04d.png')