from cProfile import label
import sys, os
import math as m
from tkinter import font

import matplotlib
from torch import arcsin
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def cylinder_w(w, h, d, sita_y=73.5):
    temp = (1 + m.pi * w * h / d**2) * m.cos(m.radians(sita_y))
    if  -1 <= temp <= 1:
        sita_w = m.acos(temp)
        return [w, h, d, sita_w]
    else:
        return [w, h, d]

def cylinder_cb(w, d, sita_y=73.5):
    temp = m.pi * w**2 / (4 * d**2) * m.cos(m.radians(sita_y)) + m.pi * w**2 / (4 * d**2) - 1
    if -1 <= temp <= 1:
        sita_cb = m.acos(temp)
        return [w, d, sita_cb]
    else:
        return [w, d]

def cylinder_c(w, h, d):
    temp = (m.pi * w**2 / (4 * d**2) - 1) / (1 + m.pi * w * h / d**2 - m.pi * w**2 / (4 * d**2))
    if  -1 <= temp <= 1:
        sita_c = m.acos(temp)
        return [w, h, d, sita_c]
    else:
        return [w, h, d]

def truncated_cone_w(w, h, d, sita_y=73.5, k=0.5):
    temp = 1 + (m.pi/4 *(k**2 * w**2 - w**2) + m.pi/2 * (k*w + w) * m.sqrt(((k**2 * w**2 - w**2)/2)**2 + h**2)) / d**2
    temp = temp * m.cos(m.radians(sita_y))
    if  -1 <= temp <= 1:
        sita_w = m.acos(temp)
        return [w, h, d, sita_w]
    else:
        return [w, h, d]


def truncated_cone_cb(w, d, sita_y=73.5, k=0.5):
    temp = m.pi * k**2 * w**2 / (4 * d**2) * m.cos(m.radians(sita_y)) + m.pi * k**2 * w**2 / (4 * d**2) - 1
    if -1 <= temp <= 1:
        sita_cb = m.acos(temp)
        return [w, d, sita_cb]
    else:
        return [w, d]


def truncated_cone_c(w, h, d, k=0.5):
    temp1 = m.pi * k**2 * w**2 / (4 * d**2) - 1
    temp2 = 1 + m.pi/4 * (k**2 * w**2 - w**2) + m.pi/2 * (k*w + w) * m.sqrt(((k**2 * w**2 - w**2)/2)**2 + h**2) / d**2 - m.pi * k**2 * w**2 / (4 * d**2)

    temp = temp1 / temp2
    if  -1 <= temp <= 1:
        sita_c = m.acos(temp)
        return [w, h, d, sita_c]
    else:
        return [w, h, d]


def hemisphere_w(w, d, sita_y=73.5):
    temp = 4 / (3 + d**2 / w**2 - 2*d/w) * m.cos(m.radians(sita_y))
    if  -1 <= temp <= 1:
        sita_w = m.acos(temp)
        return [w, d, sita_w]
    else:
        return [w, d]


def hemisphere_cb(w, d, sita_y=73.5):
    temp1 = 2 * m.pi * (1 - m.cos(m.radians(sita_y))) / (2*m.pi * (1 - m.cos(m.radians(sita_y))) + 4*d**2/w**2 - m.pi * m.sin(m.radians(sita_y))**2) * m.cos(m.radians(sita_y))
    temp2 = 2 * m.pi * (1 - m.cos(m.radians(sita_y))) / (2*m.pi * (1 - m.cos(m.radians(sita_y))) + 4*d**2/w**2 - m.pi * m.sin(m.radians(sita_y))**2)

    temp = temp1 + temp2 - 1
    if  -1 <= temp <= 1:
        sita_cb = m.acos(temp)
        return [w, d, sita_cb]
    else:
        return [w, d]

def hemisphere_c(w, d, sita_y=73.5):
    temp = 2 * m.pi * (1 - m.cos(m.radians(sita_y))) / (2*m.pi * (1 - m.cos(m.radians(sita_y))) + 4*d**2/w**2 - m.pi * m.sin(m.radians(sita_y))**2)

    temp1 = temp - 1
    temp2 = 4 / (3 + d**2 / w**2 - 2*d / w) - temp

    if temp2 != 0:
        temp = temp1 / temp2
        if  -1 <= temp <= 1:
            sita_c = m.acos(temp)
            return [w, d, sita_c]
        else:
            return [w, d]
    else:
        return [w, d]


def spike_w(w, h, d, sita_y=73.5):
    temp = (m.pi / 6 * m.sqrt(1 + (m.pi*h/w)**2) + 1 - m.pi/6 * d**2 / (w + d/2)**2) * m.cos(m.radians(sita_y))
    if  -1 <= temp <= 1:
        sita_w = m.acos(temp)
        return [w, h, d, sita_w]
    else:
        return [w, h, d]

def spike_cb(w, h, d, sita_y=73.5):

    a = 2 * m.sin(m.radians(sita_y)) * w / m.pi * h
    if -1<= a <=1:
        temp1 = 3 * m.pi * (4 - m.pi * (m.asin(2 * m.sin(m.radians(sita_y)) * w / m.pi * h) / m.pi)**2)
        b = 2 * m.sin(m.radians(sita_y))
        if -1<= b <=1:
            temp2 = m.asin(2 * m.sin(m.radians(sita_y)) * w / m.pi * h)
            temp2 = 2 * temp2**2
            a = m.pi**2 * h**2 - (m.sin(m.radians(sita_y)))**2 * d**2
            b = 1 + 2*m.pi*h / d**2 * (m.pi*h - a)
            if a >=0 and b>=0:
                temp3 = 1 + m.sqrt(1 + 2 * m.pi * h / d**2 * (m.pi * h - m.sqrt(m.pi**2 * h**2 - (m.sin(m.radians(sita_y)))**2 * d**2)))

                temp = 1 + temp1 / ( temp2 * temp3)
                temp = m.cos(m.radians(sita_y)) / temp + 1/ temp - 1

                if  -1 <= temp <= 1:
                    sita_cb = m.acos(temp)
                    return [w, h, d, sita_cb]
                else:
                    return [w, h, d]
            else:
                return [w, h, d]
        else: 
            return [w, h, d]
    else:
        return [w, h, d]


def spike_c(w, h, d, sita_y=73.5):
    temp1 = 3 * m.pi * (4 - m.pi * (m.asin(2 * m.sin(m.radians(sita_y)) * w / (m.pi * h)) / m.pi)**2)
    temp2 = m.asin(2 * m.sin(m.radians(sita_y)) * w / m.pi * h)
    temp2 = 2 * temp2**2
    temp3 = 1 + m.sqrt(1 + 2 * m.pi * h / d**2 * (m.pi * h - m.sqrt(m.pi**2 * h**2 - m.sin(m.radians(sita_y)**2) * d**2)))

    temp = 1 + temp1 / (temp2 * temp3)

    temp = (1/temp - 1) / (m.pi/6 * m.sqrt(1 + (m.pi * h / w)**2) + 1 - m.pi/6 * d**2/(w + d/2)**2 - 1/temp)

    if  -1 <= temp <= 1:
        sita_c = m.acos(temp)
        return [w, h, d, sita_c]
    else:
        return [w, h, d]

def para_curve_w(w, h, d, sita_y=73.5):
    a = m.pi * (w+d)**2 / (864 * h**2)
    b = (1 + (12*h/(w+d))**2)**(3.0/2) - 1
    temp = (a * b + 1 - m.pi/4) * m.cos(m.radians(sita_y))

    if -1 <= temp <= 1:
        sita_w = m.acos(temp)
        return [w, h, d, sita_w]
    else:
        return [w, h, d]

def para_curve_cb(w, h, d, sita_y=73.5):
    a = (12*h/(w+d)) ** 2 - 3/4 * m.pi
    b = 1 + 6/(7*m.pi) * a
    temp = 1/b * m.cos(m.radians(sita_y)) + 1/b - 1

    if -1 <= temp <= 1:
        sita_cb = m.acos(temp)
        return [w, h, d, sita_cb]
    else:
        return [w, h, d]

def para_curve_c(w, h, d, sita_y=73.5):
    a = (12*h/(w+d)) ** 2 - 3/4 * m.pi
    b = 1 + 6/(7*m.pi) * a
    c = m.pi * (w+d)**2 / (864 * h**2)
    e = (1 + (12*h/(w+d))**2)**(3.0/2) - 1
    temp = (1/b - 1) / (c*e + 1 - m.pi/4 - 1/b)
    if -1 <= temp <= 1:
        sita_c = m.acos(temp)
        return [w, h, d, sita_c]
    else:
        return [w, h, d]



r = np.arange(1, 100.05, 1)
X, Y, Z = np.meshgrid(r, r, r)
ws, hs, ds, sita_ws = [], [], [], []

for x in r:
    for y in r:
        for z in r:
            result = cylinder_c(x, y, z)   # Function
            if len(result) == 4:
                ws.append(result[0])
                hs.append(result[1])
                ds.append(result[2])
                sita_ws.append(m.degrees(result[3]))
            
# Plot the surface

config = { "font.family":'Arial', }
rcParams.update(config)

fig = plt.figure(figsize=(16, 10), dpi=600)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ws, hs, ds = np.array(ws, dtype=float),  np.array(hs, dtype=float),  np.array(ds, dtype=float)
sita_ws = np.array(sita_ws, dtype=float)
v_mim, v_max = round(np.min(sita_ws), 1), round(np.max(sita_ws), 1)
print("v_mim:{}, v_max:{}".format(v_mim, v_max))
#v_ticks = np.arange(0, 180, 30)
norm = matplotlib.colors.Normalize(vmin=0, vmax=180)
v_ticks = [0, 60, 120, 180]
cm = plt.cm.get_cmap('jet')
print(ds)
fig = ax.scatter3D(ws, hs, ds, c=sita_ws, cmap=cm, norm=norm)

plt.xlim(0, 100.05)
plt.ylim(0, 100.05)
ax.set_zlim(0, 100.05)

plt.tick_params(labelsize=18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

ax.set_xlabel(r"$\it{w}$ (nm)", fontsize=18, labelpad=8.0)
ax.set_ylabel(r"$\it{h}$ (nm)", fontsize=18, labelpad=8.0)
ax.set_zlabel(r"$\it{d}$ (nm)", fontsize=18, labelpad=8.0)

cb = plt.colorbar(fig, pad= 0.005, shrink=0.3, aspect=20, location='left', ticks=v_ticks)
cb.ax.set_title(r'$\theta_{C}$',fontsize= 18,pad=8.0)
cb.ax.tick_params(labelsize=18)
cb.outline.set_visible(False)
# print(WS.shape)
# ax.plot_surface(ws, hs, ds, facecolor=cm.Blues(sita_ws))
plt.savefig("para_curve_cb_test.tiff", format='tiff', dpi=600, bbox_inches='tight')
plt.show()

"""r = np.arange(1, 100.05, 1)
# X, Y, Z = np.meshgrid(r, r, r)
ws, ds, sita_ws = [], [], []

for x in r:
    for y in r:
        result = hemisphere_w(x, y)   # Function
        if len(result) == 3:
            ws.append(result[0])
            ds.append(result[1])
            sita_ws.append(m.degrees(result[2]))
            
# Plot the surface

config = { "font.family":'Arial',}
rcParams.update(config)

fig = plt.figure(figsize=(16, 10), dpi=600)

fig, ax = plt.subplots()
ws, ds = np.array(ws, dtype=float), np.array(ds, dtype=float)
sita_ws = np.array(sita_ws, dtype=float)
v_mim, v_max = round(np.min(sita_ws), 1), round(np.max(sita_ws), 1)
print([v_mim, v_max])
#v_ticks = np.arange(0, 180.5, 30)
norm = matplotlib.colors.Normalize(vmin=0, vmax=180)
v_ticks = [0, 60, 120, 180]
print(v_ticks)
cm = plt.cm.get_cmap('jet')
fig = ax.scatter(ws, ds, c=sita_ws, cmap=cm, norm=norm)

plt.xlim(0, 100.05)
plt.ylim(0, 100.05)

plt.tick_params(labelsize=18)

plt.xticks([0,50,100],fontsize=18)
plt.yticks([0,50,100],fontsize=18)

ax.tick_params(axis='y', pad=10)
ax.yaxis.labelpad = 10

# plt.zticks(fontsize=12)
ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")

ax.set_xlabel(r"$\it{w}$ (nm)", fontsize=18)
ax.set_ylabel(r"$\it{d}$ (nm)", fontsize=18)
#ax.set_zlabel("D (nm)", fontsize=12)

cb = plt.colorbar(fig, pad= 0.08, shrink=0.3, aspect=20, location='left', ticks=v_ticks)
cb.ax.set_title(r'$\theta_{W}$' ,fontsize= 18,pad=8.0)
cb.ax.tick_params(labelsize=18)
cb.outline.set_visible(False)
# print(WS.shape)
# ax.plot_surface(ws, hs, ds, facecolor=cm.Blues(sita_ws))
plt.savefig("hemisphere_cb.tiff", format='tiff', dpi=600, bbox_inches='tight')
plt.show()"""