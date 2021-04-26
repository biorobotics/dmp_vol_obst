"""
Ginesi et al. 2019, fig 3a
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

import pdb

from dmp import dmp_cartesian
from dmp import obstacle_superquadric
from dmp import point_obstacle
from dmp import dmp_quaternion
"""
Here we create the trajectory to learn
"""

t_f = 1.0 * np.pi # final time
t_steps = 10 ** 3 # time steps
t = np.linspace(0, t_f, t_steps)

a_x = 1.0 / np.pi
b_x = 1.0
a_y = 1.0 / np.pi
b_y = 1.0

x = a_x * t * np.cos(b_x*t)
y = a_y * t * np.sin(b_y*t)

x_des = np.ndarray([t_steps, 2])
x_des[:, 0] = x
x_des[:, 1] = y

x_des -= x_des[0]

Rs=R.from_quat([[0,0,0,1],[0.5709415,0.16752,0.57094,0.56568]])
Rpath_gen=Slerp(np.array([0.0,t_f]),Rs)
Rpath=Rpath_gen(t)
qpath=Rpath.as_quat()
q_des=np.block([qpath[:,3:4],qpath[:,0:3]])
#TODO: merge with cartesian DMP


# Learning of the trajectory
dmp = dmp_cartesian.DMPs_cartesian(n_dmps = 2, n_bfs = 40, K = 1000.0, dt = 0.01, alpha_s = 3.0, tol = 2e-02)
#testing quaternion implementation
qdmp = dmp_quaternion.DMPs_quaternion(n_dmps = 1, n_bfs = 40, K = 1000.0, dt = 0.01, alpha_s = 3.0, tol = 2e-02)
dmp.imitate_path(x_des = x_des)
qdmp.imitate_path(q_des=q_des)
x_track, _, _, _ = dmp.rollout()
q_track,_,_,q_time=qdmp.rollout()

print("rolled out quaternions")
#convert to euler angles to approximate what it would look like
q_track_rearr=np.block([q_track[:,1:4],q_track[:,0:1]])
q_track_R=R.from_quat(q_track_rearr)
q_track_eul=q_track_R.as_euler("xyz",degrees=False)
plt.figure()
plt.plot(q_time,q_track_eul[:,0])
plt.plot(q_time,q_track_eul[:,1])
plt.plot(q_time,q_track_eul[:,2])
plt.show()
#testing the step functionality
qdmp.reset_state()
qflag=False
qdmp.t=0
q_track = np.zeros([1, 4])
q_track[0,0]=1.0
eta_track = np.zeros([1, 3])
deta_track = np.zeros([1, 3])
while not qflag:
	q_track_s, eta_track_s, deta_track_s = qdmp.step(external_force = None, adapt=False)
	q_track = np.append(q_track, [q_track_s], axis = 0)
	eta_track = np.append(eta_track, [eta_track_s],axis = 0)
	deta_track = np.append(deta_track, [deta_track_s],axis = 0)
	qdmp.t += 1
	qflag=(np.linalg.norm(q_track_s-qdmp.q_goal)<qdmp.tol)
	#qflag = (np.linalg.norm(q_track_s - qdmp.q_goal) / np.linalg.norm(qdmp.q_goal - qdmp.q_0) <= dmp.tol)
print("stepped out quaternions")
#convert to euler angles to approximate what it would look like
step_time=list(range(0,qdmp.t+1))
q_track_rearr=np.block([q_track[:,1:4],q_track[:,0:1]])
q_track_R=R.from_quat(q_track_rearr)
q_track_eul=q_track_R.as_euler("xyz",degrees=False)
plt.figure()
print()
plt.plot(step_time,q_track_eul[:,0])
plt.plot(step_time,q_track_eul[:,1])
plt.plot(step_time,q_track_eul[:,2])
plt.show()


x_classical = x_track
# Execution with the obstacles
dmp.reset_state()
x_track = np.zeros([1, dmp.n_dmps])
dx_track = np.zeros([1, dmp.n_dmps])
ddx_track = np.zeros([1, dmp.n_dmps])

dmp.dx_old = np.zeros(dmp.n_dmps)
dmp.ddx_old = np.zeros(dmp.n_dmps)
flag = False
dmp.t = 0

"""
Volumetric Obstacle
"""
x_track_s = x_track[0]
x_c_1 = - 0.5
y_c_1 = 0.7
n = 2
a_1 = 0.3
b_1 = 0.2
center_1 = np.array([x_c_1, y_c_1])
axis_1 = np.array([a_1, b_1])
A = 50.0
eta = 1.0
obst_volume_1 = obstacle_superquadric.Obstacle_Static(center = center_1, axis = axis_1, A = A, eta = eta)

def F_sq(x, v):
	return obst_volume_1.gen_external_force(x)

while (not flag):
	print(qdmp.q)
	x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force = F_sq, adapt=False)
	print(qdmp.q)
	x_track = np.append(x_track, [x_track_s], axis = 0)
	dx_track = np.append(dx_track, [dx_track_s],axis = 0)
	ddx_track = np.append(ddx_track, [ddx_track_s],axis = 0)
	dmp.t += 1
	flag = (np.linalg.norm(x_track_s - dmp.x_goal) / np.linalg.norm(dmp.x_goal - dmp.x_0) <= dmp.tol)
	fig = plt.figure(1)
	plt.clf()
plt.figure(1, figsize = (6,6))
plt.plot(x_classical[:,0], x_classical[:, 1], '--', color='blue', lw=2, label = 'without obstacle')
plt.plot(x_track[:,0], x_track[:,1], '-g', lw=2, label = 'static vol obst')

"""
Point cloud obstacle
"""

dmp.reset_state()
x_track = np.zeros((1, dmp.n_dmps))
dx_track = np.zeros((1, dmp.n_dmps))
ddx_track = np.zeros((1, dmp.n_dmps))

dmp.dx_old = np.zeros(dmp.n_dmps)
dmp.ddx_old = np.zeros(dmp.n_dmps)
flag = False
dmp.t = 0
dmp.tol = 5e-02
# Obstacle definition
num_obst_1 = 50
num_obst_2 = 50
t_1 = np.linspace(0.0, np.pi * 2.0, num_obst_1)
obst_list_1 = []
for n in range(num_obst_1):
	obst = point_obstacle.Obstacle_Steering(x = np.array([x_c_1 + a_1*np.cos(t_1[n]), y_c_1 + b_1*np.sin(t_1[n])]))
	obst_list_1.append(obst)

def F_sa(x, v):
	f = np.zeros(2)
	for _n in range(num_obst_1):
		f += obst_list_1[_n]. gen_external_force(x, v, dmp.x_goal)
	return f

while (not flag):
	# run and record timestep
	x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force = F_sa, adapt=False)
	x_track = np.append(x_track, [x_track_s], axis = 0)
	dx_track = np.append(dx_track, [dx_track_s],axis = 0)
	ddx_track = np.append(ddx_track, [ddx_track_s],axis = 0)
	dmp.t += 1
	flag = (np.linalg.norm(x_track_s - dmp.x_goal) / np.linalg.norm(dmp.x_goal - dmp.x_0) <= dmp.tol)
plt.plot(x_track[:, 0], x_track[:, 1], '-.', color='orange', lw = 2, label = 'Pastor (steering angle)')

# Obstacle plot
x_plot_1 = x_c_1 + a_1*np.cos(t_1)
y_plot_1 = y_c_1 + b_1 * np.sin(t_1)
plt.plot (x_plot_1, y_plot_1, '.r', lw = 2)
plt.xlabel(r'$x$',fontsize = 14)
plt.ylabel(r'$y$',fontsize = 14)
plt.axis('equal')
plt.legend(loc = 'best')
plt.show()
