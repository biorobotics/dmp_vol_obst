#!/usr/bin/env python3.6
"""
Ginesi et al. 2019, fig 3a
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn
from dmp_vol_obst.srv import dmpPath,dmpPathResponse
import rospy
from dmp import dmp_cartesian
from dmp import dmp_quaternion
from dmp import obstacle_superquadric
import copy
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
#from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.msg import PlanningScene
def cuboid_data(o, size=(1,1,1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  
         [o[1], o[1], o[1], o[1], o[1]],          
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]   
    z = [[o[2], o[2], o[2], o[2], o[2]],                       
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],   
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],               
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]               
    return np.array(x), np.array(y), np.array(z)

def plotCubeAt(pos=(0,0,0), size=(1,1,1), ax=None,**kwargs):
    # Plotting a cube element at position pos
    if ax !=None:
        X, Y, Z = cuboid_data( pos, size )
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1,**kwargs)


#TODO: make it easier to input primitives
def genPrimitive():
	n=100
	primitive=np.zeros((n,3))
	for i in range(n):
		#quat=[1,0,0,0]
		#r=0.01
		r=0.3
		#primitive[i,:]=np.array([-r*i/100.,
		#						-r*i/100.,
		#						0.6+0.1*np.sin(np.pi*i/100.)])
		primitive[i,:]=np.array([0.,0.,0.6])

	return primitive
def genPrimitiveQ():
	n=100
	#target_R=R.from_euler('zyx',[0,0,1],degrees=True)
	#Rs=R.from_quat([np.array([0,0,0,1]),target_R.as_quat()])
	#Rpath_gen=Slerp(np.array([0.0,1.0]),Rs)
	#times=np.linspace(0,1.0,n)
	#Rpath=Rpath_gen(times)
	#qpath=Rpath.as_quat()
	#primitive=np.block([qpath[:,3:4],qpath[:,0:3]])
	primitive=np.zeros((n,4))
	for i in range(n):
		rx=(np.pi/200)*i/(50.)
		if i>=50:
			rx=(np.pi/200)*(99-i)/(50.)
		Ri=R.from_euler('zyx',[0,0,rx],degrees=False)
		qi=Ri.as_quat()
		primitive[i,0]=qi[3]
		primitive[i,1:4]=qi[0:3]
		#print(primitive[i])
		print(rx)
	return primitive
	 
#defined 
class box_obstacle:
	def __init__(self,pos,dim,coeffs,name="",A=200,eta=1.0):
		self.id=name
		self.dim=copy.deepcopy(dim)

		self.pos=copy.deepcopy(pos)
		self.coeffs=copy.deepcopy(coeffs)
		self.obs=obstacle_superquadric.Obstacle_Static(pos,dim,coeffs,A,eta)
	def F_sq(self,x,v=np.zeros(3)):
		return self.obs.gen_external_force(x)

#defines a class for running a ros service that calculates the dmp rollout
#given a start and goal.
class dmp_server:
	def __init__(self,primitive,qprimitive,num_dmps=3,num_bfs=40,K_val=1000.0,dtime=0.01,a_s=3.0,tolerance=2e-02):
		#self.can_update=True
		self.num_line_points=50
		#primitives for modeling hte desired trajectory
		self.primitive=primitive
		self.qprimitive=qprimitive
		#cartesian DMP for xyz position
		self.dmp=dmp_cartesian.DMPs_cartesian(n_dmps=num_dmps,n_bfs=num_bfs,K=K_val,dt=dtime,alpha_s=a_s,tol=tolerance)	
		self.dmp.imitate_path(x_des=self.primitive)
		#quaternion DMP
		self.qdmp=dmp_quaternion.DMPs_quaternion(n_bfs=num_bfs+500,K=48,dt=dtime,alpha_s=a_s,tol=tolerance)
		self.qdmp.imitate_path(q_des=self.qprimitive)
		#list to hold all obstacles
		self.obstacles=[]
		#TODO: delete this plot. plot the primitive
		#x_track,_,_,_=self.dmp.rollout()
		#fig=plt.figure()
		#ax=fig.gca(projection='3d')
		#ax.plot(x_track[:,0],x_track[:,1],x_track[:,2])
		#ax.set_title("dmp primitive")
		#plt.show()	
		self.base_positions=self.get_base_positions()
		self.dmp.reset_state()

	def get_base_positions(self):
		base_l=6.5*0.0254
		base_rotz_1=2*np.pi/3
		base_rotz_3=4*np.pi/3
		#get the pivot joint positions
		base_pos_1=np.array([-np.sin(base_rotz_1)*base_l,
							np.cos(base_rotz_1)*base_l,
							0.0])
		base_pos_2=np.array([0.0,base_l,0.0])
		base_pos_3=np.array([-np.sin(base_rotz_3)*base_l,
							np.cos(base_rotz_3)*base_l,
							0.0])
		return np.array([base_pos_1,base_pos_2,base_pos_3])

	#updates the collision objects, to be later used in the dmp rollout
	def update_collision_objects(self,scene):
		print("updating collision objects\n")
		#while(not self.can_update):
		#	continue
		#signal that we cant update the stored collision objects right now
		#self.can_update=False
		#replace stored collision objects
		if not scene.is_diff:
			self.obstacles=[]
		for obj in scene.world.collision_objects:
			print("adding object: "+obj.id)
			#assume every obstacle is a box. TODO: replace with
			#something that fits a superquadric function for given shape
			#TODO: dont hardcode this to mug
			dim=np.array([0.045,0.044,0.055])*1.3
			#pos=np.array([obj.mesh_poses[0].position.x,
			#			obj.mesh_poses[0].position.y,
			#			obj.mesh_poses[0].position.z])
			
			dim=np.array([obj.primitives[0].dimensions[0],
						  obj.primitives[0].dimensions[1],
						  obj.primitives[0].dimensions[2]])
			pos=np.array([obj.primitive_poses[0].position.x,
						  obj.primitive_poses[0].position.y,
						  obj.primitive_poses[0].position.z])
			
			#TODO: don't hardcode this and instead find a way to determine
			#the actual pose from the message
			posMat=np.array([[1.0,0.0,0.0,pos[0]],
							[0.0,1.0,0.0,pos[1]],
							[0.0,0.0,1.0,pos[2]-1.0],
							[0,0,0,1]])
			TMat=np.array([[-1.,0.0,0.0,0.0],
							[0.0,1.0,0.0,-0.03179],
							[0.0,0.0,-1.0,0.0],
							[0,0,0,1]])
			#R=np.array([[-1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-1.0]])
			#T=np.array([[0.0],[-0.03179],[0.0]])
			mpos=(TMat@posMat)[0:3,3]
			mpos.flatten()
			print("mpos",mpos)
			print("dim",dim)
			coeffs=np.array([2.,2.,2.])
			#coeffs=np.array([2./0.478,2./0.478,2/0.444])
			self.obstacles.append(box_obstacle(mpos,dim,coeffs,obj.id))
			#estimate the superquadric function parameters based on this
		#TODO: don't hardcode this and instead find a way to determine
		#the actual pose from the message
		#remove restriction on updating objects
		#self.can_update=True
		print("num obstacles is: ",len(self.obstacles))
		return None

	#calculates position of the clevis joints
	def calc_clevis_pos(self,ee_pose):
		#note: assumes ee_pose is in the format [x,y,z,w,rx,ry,rz],
		#aka position and orientation in a quaternion format
		#convert quaternion to rotaiton matrix
		ee_r=R.from_quat(ee_pose[3:7])
		ee_rmat=ee_r.as_matrix()
		#positions of each clevis joint relative to the ee
		clevis_rel_rz1=2*np.pi/3
		clevis_rel_rz3=4*np.pi/3
		clevis_rel_tz=0.75*0.0254/2-0.02928/np.sqrt(2.0)
		clevis_rel_ty=0.02111+0.02928/np.sqrt(2.0)
		clevis_1_rel_pos=np.array([-np.sin(clevis_rel_rz1)*clevis_rel_ty,
									np.cos(clevis_rel_rz1)*clevis_rel_ty,
									clevis_rel_tz])
		clevis_2_rel_pos=np.array([0.0,clevis_rel_ty,clevis_rel_tz])
		clevis_3_rel_pos=np.array([-np.sin(clevis_rel_rz3)*clevis_rel_ty,
									np.cos(clevis_rel_rz3)*clevis_rel_ty,
									clevis_rel_tz])
		#positions of clevis joints relative to base frame
		clevis_1_pos=ee_rmat@clevis_1_rel_pos+ee_pose[0:3]
		clevis_2_pos=ee_rmat@clevis_2_rel_pos+ee_pose[0:3]
		clevis_3_pos=ee_rmat@clevis_3_rel_pos+ee_pose[0:3]
		return np.array([clevis_1_pos,clevis_2_pos,clevis_3_pos])
	#calculates points located on the arms
	def calc_arm_points(self,ee_pose):
		#get the clevis joint positions
		clevis_positions=self.calc_clevis_pos(ee_pose)
		#get intermediate points between clevis and pivot joint
		points=np.zeros((3,self.num_line_points,3))
		for i in range(3):
			base_i=self.base_positions[i]
			clevis_i=clevis_positions[i]
			#interpolate between clevis and pivot pos
			larr=np.linspace(clevis_i[0],base_i[0],self.num_line_points)
			points[i,:,0]=np.linspace(clevis_i[0],base_i[0],self.num_line_points)
			points[i,:,1]=np.linspace(clevis_i[1],base_i[1],self.num_line_points)
			points[i,:,2]=np.linspace(clevis_i[2],base_i[2],self.num_line_points)
		
		return points

	def intermediate_point_force(self,pose,obst_i):
		#return np.zeros(3)
		points=self.calc_arm_points(pose)
		best_F=np.zeros((3,3))
		best_F_mag=[0.0,0.0,0.0]
		best_arm=[0,0,0]
		best_point=np.zeros((3,3))
		#iterate over each arm
		for arm in range(3):
			#iterate over points w/in an arm
			for point_i in range(self.num_line_points):
				point=points[arm,point_i,:]
				point_F=self.obstacles[obst_i].F_sq(point)
				point_F_mag=np.linalg.norm(point_F)
				if (best_F_mag[arm]<point_F_mag):
					best_F[arm,:]=point_F
					best_F_mag[arm]=point_F_mag
					best_arm[arm]=arm
					best_point[arm,:]=point
		#scale the highest force so that it proportionally 
		#pushes the ee
		#TODO: find better approximation
		dbase=np.repeat([pose[0:3]],3,axis=0)-self.base_positions
		ee_dists=np.linalg.norm(dbase,axis=1)
		p_dists=np.linalg.norm(best_point-self.base_positions,axis=1)
		#sum all 3 arm's forces together
		return np.sum(best_F*(ee_dists/p_dists))
		#return best_F*(ee_dists/p_dists)
		#ee_dist=np.linalg.norm(pose[0:3]-self.base_positions[best_arm])
		#p_dist=np.linalg.norm(best_point-self.base_positions[best_arm])
		#print("ee pos:",pose[0:3])
		#print("point :",best_point)
		#print("scaled F:",best_F*ee_dist/p_dist)
		#return best_F*ee_dist/p_dist

		#add ee force and highest arm force to the sum
	#calculates static obstacle forces. TODO: try dynamic(incorporate v?)
	def obs_forces_fxn(self,dynamic=False):
		if len(self.obstacles)==0:
			return None
		if dynamic:
			def f(x,v): 
				#for not just add a the default quaternion. incorporate
				#the rotational quaternion stuff later
				pose=np.append(x,[1,0,0,0])
				sum_F=self.obstacles[0].F_sq(x,v)+self.intermediate_point_force(pose,0)
				#sum_F=self.intermediate_point_force(pose,0)
				i=1
				while i<len(self.obstacles):
					#add ee force
					#sum_F=sum_F+self.obstacles[i].F_sq(x,v)
					sum_F=sum_F+self.intermediate_point_force(pose,i)
					i=i+1
					#calculate forces for points along arm
				return sum_F
				
			return sum_F
		def f(x,v):
			pose=np.append(x,[1,0,0,0])
			#sum_F=self.obstacles[0].F_sq(x,v)#+self.intermediate_point_force(pose,0)
			sum_F=self.intermediate_point_force(pose,0)
			
			i=1
			while i<len(self.obstacles):
				#sum_F=sum_F+self.obstacles[i].F_sq(x,v)
				sum_F=sum_F+self.intermediate_point_force(pose,i)
				i=i+1
				#calculate forces for points along arm
				#calculate the largest force and scale it so that
				#it proportionally accelerates the ee
			return sum_F
		return f


	#calculates the dmp rollout, given a start and end. requires that
	#the dmp be trained beforehand
	def handle_dmp_path(self,req):
		#store obstacles. TODO: implement
		'''
		self.obstacles=[]
		for i in range(len(req.obstacles)):
			#mpos: obstacle position
			mpos=np.array([req.obstacles[i].pos_x,
							req.obstacles[i].pos_y,
							req.obstacles[i].pos_z])
			#angle: angle of obstacle. TODO: implement this
			angle=np.array([req.obstacles[i].angle])
			#dim: obstacle dimensions
			dim=np.array([req.obstacles[i].dim_x,
							req.obstacles[i].dim_y,
							req.obstacles[i].dim_z])
			#coeffs: obstacle superquadric coeffs
			coeffs=np.array([req.obstacles[i].e1,
							 req.obstacles[i].e2])
			#name: optional name. TODO: implement this
			self.obstacles.append(box_obstacle(mpos,dim,coeffs,""))
		'''
		print("handling dmp path\n")
		self.dmp.reset_state()
		self.qdmp.reset_state()
		print("num obstacles is: ",len(self.obstacles))
		start=np.array([req.start.position.x,
						req.start.position.y,
						req.start.position.z,
						req.start.orientation.w,
						req.start.orientation.x,
						req.start.orientation.y,
						req.start.orientation.z])
		goal=np.array([req.goal.position.x,
						req.goal.position.y,
						req.goal.position.z,
						req.goal.orientation.w,
						req.goal.orientation.x,
						req.goal.orientation.y,
						req.goal.orientation.z])
		print("got start and goal")
		print("start=",start)
		print("goal=",goal)
		#set dmp start and goal states
		self.dmp.x_0=copy.deepcopy(start[0:3])
		self.dmp.x_goal=copy.deepcopy(goal[0:3])
		self.qdmp.q_0=copy.deepcopy(start[3:7])
		self.qdmp.q_goal=copy.deepcopy(goal[3:7])
		#calculate the path
		flag=False
		qflag=False
		x_track_s=copy.deepcopy(start[0:3])
		dx_track_s=np.zeros(self.dmp.n_dmps)
		ddx_track_s=np.zeros(self.dmp.n_dmps)
		q_track_s=copy.deepcopy(start[3:7])
		eta_track_s=np.zeros(3)
		deta_track_s=np.zeros(3)
		path=np.array([start[0:3]])
		dx_track=np.zeros((1,self.dmp.n_dmps))
		ddx_track=np.zeros((1,self.dmp.n_dmps))
		qpath=np.array([start[3:7]])
		eta_track=np.zeros((1,3))
		deta_track=np.zeros((1,3))
		self.dmp.t=0
		self.qdmp.t=0
		hasObstacles=(len(self.obstacles)>0)
		print("beginning to calculate path")
		while(not (flag and qflag)):
			forces_fxn=self.obs_forces_fxn()	
			if hasObstacles:
				x_track_s,dx_track_s,ddx_track_s=self.dmp.step(
					external_force=forces_fxn,adapt=False)
			else:
				x_track_s,dx_track_s,ddx_track_s=self.dmp.step(
					external_force=None,adapt=False)
			#if forces_fxn!=None:
			#	f=forces_fxn(x_track_s,x_track_s)
			#simulate quaternion DMP with no forces. TODO: add forces
			q_track_s,eta_track_s,deta_track_s=self.qdmp.step(
					external_force=None,adapt=False,pos=copy.deepcopy(x_track_s))
			
			path=np.append(path,[x_track_s],axis=0)
			dx_track=np.append(dx_track,[dx_track_s],axis=0)
			ddx_track=np.append(ddx_track,[ddx_track_s],axis=0)
			self.dmp.t+=1
			flag=(np.linalg.norm(x_track_s-self.dmp.x_goal)/np.linalg.norm(self.dmp.x_goal-self.dmp.x_0)<=self.dmp.tol)

			qpath=np.append(qpath,[q_track_s],axis=0)
			eta_track=np.append(eta_track,[eta_track_s],axis=0)
			deta_track=np.append(deta_track,[deta_track_s],axis=0)
			self.qdmp.t+=1
			qflag=(np.linalg.norm(q_track_s-self.qdmp.q_goal)<=self.qdmp.tol)
			#print("cartesian flag:",flag,", quaternion flag:",qflag)
			#print("cartesian state:",x_track_s,", goal:",self.dmp.x_goal)
			#print("quaternion state:",q_track_s,", goal:",self.qdmp.q_goal)

		"""
		self.dmp.reset_state()
		pathOrig,_,_,_=self.dmp.rollout()
		self.dmp.reset_state()
		#grab the start/goal positions
		start=np.array([req.start.position.x,
						req.start.position.y,
						req.start.position.z])
		goal=np.array([req.goal.position.x,
						req.goal.position.y,
						req.goal.position.z])
		print("got start and goal")
		#set dmp start and goal states
		self.dmp.x_0=copy.deepcopy(start)
		self.dmp.x_goal=copy.deepcopy(goal)
		#calculate the path
		flag=False
		x_track_s=copy.deepcopy(start)
		dx_track_s=np.zeros(self.dmp.n_dmps)
		ddx_track_s=np.zeros(self.dmp.n_dmps)
		path2=np.array([start])
		dx_track=np.zeros((1,self.dmp.n_dmps))
		ddx_track=np.zeros((1,self.dmp.n_dmps))
		self.dmp.t=0
		hasObstacles=(len(self.obstacles)>0)
		print("beginning to calculate path")
		while(not flag):
			forces_fxn=None
			if hasObstacles:
				print("planning w.r.t obstacles")
				#temporarily print out the obstacles
				for obj in self.obstacles:
					print("object name:",obj.id,", dim= ",obj.dim)
				x_track_s,dx_track_s,ddx_track_s=self.dmp.step(
					external_force=forces_fxn,adapt=False)
			else:
				print("planning w.r.t no obstacles")
				x_track_s,dx_track_s,ddx_track_s=self.dmp.step(
					external_force=None,adapt=False)
			if forces_fxn!=None:
				f=forces_fxn(x_track_s,x_track_s)
				#print("f=",f)
			path2=np.append(path2,[x_track_s],axis=0)
			dx_track=np.append(dx_track,[dx_track_s],axis=0)
			ddx_track=np.append(ddx_track,[ddx_track_s],axis=0)
			self.dmp.t+=1
			flag=(np.linalg.norm(x_track_s-self.dmp.x_goal)/np.linalg.norm(self.dmp.x_goal-self.dmp.x_0)<=self.dmp.tol)
		"""
		"""
		fig=plt.figure(1)
		ax=fig.gca(projection='3d')
		prim=genPrimitive()
		ax.plot(prim[:,0],prim[:,1],prim[:,2],color="blue",label="Original Path")
		ax.plot(prim[-1,0],prim[-1,1],prim[-1,2],"o",color="blue",label="Original Goal")	
		ax.plot(path2[:,0],path2[:,1],path2[:,2],color="yellow",label="Modified Path")
		ax.plot(path2[-1,0],path2[-1,1],path2[-1,2],"o",color="yellow",label="Modified Goal")
		ax.plot(path[:,0],path[:,1],path[:,2],color="orange",label="Modified Path Avoiding Obstacles")
		ax.set_title("Paths generated from DMP")
		ax.set_xlabel("x(m)")
		ax.set_ylabel("y(m)")
		ax.set_zlabel("z(m)")
		plt.legend()
		if len(self.obstacles)>0:
			plotCubeAt(pos=self.obstacles[0].pos,
				size=self.obstacles[0].dim,ax=ax,color="crimson")
		plt.show()
		"""
		#convert the path to geometry_msgs/Pose array
		print("found path")#. path is:\n")
		#print(path)
		geomPath=[Pose() for i in range(path.shape[0])]
		for i in range(path.shape[0]):
			geomPath[i].position.x=path[i,0]
			geomPath[i].position.y=path[i,1]
			geomPath[i].position.z=path[i,2]
			#TODO: do for rotation as well
			geomPath[i].orientation.w=qpath[i,0]
			geomPath[i].orientation.x=qpath[i,1]
			geomPath[i].orientation.y=qpath[i,2]
			geomPath[i].orientation.z=qpath[i,3]
		#remove restriction on updating objects
		#self.can_update=True
		return dmpPathResponse(geomPath)

	#starts the ros service, defines the primitive to be used, and
	#finds the parameters for the dmp to follow that trajectory.
	def dmp_path_server(self):
		#generate the modeled primitive. TODO: make this more easily
		#accessible
		primitive=genPrimitive()
		#initialize the dmp and imitate the primitive
		dmp=dmp_cartesian.DMPs_cartesian(n_dmps=3,n_bfs=100,K=1000.0,dt=0.01,alpha_s=3.0,tol=2e-02)
		#dmp.imitate_path(x_des=primitive)
		#start the service
		rospy.init_node('dmp_path_server')
		s = rospy.Service('dmp_calc_path',dmpPath,self.handle_dmp_path)
		#start subscriber; checks if any objects have been updated
		rospy.Subscriber("/move_group/monitored_planning_scene",PlanningScene,self.update_collision_objects)
		print("Ready to calculate dmp.")
		rospy.spin()
if __name__=="__main__":
	primitive=genPrimitive()
	primitiveQ=genPrimitiveQ()
	d_server=dmp_server(primitive,primitiveQ)
	d_server.dmp_path_server()
