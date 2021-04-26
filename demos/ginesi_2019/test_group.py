#!/usr/bin/env python3
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
from dmp import obstacle_superquadric
import copy
from geometry_msgs.msg import Pose
from moveit_msgs.msg import PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene,GetPlanningSceneResponse
def genPrimitive():
	n=100
	primitive=np.zeros((n,3))
	for i in range(n):
		#quat=[1,0,0,0]
		primitive[i,:]=np.array([0.3*i/100.,
								0.3*i/100.,
								0.6+0.1*np.sin(np.pi*i/100.)])
	return primitive
class dmp_server:
	def __init__(self,primitive,num_dmps=3,num_bfs=40,K_val=1000.0,dtime=0.01,a_s=3.0,tolerance=2e-02):
		self.primitive=primitive
		self.dmp=dmp_cartesian.DMPs_cartesian(n_dmps=num_dmps,n_bfs=num_bfs,K=K_val,dt=dtime,alpha_s=a_s,tol=tolerance)	
		self.dmp.imitate_path(x_des=self.primitive)
		#TODO: delete this plot. plot the primitive
		#x_track,_,_,_=self.dmp.rollout()
		#fig=plt.figure()
		#ax=fig.gca(projection='3d')
		#ax.plot(x_track[:,0],x_track[:,1],x_track[:,2])
		#ax.set_title("dmp primitive")
		#plt.show()	
		self.dmp.reset_state()
		
	def handle_dmp_path(self,req):
		#grab the obstacles
		print("attempting to grab obstacles:\n")
		get_planning_scene=rospy.ServiceProxy('/get_planning_scene',GetPlanningScene)
		request=PlanningSceneComponents(components=PlanningSceneComponents.WORLD_OBJECT_NAMES)
		response=get_planning_scene(request)
		print("got a response!\n")
		print("collision object name:"+response.world.collision_objects[0].id)

		self.dmp.reset_state()
		start=np.array([req.start.position.x,
						req.start.position.y,
						req.start.position.z])
		goal=np.array([req.goal.position.x,
						req.goal.position.y,
						req.goal.position.z])
		#set dmp start and goal states
		self.dmp.x_0=copy.deepcopy(start)
		self.dmp.x_goal=copy.deepcopy(goal)
		#use rollout to calculate the path
		path,_,_,_=self.dmp.rollout()
		#fig=plt.figure()
		#ax=fig.gca(projection='3d')
		#ax.plot(path[:,0],path[:,1],path[:,2])
		#ax.set_title("dmp planned path")
		#plt.show()
		#convert the path to geometry_msgs/Pose array
		geomPath=[Pose() for i in range(path.shape[0])]
		for i in range(path.shape[0]):
			geomPath[i].position.x=path[i,0]
			geomPath[i].position.y=path[i,1]
			geomPath[i].position.z=path[i,2]
			#TODO: do for rotation as well
			geomPath[i].orientation.w=1.0
			geomPath[i].orientation.x=0.0
			geomPath[i].orientation.y=0.0
			geomPath[i].orientation.z=0.0
		return dmpPathResponse(geomPath)
	def dmp_path_server(self):
		#generate the modeled primitive. TODO: make this more easily
		#accessible
		primitive=genPrimitive()
		#initialize the dmp and imitate the primitive
		dmp=dmp_cartesian.DMPs_cartesian(n_dmps=3,n_bfs=40,K=1000.0,dt=0.01,alpha_s=3.0,tol=2e-02)
		dmp.imitate_path(x_des=primitive)
		rospy.init_node('dmp_path_server')
		s = rospy.Service('dmp_calc_path',dmpPath,self.handle_dmp_path)
		print("Ready to calculate dmp.")
		rospy.spin()
if __name__=="__main__":
	print("attempting to grab obstacles:\n")
	get_planning_scene=rospy.ServiceProxy('/get_planning_scene',GetPlanningScene)
	request=PlanningSceneComponents(components=PlanningSceneComponents.WORLD_OBJECT_NAMES)
	response=get_planning_scene(request)
	print("got a response!\n")
	print("collision object name:"+response.scene.world.collision_objects[0].id)
