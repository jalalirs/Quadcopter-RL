import numpy as np
from physics_sim import PhysicsSim
import math
import random

distance = lambda a,b: np.linalg.norm(np.array(a)-np.array(b))
dot = lambda v1,v2: sum((a*b) for a, b in zip(v1, v2))
length = lambda v: math.sqrt(dot(v, v))
normalize = lambda a: math.atan2(math.sin(a), math.cos(a))
gaussian = lambda a: np.exp(-a**2)
steepen = lambda a,degree: (gaussian(normalize(a) * degree) * 2) - 1

#angle_btw = lambda v1,v2: math.acos(dot(v1, v2) / (length(v1) * length(v2)))
def angle_btw(v1,v2):
    v1u = v1 / np.linalg.norm(v1)
    v2u = v2 / np.linalg.norm(v1)
    return np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0))

transform = lambda a: 1 - 2*a/math.pi

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=100, target_pos=None,reward="simple"):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """

        RewardFunctions = {
            "simple": self.reward_simple,
            "angle": self.reward_angle,
            "distance_angle": self.reward_distnace_angle
        }

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high    = 900
        self.action_size = 4
        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        
        self._dtt = distance(self.target_pos,self.sim.pose[:3])
        self._rf = RewardFunctions[reward]
        self._att = angle_btw(self.target_pos - self.sim.pose[:3],self.sim.v)


    def reward_simple(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward
    
    def reward_angle(self):
        """Uses angle between target and current velocity vector"""
        ttv = self.target_pos - self.sim.pose[:3]
        angle = angle_btw(ttv,self.sim.v)
        diff = (self._att - angle)/self._att
        reward =  np.clip(transform(angle)+diff,-1,1)
        # if( (1.0 < reward) or (reward < -1.0) ):
        #     print(reward,self._att,angle)
        #     print(x, 'is out of scope from -1 to 1')
        #     assert True
        return reward
    def reward_distnace_angle(self):
        """Uses average of distance between target and current position and reward_angle"""
        #ra = self.reward_angle()
        props = np.array([0.25,0.25,0.25,0.25])#*4#np.random.dirichlet(np.ones(4))

        ttv = self.target_pos - self.sim.pose[:3]
        angle = angle_btw(ttv,self.sim.v)
        reward_a = steepen(angle,7)#transform(angle) #+ (angle - self._att)/self._att


        dtt = distance(self.target_pos,self.sim.pose[:3])
        if dtt < 20:
             reward_d = dtt/20.0
        else:
             reward_d = 1 if self._dtt > dtt else  -1

        reward_r = steepen(self.sim.pose[3],7)
        reward_p = steepen(self.sim.pose[4],7)
        return sum(props*np.array([reward_d,reward_a,reward_r,reward_p]))

    def get_reward(self):
        return self._rf()
        

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            if done and self.sim.time < self.sim.runtime:
                reward += -1
            else:
                reward += self.get_reward()

            pose_all.append(self.sim.pose)
            self._dtt = distance(self.target_pos,self.sim.pose[:3])
            ttv = self.target_pos - self.sim.pose[:3]
            self._att = angle_btw(ttv,self.sim.v)
        
        next_state = np.concatenate(pose_all)
        if self._dtt < 10:
            done = True
            return next_state, 10, True
        return next_state, np.clip(reward,-1,1), done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state