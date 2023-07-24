# Local Variables:
# python-indent: 2
# End:

import numpy as np
from robot import Robot

class MPC:

  def __init__(self):
    self.control_horizon = 10
    self.delta_u = 0.5
    self.calculate_times = 10   
    self.du_para = 2  

  def Roll_Out(self, dynamics, state, goal, action):
    arm = Robot(dynamics)
    arm.set_state(state)
    arm.set_action(action)
    
    for step_ in range(30):
      state = arm.get_state()
      arm.advance()
      
    new_state = arm.get_state()

    pos_ee = dynamics.compute_fk(new_state)
    dist = np.linalg.norm(goal-pos_ee)
    vel_ee = np.linalg.norm(dynamics.compute_vel_ee(new_state))

    return dist, vel_ee

  def Cost_Func(self, dist, vel_ee):
    cost = 100 * pow(dist,2) + 4 * pow(vel_ee,2)
    return cost


  def compute_action(self, dynamics, state, goal, action):
    # Put your code here. You must return an array of shape (num_links, 1)

    #Roll out and calculate the old cost
    dist, vel_ee = self.Roll_Out(dynamics, state, goal, action)
    cost_old = self.Cost_Func(dist, vel_ee)
    #print("cost", cost_old)
    
    #Generate new action
    for i in range(self.calculate_times):
      new_action_list = np.zeros((len(action), 2 * len(action)))
      for num_of_new_act in range(2 * len(action)):
        new_action_list[:, num_of_new_act] = action[:, 0]
        if cost_old < self.du_para:
          if num_of_new_act <= len(action) - 1:
            new_action_list[num_of_new_act, num_of_new_act] += self.delta_u*(1 - (i/self.calculate_times))*(cost_old/self.du_para)
          else:
            new_action_list[num_of_new_act - len(action), num_of_new_act] -= self.delta_u*(1 - (i/self.calculate_times))*(cost_old/self.du_para)
        else:
          if num_of_new_act <= len(action) - 1:
            new_action_list[num_of_new_act, num_of_new_act] += self.delta_u*(1 - (i/self.calculate_times))
          else:
            new_action_list[num_of_new_act - len(action), num_of_new_act] -= self.delta_u*(1 - (i/self.calculate_times))

      #Roll out and calculate the new cost function
      cost_new = np.zeros(new_action_list.shape[1])
      for i in range(new_action_list.shape[1]):
        dist, vel_ee = self.Roll_Out(dynamics, state, goal, new_action_list[:, i])
        cost_new[i] = self.Cost_Func(dist, vel_ee)
      best_action = np.zeros((new_action_list.shape[0], 1))
      if np.amin(cost_new) < cost_old:
        best_action = new_action_list[:, np.argmin(cost_new)].reshape(len(action),1)
        action = best_action
        cost_old = np.amin(cost_new)
  
    return best_action