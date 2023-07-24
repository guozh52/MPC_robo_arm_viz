from arm_dynamics_base import ArmDynamicsBase
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self, num_links, time_step):
		super().__init__()
		self.num_links = num_links
		self.time_step = time_step

		self.fc1 = nn.Linear(3 * self.num_links, 64)
		self.fc2 = nn.Linear(64, 64)
		self.fc3 = nn.Linear(64, 32)
		self.fc4 = nn.Linear(32, self.num_links)

	def forward(self, x):
			x = F.relu(self.fc1(x))
			x = F.relu(self.fc2(x))
			x = F.relu(self.fc3(x))
			x = self.fc4(x)
			return x	

	def compute_next_state(self, state, qddot):

		# Your code goes here
		new_q = np.zeros(self.num_links)
		new_qd = np.zeros(self.num_links)
		next_state = np.zeros(2 * self.num_links)
		for i in range(self.num_links):
				new_qd[i] = state[i + self.num_links] + qddot[i] * self.time_step
				new_q[i] = state[i] + state[i + self.num_links] * self.time_step + 0.5 * qddot[i] * self.time_step * self.time_step
		for i in range(self.num_links):
				next_state[i] = new_q[i]
				next_state[i + self.num_links] = new_qd[i + self.num_links]

		print("next_state:", next_state)
		return next_state

	def compute_qddot(self, x):
		pass

class Model2Link(Model):
		def __init__(self, time_step):
				super().__init__(2, time_step)
				# Your code goes here
				# self.fc1 = nn.Linear(3 * self.num_links, 64)
				# self.fc2 = nn.Linear(64, 32)
				# self.fc3 = nn.Linear(32, self.num_links)	

		def compute_qddot(self, x):
				# Your code goes here
				self.eval()	
				features = torch.from_numpy(x).float()
				qddot = self.forward(features).detach().numpy()
				return qddot
		
class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, time_step, device):

        self.model = Model2Link(0.01)
        self.model.load_state_dict(torch.load(model_path))
        # ---
        self.model_loaded = True


    def dynamics_step(self, state, action, dt):
        if self.model_loaded:

            self.model.eval() 
            input = np.append(state, action)
            new_acceleration = self.model.compute_qddot(input)
            new_q = np.zeros(2)
            new_qd = np.zeros(2)
            for i in range(2):
              new_qd[i] = state[i+2] + new_acceleration[i] * 0.01
              new_q[i] = state[i] + state[i+2] * 0.01 + 0.5 * new_acceleration[i] * 0.01 * 0.01
            new_state = np.reshape(np.append(new_q, new_qd), (4,1))
            return new_state
            # ---
        else:
            return state
