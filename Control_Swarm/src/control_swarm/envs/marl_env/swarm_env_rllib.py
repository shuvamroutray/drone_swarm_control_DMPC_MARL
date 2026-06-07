import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces


class SwarmEnvRLlib(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()

        config = config or {}

        self.grid_size = config.get("grid_size", 6)
        self.n_agents = config.get("n_agents", 2)
        self.max_steps = config.get("max_steps", 100)

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        # -----------------------------
        # SINGLE AGENT SPACES (RLlib req)
        # -----------------------------
        obs_dim = self.grid_size * self.grid_size + 2 + self.n_agents

        """     
        format of the observation vector that each agent is allowed to receive..
        "Observation" is a 1D vector of size obs_dim. Also Box Box is Gymnasium’s way of defining a continuous numerical space.
        The low value of the observation space can be 0 and the high value can be 10 which is acceptable as grid values are 0,1,2
        and position is small integer values. It creates a tuple of size obs_dim which stores values of type float32
        """
        self.single_observation_space = spaces.Box(
            low=0,
            high=10,
            shape=(obs_dim,),
            dtype=np.float32
        )

        """
        Below line defines the action space one agent is allowed to take. 
        ACTIONS = {
            0: (0, 1),    # UP
            1: (0, -1),   # DOWN
            2: (-1, 0),   # LEFT
            3: (1, 0),    # RIGHT
            4: (0, 0),    # STAY
        }        
        """
        self.single_action_space = spaces.Discrete(5)



        """
        Below lines take the per-agent observation/action spaces we defined earlier and scale them to the entire swarm.
        
        Observation space:
        {
            "agent_0": Box(...),
            "agent_1": Box(...)
        }

        Action space:
        {
            "agent_0": Discrete(5),
            "agent_1": Discrete(5)
        }
        These are multi agent observations that go to RLlib. It expects it in dictionary format:
        
        {
            agent_id: observation
        }

        {
            agent_id: action
        }

        Real execution example

        agent_0 at (1,2)
        agent_1 at (4,5)

        RLlib sends:
        obs = {
            "agent_0": obs_vector_0,
            "agent_1": obs_vector_1
        }

        Policy outputs:
        actions = {
            "agent_0": 3,   # RIGHT
            "agent_1": 0    # UP
        }

        Environment step:
        env.step(actions)
        """

        # -----------------------------
        # MULTI AGENT SPACES
        # -----------------------------
        self.observation_space = {
            agent: self.single_observation_space
            for agent in self.agents
        }

        self.action_space = {
            agent: self.single_action_space
            for agent in self.agents
        }

        self.reset()  # initializes the environment immediately when the class is created.

    # -----------------------------
    # REQUIRED BY RLlib
    # -----------------------------

    """
    Basically used by RLlib to get individual agent action and observation space.
    """    
    def get_action_space(self, agent_id):
        return self.single_action_space

    def get_observation_space(self, agent_id):
        return self.single_observation_space
    


    
    # -----------------------------
    # RESET
    # -----------------------------

    """
    Episode Start:
    reset()

    Then:
    step() → step() → step() ...

    Episode Ends:
    reset()

    seed=NONE--> No fixed seed,Random generator continues naturally, New layouts each episode

    """   
    def reset(self, *, seed=None, options=None):
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Place anomaly
        self.anomaly_pos = (
            np.random.randint(self.grid_size),
            np.random.randint(self.grid_size)
        )
        self.grid[self.anomaly_pos] = 2

        # Initialize agent positions (NO OVERLAP)
        self.agent_pos = {}

        """
        Key Characteristics of set()

        Uniqueness: Sets automatically eliminate duplicate values.
        Unordered: Items have no defined index; they may appear in a different order each time the set is accessed.
        Mutable: You can add or remove items after the set is created.
        Hashable Elements: While the set itself is mutable, the individual elements must be immutable (e.g., strings, numbers, or tuples). You cannot put a list or another set inside a set.
        """
        used_positions = set() 

        for agent in self.agents:
            while True:
                pos = (
                    np.random.randint(self.grid_size),
                    np.random.randint(self.grid_size)
                )
                if pos not in used_positions:
                    used_positions.add(pos)
                    self.agent_pos[agent] = pos
                    break

        self.step_count = 0

        return self._get_obs(), {}

    # -----------------------------
    # OBSERVATION
    # -----------------------------

    """
    Internal/private helper function
    This function converts internal simulator state:grid + positions + agent identity 
    into
    Neural network input vector
    """

    def _get_obs(self):  #intenal fucntion to the class
        obs = {}

        for agent in self.agents:
            x, y = self.agent_pos[agent]
            flat_grid = self.grid.flatten()

            # Agent identity (break symmetry)
            """
            For agent identity to break symmetry.            
            """
            agent_id = int(agent.split("_")[1]) # "agent_0" → 0 "agent_1" → 1
            
            id_vec = np.zeros(self.n_agents)  # for 2 agents [0,0]
            id_vec[agent_id] = 1  # Agent 0: [1,0]. For Agent 1: [0 1]. One hot encoding identity vector


            obs_vec = np.concatenate([
                flat_grid,
                np.array([x, y]),
                id_vec
            ]) # Concatenate everythiing to create the obs vector for neural network.

            obs[agent] = obs_vec.astype(np.float32)

        return obs

    # -----------------------------
    # STEP
    # -----------------------------

    """
    Actions → Movement → Rewards → Penalties → Team reward → Episode termination
    Every training cycle:
    obs_t → policy → actions → step(actions) → next_obs + rewards
    RLlib sends action_dict
            {
            "agent_0": 3,    --> Right
            "agent_1": 0     --> Up
            }

    """
    def step(self, action_dict):
        rewards = {}
        infos = {}

        self.step_count += 1  # increment timestep --> to track episode duration

        ACTIONS = {
            0: (0, 1),
            1: (0, -1),
            2: (-1, 0),
            3: (1, 0),
            4: (0, 0),
        }

        # -----------------------------
        # MOVE AGENTS
        # -----------------------------
        for agent, action in action_dict.items():
            dx, dy = ACTIONS[action]
            x, y = self.agent_pos[agent]

            new_x = np.clip(x + dx, 0, self.grid_size - 1)
            new_y = np.clip(y + dy, 0, self.grid_size - 1)

            self.agent_pos[agent] = (new_x, new_y)

        # -----------------------------
        # REWARD CALCULATION
        # -----------------------------
        for agent in self.agents:
            x, y = self.agent_pos[agent]

            if self.grid[x, y] == 0:
                rewards[agent] = 3.0  # reduced exploration reward
                self.grid[x, y] = 1

            elif self.grid[x, y] == 2:
                rewards[agent] = 20.0
                self.grid[x, y] = 1  # anomaly handled

            else:
                rewards[agent] = -1.0  # revisiting

            rewards[agent] -= 0.1  # time penalty

            # discourage staying
            if action_dict[agent] == 4:
                rewards[agent] -= 0.5

        # -----------------------------
        # ANTI-OVERLAP PENALTY 
        # -----------------------------
        positions = list(self.agent_pos.values())

        for i, agent_i in enumerate(self.agents):
            xi, yi = self.agent_pos[agent_i]

            for j, agent_j in enumerate(self.agents):
                if i == j:
                    continue

                xj, yj = self.agent_pos[agent_j]

                dist = abs(xi - xj) + abs(yi - yj)

                if dist == 0:
                    rewards[agent_i] -= 5.0
                elif dist == 1:
                    rewards[agent_i] -= 2.0

        # -----------------------------
        # GLOBAL TEAM REWARD 
        # -----------------------------
        explored = np.sum(self.grid == 1)

        for agent in self.agents:
            rewards[agent] += 0.05 * explored

        # -----------------------------
        # DONE CONDITIONS
        # -----------------------------
        all_explored = np.all(self.grid != 0)
        timeout = self.step_count >= self.max_steps
        done = all_explored or timeout

        terminateds = {agent: done for agent in self.agents}
        truncateds = {agent: False for agent in self.agents}

        terminateds["__all__"] = done
        truncateds["__all__"] = False

        for agent in self.agents:
            infos[agent] = {}

        return self._get_obs(), rewards, terminateds, truncateds, infos












































# import numpy as np
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
# from gymnasium import spaces


# class SwarmEnvRLlib(MultiAgentEnv):
#     def __init__(self, config=None):
#         super().__init__()

#         config = config or {}

#         self.grid_size = config.get("grid_size", 6)
#         self.n_agents = config.get("n_agents", 2)
#         self.max_steps = config.get("max_steps", 100)

#         # Agent IDs
#         self.agents = [f"agent_{i}" for i in range(self.n_agents)]

#         # Define SINGLE-agent spaces FIRST (VERY IMPORTANT)
#         obs_dim = self.grid_size * self.grid_size + 2

#         self.single_observation_space = spaces.Box(
#             low=0,
#             high=10,
#             shape=(obs_dim,),
#             dtype=np.float32
#         )

#         self.single_action_space = spaces.Discrete(5)

#         # THEN define multi-agent dict spaces
#         self.observation_space = {
#             agent: self.single_observation_space
#             for agent in self.agents
#         }

#         self.action_space = {
#             agent: self.single_action_space
#             for agent in self.agents
#         }

#         # Finally reset
#         self.reset()

#     def reset(self, *, seed=None, options=None):
#         self.grid = np.zeros((self.grid_size, self.grid_size))

#         # Place anomaly
#         self.anomaly_pos = (
#             np.random.randint(self.grid_size),
#             np.random.randint(self.grid_size)
#         )
#         self.grid[self.anomaly_pos] = 2

#         # Agent positions
#         self.agent_pos = {}
#         for i, agent in enumerate(self.agents):
#             self.agent_pos[agent] = (
#                 np.random.randint(self.grid_size),
#                 np.random.randint(self.grid_size)
#             )

#         self.step_count = 0

#         return self._get_obs(), {}
    

#     def get_action_space(self, agent_id):
#         return self.single_action_space

#     def get_observation_space(self, agent_id):
#         return self.single_observation_space

#     def _get_obs(self):
#         obs = {}
#         for agent in self.agents:
#             x, y = self.agent_pos[agent]

#             flat_grid = self.grid.flatten()

#             obs_vec = np.concatenate([
#                 flat_grid,
#                 np.array([x, y])
#             ])

#             obs[agent] = obs_vec.astype(np.float32)

#         return obs

#     def step(self, action_dict):
#         rewards = {}
#         dones = {}
#         infos = {}

#         self.step_count += 1

#         ACTIONS = {
#             0: (0, 1),
#             1: (0, -1),
#             2: (-1, 0),
#             3: (1, 0),
#             4: (0, 0),
#         }

#         for agent, action in action_dict.items():
#             dx, dy = ACTIONS[action]
#             x, y = self.agent_pos[agent]

#             new_x = np.clip(x + dx, 0, self.grid_size - 1)
#             new_y = np.clip(y + dy, 0, self.grid_size - 1)

#             self.agent_pos[agent] = (new_x, new_y)

#             # Reward logic
#             if self.grid[new_x, new_y] == 0:
#                 rewards[agent] = 5.0
#                 self.grid[new_x, new_y] = 1

#             elif self.grid[new_x, new_y] == 2:
#                 rewards[agent] = 20.0
#                 self.grid[new_x, new_y] = 1  # mark anomaly handled

#             else:
#                 rewards[agent] = -1.0

#             rewards[agent] -= 0.1

#         # Done condition
#         all_explored = np.all(self.grid != 0)
#         timeout = self.step_count >= self.max_steps
#         done = all_explored or timeout

#         for agent in self.agents:
#             dones[agent] = done
#             infos[agent] = {}

#         dones["__all__"] = done

#         terminateds = {agent: done for agent in self.agents}
#         truncateds = {agent: False for agent in self.agents}

#         terminateds["__all__"] = done
#         truncateds["__all__"] = False

#         return self._get_obs(), rewards, terminateds, truncateds, infos