import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces


class SwarmEnvRLlib(MultiAgentEnv):

    """
    What this means:

    You are inheriting RLlib’s multi-agent API.

    So RLlib expects:
    reset()

    Starts episode

    step(actions)

    Advances simulation

    observation/action spaces                               
    RLlib treats this like:
    “A game with multiple players.”

    """

    def __init__(self, config=None):

        super().__init__()

        config = config or {}

        self.grid_size = config.get("grid_size", 6)
        self.n_agents = config.get("n_agents", 2)
        self.max_steps = config.get("max_steps", 100) # episode horizon

        self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        # -------------------------------------------------
        # OBSERVATION SPACE
        # -------------------------------------------------
        obs_dim = (
            self.grid_size * self.grid_size +  # flattened grid
            2 +                                # own x,y
            2 * (self.n_agents - 1) +          # other agents
            self.n_agents                      # one-hot ID
        )

        self.single_observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # 0=UP,1=DOWN,2=LEFT,3=RIGHT,4=STAY
        self.single_action_space = spaces.Discrete(5)

        self.observation_space = {
            agent: self.single_observation_space
            for agent in self.agents
        }

        self.action_space = {
            agent: self.single_action_space
            for agent in self.agents
        }

        self.reset()

    # -------------------------------------------------
    # REQUIRED API
    # -------------------------------------------------
    def get_action_space(self, agent_id):
        return self.single_action_space

    def get_observation_space(self, agent_id):
        return self.single_observation_space

    # -------------------------------------------------
    # RESET
    # -------------------------------------------------
    def reset(self, *, seed=None, options=None):

        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Place anomaly
        self.anomaly_pos = (
            np.random.randint(self.grid_size),
            np.random.randint(self.grid_size)
        )

        self.grid[self.anomaly_pos] = 2

        # Agent positions
        self.agent_pos = {}
        
        used_positions = set()
        used_positions.add(self.anomaly_pos)

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

        # -----------------------------------------
        # ANOMALY STATE MACHINE
        # -----------------------------------------
        self.detected = False
        self.resolved = False
        self.detected_by = None
        self.investigation_timer = 0
        self.anomaly_cleared = False
        
        return self._get_obs(), {}

    # -------------------------------------------------
    # OBSERVATIONS
    # -------------------------------------------------

    # observations without relative position of other agents
    # def _get_obs(self):

    #     obs = {}

    #     for agent in self.agents:

    #         x, y = self.agent_pos[agent]

    #         visible_grid = self.grid.copy()

    #         if not self.detected:
    #             visible_grid[self.anomaly_pos] = 0

    #         flat_grid = visible_grid.flatten()

    #         agent_id = int(agent.split("_")[1])

    #         id_vec = np.zeros(self.n_agents)
    #         id_vec[agent_id] = 1

    #         obs_vec = np.concatenate([
    #             flat_grid,
    #             np.array([x, y]),
    #             id_vec
    #         ])

    #         obs[agent] = obs_vec.astype(np.float32)

    #     return obs



    def _get_obs(self):

        obs = {}

        for agent in self.agents:

            x, y = self.agent_pos[agent]

            visible_grid = self.grid.copy()

            if not self.detected:
                visible_grid[self.anomaly_pos] = 0

            flat_grid = visible_grid.flatten()

            agent_id = int(agent.split("_")[1])

            id_vec = np.zeros(self.n_agents)
            id_vec[agent_id] = 1

            other_positions = []

            for other in self.agents:

                if other == agent:
                    continue

                ox, oy = self.agent_pos[other]

                other_positions.extend([
                    (ox - x) / self.grid_size,
                    (oy - y) / self.grid_size
                ])

            obs_vec = np.concatenate([
                flat_grid,
                np.array([
                    x / self.grid_size,
                    y / self.grid_size
                ]),
                np.array(other_positions),
                id_vec
            ])

            obs[agent] = obs_vec.astype(np.float32)

        return obs



    # -------------------------------------------------
    # DISTANCE
    # -------------------------------------------------
    def manhattan(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    # -------------------------------------------------
    # STEP
    # -------------------------------------------------
    def step(self, action_dict):

        rewards = {}
        infos = {}
        resolved_this_step = False
        proposed_positions = {}


        self.step_count += 1

        ACTIONS = {
            0: (0, 1),    # UP
            1: (0, -1),   # DOWN
            2: (-1, 0),   # LEFT
            3: (1, 0),    # RIGHT
            4: (0, 0),    # STAY
        }

        prev_positions = self.agent_pos.copy()

        # -------------------------------------------------
        # MOVE
        # -------------------------------------------------
        for agent, action in action_dict.items():

            dx, dy = ACTIONS[action]

            x, y = self.agent_pos[agent]

            new_x = np.clip(x + dx, 0, self.grid_size - 1)
            new_y = np.clip(y + dy, 0, self.grid_size - 1)

            self.agent_pos[agent] = (new_x, new_y)
            proposed_positions[agent] = (new_x, new_y)

        # -------------------------------------------------
        # BASE REWARDS
        # -------------------------------------------------
        for agent in self.agents:

            x, y = self.agent_pos[agent]

            rewards[agent] = 0.0

            # Exploration
            if self.grid[x, y] == 0:
                rewards[agent] += 3.0
                self.grid[x, y] = 1

            elif self.grid[x, y] in [1,4]:
                rewards[agent] -= (1.0 + 0.02 * self.step_count)


            # -----------------------------------------
            # DETECTION
            # -----------------------------------------
            elif self.grid[x, y] == 2:

                # First detector
                if not self.detected:
                    self.detected = True
                    self.detected_by = agent
                    self.investigation_timer = 0

                    rewards[agent] += 20.0

                    # Mark detected anomaly
                    self.grid[x, y] = 3

                else:
                    rewards[agent] += 1.0

            # Time penalty
            rewards[agent] -= 0.1

            # Hover penalty
            if action_dict[agent] == 4:
                rewards[agent] -= 0.5

        # -----------------------------------------
        # PROACTIVE COLLISION AVOIDANCE
        # -----------------------------------------

        for agent_i in self.agents:
            for agent_j in self.agents:
                if agent_i >= agent_j:
                    continue

                if proposed_positions[agent_i] == proposed_positions[agent_j]:
                    rewards[agent_i] -= 8.0
                    rewards[agent_j] -= 8.0

                if (
                    proposed_positions[agent_i] == prev_positions[agent_j]
                    and proposed_positions[agent_j] == prev_positions[agent_i]
                ):
                    rewards[agent_i] -= 6.0
                    rewards[agent_j] -= 6.0

        # -------------------------------------------------
        # REACTIVE COLLISION PENALTY
        # -------------------------------------------------
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


        # -------------------------------------------------
        # UNIQUE REGION COVERAGE REWARD
        # -------------------------------------------------

        for agent in self.agents:

            my_pos = self.agent_pos[agent]

            min_dist = min(
                self.manhattan(my_pos, self.agent_pos[other])
                for other in self.agents if other != agent
            )

            spread_reward = min_dist / (self.grid_size)
            early_factor = max(0.5, 1 - self.step_count / 20)

            rewards[agent] += early_factor*0.5 * min(spread_reward, 0.5)

        # -------------------------------------------------
        # GLOBAL EXPLORE BONUS
        # -------------------------------------------------
        explored = np.sum((self.grid == 1) | (self.grid == 3) | (self.grid == 4)) 

        for agent in self.agents:
            rewards[agent] += 0.03 * explored

        # -------------------------------------------------
        # INVESTIGATION PHASE
        # -------------------------------------------------
        if self.detected and not self.resolved:

            self.investigation_timer += 1

            # -----------------------------------------
            # Find nearest NON-detector
            # Tie-break = lower index
            # -----------------------------------------
            candidates = []

            for agent in self.agents:

                if agent == self.detected_by:
                    continue

                dist = self.manhattan(
                    self.agent_pos[agent],
                    self.anomaly_pos
                )

                idx = int(agent.split("_")[1])

                candidates.append((dist, idx, agent))

            candidates.sort()

            assigned_agent = candidates[0][2]

            # -----------------------------------------
            # Reward shaping
            # -----------------------------------------
            for agent in self.agents:

                # Detector should leave
                if agent == self.detected_by:

                    det_dist = self.manhattan(
                    self.agent_pos[agent],
                    self.anomaly_pos
                )

                    if det_dist <= 1:
                        rewards[agent] -= 4.0

                    elif det_dist == 2:
                        rewards[agent] -= 1.5

                # Assigned investigator
                elif agent == assigned_agent:

                    old_dist = self.manhattan(
                        prev_positions[agent],
                        self.anomaly_pos
                    )

                    new_dist = self.manhattan(
                        self.agent_pos[agent],
                        self.anomaly_pos
                    )

                    # Moving closer good
                    rewards[agent] += (old_dist - new_dist) * 4.0

                    # Delay penalty
                    #rewards[agent] -= 0.3 * self.investigation_timer
                    rewards[agent] -= 1.0 * self.investigation_timer

                    # Resolve
                    if self.agent_pos[agent] == self.anomaly_pos:

                        self.resolved = True

                        rewards[agent] += 50.0
                        rewards[self.detected_by] += 10.0

                # Other drones discouraged
                else:

                    if self.agent_pos[agent] == self.anomaly_pos:
                        rewards[agent] -= 10

                    # new_dist = self.manhattan(
                    #     self.agent_pos[agent],
                    #     self.anomaly_pos
                    # )

                    # if new_dist < 2:
                    #     rewards[agent] -= 3.0

            # -----------------------------------------
            # Fail-safe:
            # If assigned too slow, others can take over
            # -----------------------------------------
            if self.investigation_timer > 15:

                for agent in self.agents:

                    if agent == self.detected_by:
                        continue

                    if self.agent_pos[agent] == self.anomaly_pos:

                        self.resolved = True
                        rewards[agent] += 30.0


        # -------------------------------------------------
        # STRONGER TERRITORY SEPARATION
        # -------------------------------------------------          

        for other in self.agents:
            if other == agent:
                continue

            dist = self.manhattan(
                proposed_positions[agent],
                proposed_positions[other]
            )

            if dist <= 2:
                rewards[agent] -= 2.0


        # -------------------------------------------------
        # NEW FRONTIER EXPLORATION
        # -------------------------------------------------  
        
        for agent in self.agents:

            x, y = self.agent_pos[agent]

            unexplored = np.argwhere(self.grid == 0)

            if len(unexplored) > 0:

                dists = [
                    abs(x - ux) + abs(y - uy)
                    for ux, uy in unexplored
                ]

                nearest = min(dists)

                rewards[agent] -= 0.2 * (nearest / self.grid_size)

        # -------------------------------------------------
        # RESET AFTER RESOLUTION
        # -------------------------------------------------
        if self.resolved:

            self.grid[self.anomaly_pos] = 4
            self.detected = False
            self.resolved = False
            self.detected_by = None
            self.investigation_timer = 0
            self.anomaly_cleared = True
            resolved_this_step = True

            # # New anomaly
            # while True:

            #     new_anomaly = (
            #         np.random.randint(self.grid_size),
            #         np.random.randint(self.grid_size)
            #     )

            #     occupied = set(self.agent_pos.values())

            #     if new_anomaly not in occupied:
            #         break

            # self.anomaly_pos = new_anomaly
            # self.grid[self.anomaly_pos] = 2

        # -------------------------------------------------
        # DONE
        # -------------------------------------------------
        all_cells_explored = np.all(
            (self.grid == 1) |
            (self.grid == 3) |
            (self.grid == 4)
        )

        #anomaly_cleared = self.grid[self.anomaly_pos] == 4

        mission_complete = all_cells_explored and self.anomaly_cleared

        timeout = self.step_count >= self.max_steps

        terminateds = {
            agent: mission_complete for agent in self.agents
        }

        truncateds = {
            agent: timeout for agent in self.agents
        }

        terminateds["__all__"] = mission_complete
        truncateds["__all__"] = timeout

        # -------------------------------------------------
        # INFO
        # -------------------------------------------------
        for agent in self.agents:
            infos[agent] = {
                "detected": self.detected,                
                "detected_by": self.detected_by,
                "timer": self.investigation_timer,
                "resolved_this_step":resolved_this_step,
                "anomaly_cleared": self.anomaly_cleared
            }

        return self._get_obs(), rewards, terminateds, truncateds, infos


