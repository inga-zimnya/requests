class MultiAgentMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class RLAgent:
    def __init__(self, player_id: int, state_processor: StateProcessor):
        self.player_id = player_id
        self.state_processor = state_processor
        self.input_dim = state_processor.get_state_shape()
        self.output_dim = 5  # [up, down, left, right, shoot]

        # Initialize policy network
        self.policy_net = MultiAgentMLP(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

        # Training buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.2  # Exploration rate
        self.batch_size = 32

    def get_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Get action from policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Epsilon-greedy exploration
        if explore and np.random.rand() < self.epsilon:
            return np.random.rand(self.output_dim)

        with torch.no_grad():
            action_probs = torch.sigmoid(self.policy_net(state_tensor))

        return action_probs.numpy().squeeze()

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def update_policy(self):
        """Train policy network using experience replay"""
        if len(self.states) < self.batch_size:
            return

        # Sample batch
        indices = np.random.choice(len(self.states), self.batch_size, replace=False)
        states = torch.FloatTensor(np.array(self.states)[indices])
        actions = torch.FloatTensor(np.array(self.actions)[indices])
        rewards = torch.FloatTensor(np.array(self.rewards)[indices])
        next_states = torch.FloatTensor(np.array(self.next_states)[indices])
        dones = torch.FloatTensor(np.array(self.dones)[indices])

        # Compute Q values
        current_q = self.policy_net(states)
        next_q = self.policy_net(next_states).detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q.max(1)[0]

        # Compute loss
        loss = nn.MSELoss()(current_q.gather(1, actions.argmax(1).unsqueeze(1)), target_q.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear buffers if they get too large
        if len(self.states) > 10000:
            self.states = self.states[-5000:]
            self.actions = self.actions[-5000:]
            self.rewards = self.rewards[-5000:]
            self.next_states = self.next_states[-5000:]
            self.dones = self.dones[-5000:]

    def save_model(self, path: str):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])