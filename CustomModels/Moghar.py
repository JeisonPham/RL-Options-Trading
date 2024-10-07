from typing import List, Tuple

import numpy as np
import gymnasium as gym

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input

model = Sequential()


tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class Moghar(TorchRNN, nn.Module):
    def __init__(
            self,
            obs_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
            dropout: float = 0.5,
            num_layers: int = 4
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.num_layers = num_layers

        self.obs_size = get_preprocessor(obs_space)(obs_space).size

        self.lstm = nn.LSTM(self.obs_size, 96, num_layers=self.num_layers, dropout=dropout, batch_first=True)
        self.action_branch = nn.Linear(96, num_outputs)
        self.value_branch = nn.Linear(96, 1)

    @override(ModelV2)
    def get_initial_state(self):
        return [torch.zeros(self.num_layers, 96), torch.zeros(self.num_layers, 96)]

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.value_branch(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(
            self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:

        # if state[0].shape[1] == self.num_layers:
        #     state[0] = torch.swapaxes(state[0], 0, 1)
        #     state[1] = torch.swapaxes(state[1], 0, 1)
        if state[0]:
            self._features, [h, c] = self.lstm(inputs, [state[0], state[1]])
        else:
            self._features, [h, c] = self.lstm(inputs)

        action_out = self.action_branch(self._features)
        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]


class MogharTF(RecurrentNetwork):
    """Example implementation of the Moghar using keras"""

    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, hidden_size=256, cell_size=64):
        super(MogharTF, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        self.cell_size = cell_size

        model = Sequential()
        model.add(Input(shape=(None, obs_space.shape[0]), name='inputs'))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        self.model = model
        self.model.summary()

    @override(RecurrentNetwork)
    def forward_rnn(
        self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        model_out, self._value_out, h, c = self.model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self):
        return [
            np.zeros(50, np.float32),
            np.zeros(50, np.float32)
        ]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])


