import math
from abc import ABC, abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class MuZeroNetwork:
    def __new__(cls, config):
        if config.network == "fullyconnected":
            return MuZeroFullyConnectedNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.encoding_size,
                config.fc_reward_layers,
                config.fc_value_layers,
                config.fc_policy_layers,
                config.fc_representation_layers,
                config.fc_dynamics_layers,
                config.support_size,
            )
        elif config.network == "resnet":
            return MuZeroResidualNetwork(
                config.observation_shape,
                config.stacked_observations,
                len(config.action_space),
                config.blocks,
                config.channels,
                config.reduced_channels_reward,
                config.reduced_channels_value,
                config.reduced_channels_policy,
                config.resnet_fc_reward_layers,
                config.resnet_fc_value_layers,
                config.resnet_fc_policy_layers,
                config.support_size,
                config.downsample,
            )
        elif config.network == "Transformer":
            return AlphaDevNetwork(
                ##basic params
                #
                config.n_qubits,
                len(config.action_space),
                config.value_max,
                config.value_min,   
                ## shape of observation
                #
                config.batch_size,
                config.max_length_circuit,
                ## represent net params
                #
                config.input_dim,
                config.embedding_dim,
                config.nhead,
                config.num_encoderLayer,
                ## predict net params
                #
                config.policy_layers,
                ## value net params,
                #
                config.correctness_value_layers,
                config.length_value_layers,
                config.support_size,
            )


        else:
            raise NotImplementedError(
                'The network parameter should be "fullyconnected" or "resnet".'
            )


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class AbstractNetwork(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass

    def get_weights(self): # get network weights
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights): # set network weights
        self.load_state_dict(weights)


##################################
######## AlphaDevNetwork #########
class MultiQueryAttentionLayer(torch.nn.Module):
    def __init__(self, hid_dim, n_heads,  dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = self.hid_dim // self.n_heads

        self.fc_q = torch.nn.Linear( self.hid_dim, self.hid_dim)
        self.fc_k = torch.nn.Linear( self.hid_dim, self.head_dim)
        self.fc_v = torch.nn.Linear(self.hid_dim, self.head_dim)  
        self.fc_o = torch.nn.Linear(self.hid_dim, self.hid_dim)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
               
        Qbank = self.fc_q(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        Kbank = self.fc_k(key).view(batch_size, -1, 1, self.head_dim).permute(0, 2, 3, 1)
        Vbank = self.fc_v(value).view(batch_size, -1, 1, self.head_dim).permute(0, 2, 1, 3)   
        
        #Qbank = [batch size, n heads, query len, head dim]
        #Kbank = [batch size, 1, head dim, key len]
        #Vbank = [batch size, 1, value len, head dim]

        energy = torch.matmul(Qbank, Kbank) / self.scale

        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = F.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), Vbank)

        x = x.permute(0, 2, 1, 3).contiguous()

        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, seq len, hid dim]
        
        x = self.fc_o(x)
        
        return x, attention

class RepresentNet_Trans(torch.nn.Module):
    def __init__(self, 
                 batch_size,
                 num_qubits,
                 action_space_size, 
                 max_length_circuit, 
                 input_dim,
                 embedding_dim, 
                 num_encoderLayer,
                 nhead = 4, 
                 name: str = 'representation'
                 ):
        super(RepresentNet_Trans, self).__init__()
        self.batch_size = batch_size
        self.num_qubits = num_qubits
        self.num_gates = len(action_space_size)
        self.num_gateTypes = int(self.num_gates / self.num_qubits)
        self.num_positions = max_length_circuit
        self.input_dim = input_dim # input_dim = gate_onehot.shape[-1] + control_onehot.shape[-1] + target_onehot.shape[-1]
        self.embedding_dim = embedding_dim
        self.nhead = nhead  # featsize // nhead == 0
        self.num_encoderLayer = num_encoderLayer


        self.mlp_embedder = torch.nn.Sequential(
            nn.Linear(self.input_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim))
        
        # self.observation_embedding = torch.nn.Embedding(num_embeddings=self.num_gates, embedding_dim=self.embedding_dim)
        self.position_embedding = torch.nn.Embedding(num_embeddings=self.num_positions, embedding_dim=self.embedding_dim)
        self.attenion_block = torch.nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=8, dropout=0.1)
        # self.attenion_block = MultiQueryAttentionLayer(hid_dim=self.embedding_dim, n_heads=8, dropout=0.1, device='gpu')    
        self.TransformerEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead= self.nhead )
        self.TransformerEncoder = torch.nn.TransformerEncoder(encoder_layer=self.TransformerEncoderLayer, num_layers=self.num_encoderLayer)
        ## if necessary
        # self.output_layer = torch.nn.Linear(self.embedding_dim, self.output_size)
    def forward(self, obervation):
        ## 
        # for stack is ok; for single sample?
    
        # one-hot encoding for circuit_stack
        obervation_onehot = self.Cir2onehot(obervation, self.batch_size, self.num_positions)
        # mlp embed the one-hots
        obervation_embedded = self.mlp_embedder(obervation_onehot)
        # position encoding
        batch_size, seq_length, feat_size = obervation_embedded.shape
        position_encodings = sinusoidal_position_encoding(batch_size, seq_length, feat_size)
        if isinstance(position_encodings, np.ndarray):
            position_encodings = torch.from_numpy(position_encodings).to(obervation_embedded.device)
        # add position encoding to circuits_onehot
        obervation_position_embedded = obervation_embedded + position_encodings
        # Transformer Encoder 
        obervation_position_embedded = obervation_position_embedded.permute(1, 0, 2)
        output = self.TransformerEncoder(obervation_position_embedded)
        
        return output 
    
    def Cir2onehot(self, circuit_stack, batch_size, max_length_circuit):
        ## convert circuit to one-hot encoding
        #
        gate = circuit_stack[:, :, 0]
        control = circuit_stack[:, :, 1]
        target = circuit_stack[:, :, 2]

        gate_onehot = F.one_hot(gate, num_classes=self.num_gateTypess)
        control_onehot = F.one_hot(control, num_classes=self.num_qubits)
        target_onehot = F.one_hot(target, num_classes=self.num_qubits)

        ##
        # one-hot encoding for gate: [gate_type, control, target]
        # circuits_onehot.shape : [batch_size, length_circuit, gates_onehot]
        circuits_onehot = torch.cat([gate_onehot, control_onehot, target_onehot], dim=-1)

        ## padding
        # 
        if circuits_onehot.shape[1] < max_length_circuit:
            padding = torch.zeros(batch_size, max_length_circuit - circuits_onehot.shape[1], circuits_onehot.shape[2])
            circuits_onehot = torch.cat([circuits_onehot, padding], dim=1)   
        
        assert circuits_onehot.shape == (batch_size, max_length_circuit, gate_onehot.shape[-1] + control_onehot.shape[-1] + target_onehot.shape[-1]), \
        "Shape mismatch in circuits stack one-hot encoding."
        return circuits_onehot
    
def sinusoidal_position_encoding(batch_size, seq_length, feat_size):
    ## generate position encoding
    #
    position_enc = np.array([
        [pos / np.power(10000, 2. * i / feat_size) for i in range(feat_size)]
        for pos in range(seq_length)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    
    position_encodings = np.broadcast_to(position_enc, (batch_size, seq_length, feat_size))
    return position_encodings

class PolicyNetwork(torch.nn.Module):
    def __init__(self, 
                 action_space_size, 
                 embedding_dim, 
                 num_layers=2
        ):
        super(PolicyNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.output_size = action_space_size

        ## define mlp
        layers = []
        input_dim = embedding_dim
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(input_dim, embedding_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embedding_dim

        self.mlp = torch.nn.Sequential(*layers)
        
        ## define output layer
        self.output_layer = torch.nn.Linear(embedding_dim, self.output_size)

    def forward(self, x):
        x = self.mlp(x)
        x = self.output_layer(x)
        return x

class DistributionSupport(object):

    def __init__(self, value_max: float, value_min: float, num_bins: int):
        self.value_max = value_max
        self.value_min = value_min
        self.num_bins = num_bins
        self.full_support_size = 2 * num_bins + 1

    @property  
    def support(self) -> np.ndarray:
        delta = self.value_max - self.value_min / (self.full_support_size)      
        return np.array([self.value_min + i * delta for i in range(self.full_support_size)], dtype="float32")

class CategoricalNet(torch.nn.Module):
    ## output is categorical distribution
    # prediction of vlaue based on the distribution 
    # instead of the value directly
    def __init__(
        self,
        embedding_dim: int,
        support: DistributionSupport,
        num_layers = 2,
        name = 'CategoricalHead'
    ):
        super().__init__(name=name)
        self._value_support = support
        self.input_dim = embedding_dim
        self.num_layers = num_layers
        self.output_size = self._value_support.full_support_size

        ## define mlp
        layers = []
        input_dim = self._embedding_dim

        for _ in range(num_layers):
            layers.append(torch.nn.Linear(input_dim, embedding_dim))
            layers.append(torch.nn.ReLU())
            input_dim = embedding_dim

        ## define output layer
        output_layer = torch.nn.Linear(embedding_dim, self.output_size)
        layers += output_layer
        
        self.CateNet = torch.nn.Sequential(*layers)


    def forward(self, x):
        # For training returns the logits
        # For inference the mean
        logits = self.CateNet(x)
        ## weighted mean
        probs = torch.nn.softmax(logits)
        support = torch.tensor(self._value_support.support, device=x.device)
        mean = torch.sum(probs * support, dim=-1)  
        return dict(logits=logits, mean=mean)

  
class AlphaDevNetwork(AbstractNetwork):
    def __init__(
        self,
        ## basic params
        num_qubits,
        action_space_size, 
        value_max,
        value_min,
        ## shape of observation
        batch_size,
        max_length_circuit, 
        ## represent net params
        input_dim, ## feat_size
        embedding_dim, 
        num_encoderLayer,
        nhead,  # 4
        ## predict net params
        policy_layers,
        correctness_value_layers,
        cirlength_value_layers,
        support_size,

        ## ???
        observation_shape,
        stacked_observations,
        
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.action_space_size = action_space_size
        self.max_length_circuit = max_length_circuit
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_encoderLayer = num_encoderLayer
        self.nhead = nhead
        self.batch_size = batch_size 
        self.support = DistributionSupport(value_max, value_min, num_bins=support_size)
        self.full_support_size = self.support.full_support_size

        self.representation_network = torch.nn.DataParallel(
            RepresentNet_Trans(
                self.batch_size,
                self.num_qubits,
                self.action_space_size, 
                self.max_length_circuit, 
                self.input_dim, ## feat_size
                self.embedding_dim, 
                self.num_encoderLayer,
                self.nhead,                                                                                                                                                               
        )
            )

        self.policy_network = torch.nn.DataParallel(
            PolicyNetwork(  
                self.action_space_size,
                embedding_dim,
                policy_layers
                
        )     
            )
        
        self.correctness_value_network = torch.nn.DataParallel(
            CategoricalNet(  
                embedding_dim,
                self.support,
                num_layers = correctness_value_layers
                
        )
           )

        self.cirlength_value_network = torch.nn.DataParallel(
            CategoricalNet(
                embedding_dim,
                self.support,
                num_layers = cirlength_value_layers 
                
        )
           )

    def prediction(self, encoded_state):
        policy_logits = self.policy_network(encoded_state)
        correctness_value = self.correctness_value_network(encoded_state)
        cirlength_value = self.cirlength_value_network(encoded_state)
        value = correctness_value["mean"] + cirlength_value["mean"] 
        return policy_logits, value, correctness_value["logits"], cirlength_value["logits"]

    def representation(self, observation):
        return self.representation_network(observation)
        
    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value, correctness_value_logits, length_value_logits = self.prediction(encoded_state)

        return (
            value,
            correctness_value_logits, 
            length_value_logits,
            policy_logits,
        )

    def recurrent_inference(self):
        ## use dynamics net to get next_hidden_state
        ## then use next_hidden_state to predict value, policy_logits
        pass ## not be used in AlphaDevNetwork



###### End AlphaDevNetwork #######
##################################

##################################
######## Fully Connected #########


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        encoding_size,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        fc_representation_layers,
        fc_dynamics_layers,
        support_size,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = torch.nn.DataParallel(
            mlp(
                observation_shape[0]
                * observation_shape[1]
                * observation_shape[2]
                * (stacked_observations + 1)
                + stacked_observations * observation_shape[1] * observation_shape[2],
                fc_representation_layers,
                encoding_size,
            )
        )

        self.dynamics_encoded_state_network = torch.nn.DataParallel(
            mlp(
                encoding_size + self.action_space_size,
                fc_dynamics_layers,
                encoding_size,
            )
        )
        self.dynamics_reward_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_reward_layers, self.full_support_size)
        )

        self.prediction_policy_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_policy_layers, self.action_space_size)
        )
        self.prediction_value_network = torch.nn.DataParallel(
            mlp(encoding_size, fc_value_layers, self.full_support_size)
        )

    def prediction(self, encoded_state):
        policy_logits = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logits, value

    def representation(self, observation):
        encoded_state = self.representation_network(
            observation.view(observation.shape[0], -1)
        )
        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = encoded_state.min(1, keepdim=True)[0]
        max_encoded_state = encoded_state.max(1, keepdim=True)[0]
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)

        reward = self.dynamics_reward_network(next_encoded_state)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = next_encoded_state.min(1, keepdim=True)[0]
        max_next_encoded_state = next_encoded_state.max(1, keepdim=True)[0]
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state

        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )

        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


###### End Fully Connected #######
##################################


##################################
############# ResNet #############


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = torch.nn.functional.relu(out)
        return out


# Downsample observations before representation network (See paper appendix Network Architecture)
class DownSample(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks1 = torch.nn.ModuleList(
            [ResidualBlock(out_channels // 2) for _ in range(2)]
        )
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.resblocks2 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = torch.nn.ModuleList(
            [ResidualBlock(out_channels) for _ in range(3)]
        )
        self.pooling2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


class DownsampleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, h_w):
        super().__init__()
        mid_channels = (in_channels + out_channels) // 2
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels, mid_channels, kernel_size=h_w[0] * 2, stride=4, padding=2
            ),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(h_w)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x


class RepresentationNetwork(torch.nn.Module):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        num_blocks,
        num_channels,
        downsample,
    ):
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            if self.downsample == "resnet":
                self.downsample_net = DownSample(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                )
            elif self.downsample == "CNN":
                self.downsample_net = DownsampleCNN(
                    observation_shape[0] * (stacked_observations + 1)
                    + stacked_observations,
                    num_channels,
                    (
                        math.ceil(observation_shape[1] / 16),
                        math.ceil(observation_shape[2] / 16),
                    ),
                )
            else:
                raise NotImplementedError('downsample should be "resnet" or "CNN".')
        self.conv = conv3x3(
            observation_shape[0] * (stacked_observations + 1) + stacked_observations,
            num_channels,
        )
        self.bn = torch.nn.BatchNorm2d(num_channels)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = torch.nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        fc_reward_layers,
        full_support_size,
        block_output_size_reward,
    ):
        super().__init__()
        self.conv = conv3x3(num_channels, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        self.conv1x1_reward = torch.nn.Conv2d(
            num_channels - 1, reduced_channels_reward, 1
        )
        self.block_output_size_reward = block_output_size_reward
        self.fc = mlp(
            self.block_output_size_reward,
            fc_reward_layers,
            full_support_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.nn.functional.relu(x)
        for block in self.resblocks:
            x = block(x)
        state = x
        x = self.conv1x1_reward(x)
        x = x.view(-1, self.block_output_size_reward)
        reward = self.fc(x)
        return state, reward


class PredictionNetwork(torch.nn.Module):
    def __init__(
        self,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_value,
        reduced_channels_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_size_value,
        block_output_size_policy,
    ):
        super().__init__()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

        self.conv1x1_value = torch.nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.conv1x1_policy = torch.nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.block_output_size_value = block_output_size_value
        self.block_output_size_policy = block_output_size_policy
        self.fc_value = mlp(
            self.block_output_size_value, fc_value_layers, full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_size_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        value = self.conv1x1_value(x)
        policy = self.conv1x1_policy(x)
        value = value.view(-1, self.block_output_size_value)
        policy = policy.view(-1, self.block_output_size_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value






class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(
        self,
        observation_shape,
        stacked_observations,
        action_space_size,
        num_blocks,
        num_channels,
        reduced_channels_reward,
        reduced_channels_value,
        reduced_channels_policy,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        support_size,
        downsample,
    ):
        super().__init__()
        self.action_space_size = action_space_size
        self.full_support_size = 2 * support_size + 1
        block_output_size_reward = (
            (
                reduced_channels_reward
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_reward * observation_shape[1] * observation_shape[2])
        )

        block_output_size_value = (
            (
                reduced_channels_value
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_value * observation_shape[1] * observation_shape[2])
        )

        block_output_size_policy = (
            (
                reduced_channels_policy
                * math.ceil(observation_shape[1] / 16)
                * math.ceil(observation_shape[2] / 16)
            )
            if downsample
            else (reduced_channels_policy * observation_shape[1] * observation_shape[2])
        )

        self.representation_network = torch.nn.DataParallel(
            RepresentationNetwork(
                observation_shape,
                stacked_observations,
                num_blocks,
                num_channels,
                downsample,
            )
        )

        self.dynamics_network = torch.nn.DataParallel(
            DynamicsNetwork(
                num_blocks,
                num_channels + 1,
                reduced_channels_reward,
                fc_reward_layers,
                self.full_support_size,
                block_output_size_reward,
            )
        )

        self.prediction_network = torch.nn.DataParallel(
            PredictionNetwork(
                action_space_size,
                num_blocks,
                num_channels,
                reduced_channels_value,
                reduced_channels_policy,
                fc_value_layers,
                fc_policy_layers,
                self.full_support_size,
                block_output_size_value,
                block_output_size_policy,
            )
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_encoded_state = (
            encoded_state.view(
                -1,
                encoded_state.shape[1],
                encoded_state.shape[2] * encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_encoded_state = max_encoded_state - min_encoded_state
        scale_encoded_state[scale_encoded_state < 1e-5] += 1e-5
        encoded_state_normalized = (
            encoded_state - min_encoded_state
        ) / scale_encoded_state
        return encoded_state_normalized

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with a game specific one hot encoded action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                    encoded_state.shape[2],
                    encoded_state.shape[3],
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = (
            action[:, :, None, None] * action_one_hot / self.action_space_size
        )
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward = self.dynamics_network(x)

        # Scale encoded state between [0, 1] (See paper appendix Training)
        min_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_next_encoded_state = (
            next_encoded_state.view(
                -1,
                next_encoded_state.shape[1],
                next_encoded_state.shape[2] * next_encoded_state.shape[3],
            )
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_next_encoded_state = max_next_encoded_state - min_next_encoded_state
        scale_next_encoded_state[scale_next_encoded_state < 1e-5] += 1e-5
        next_encoded_state_normalized = (
            next_encoded_state - min_next_encoded_state
        ) / scale_next_encoded_state
        return next_encoded_state_normalized, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(observation), 1)
                .to(observation.device)
            )
        )
        return (
            value,
            reward,
            policy_logits,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


########### End ResNet ###########
##################################


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

def supoprt_to_scalar_simply(logits, support_size):
    """
    Transform a categorical representation to a scalar
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = torch.tensor([x for x in range(-support_size, support_size + 1)]).float().to(device=probabilities.device)
    x = torch.sum(support * probabilities, dim=1, keepdim=True)
    return x    


def support_to_scalar(logits, support_size):
    """
    Transform a categorical representation to a scalar
    See paper appendix Network Architecture
    """
    # Decode to a scalar
    probabilities = torch.softmax(logits, dim=1)
    support = (
        torch.tensor([x for x in range(-support_size, support_size + 1)])
        .expand(probabilities.shape)
        .float()
        .to(device=probabilities.device)
    )
    x = torch.sum(support * probabilities, dim=1, keepdim=True)

    # Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (
        ((torch.sqrt(1 + 4 * 0.001 * (torch.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        ** 2
        - 1
    )
    return x


def scalar_to_support(x, support_size):
    """
    Transform a scalar to a categorical representation with (2 * support_size + 1) categories
    See paper appendix Network Architecture
    """
    # Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
    x = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x

    # Encode on a vector
    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits

def scalar_to_support_simply(x, support_size):

    x = torch.clamp(x, -support_size, support_size)
    floor = x.floor()
    prob = x - floor
    logits = torch.zeros(x.shape[0], x.shape[1], 2 * support_size + 1).to(x.device)
    logits.scatter_(
        2, (floor + support_size).long().unsqueeze(-1), (1 - prob).unsqueeze(-1)
    )
    indexes = floor + support_size + 1
    prob = prob.masked_fill_(2 * support_size < indexes, 0.0)
    indexes = indexes.masked_fill_(2 * support_size < indexes, 0.0)
    logits.scatter_(2, indexes.long().unsqueeze(-1), prob.unsqueeze(-1))
    return logits
