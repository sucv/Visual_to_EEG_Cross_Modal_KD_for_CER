from models.arcface_model import Backbone
from models.temporal_convolutional_model import TemporalConvNet

import torch
from torch import nn

from torch.nn import Linear, BatchNorm1d, BatchNorm2d, Dropout, Sequential, Module


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class my_res50(nn.Module):
    def __init__(self, num_classes=8, embedding_dim=512):
        super().__init__()
        self.backbone = Backbone(input_channels=3, num_layers=50, drop_ratio=0.4, mode="ir")


        self.backbone.output_layer = Sequential(BatchNorm2d(embedding_dim),
                                                Dropout(0.4),
                                                Flatten(),
                                                Linear(embedding_dim * 5 * 5, embedding_dim),
                                                BatchNorm1d(embedding_dim))

        self.logits = nn.Linear(in_features=embedding_dim, out_features=num_classes)

        from torch.nn.init import xavier_uniform_, constant_

        for m in self.backbone.output_layer.modules():
            if isinstance(m, nn.Linear):
                m.weight = xavier_uniform_(m.weight)
                m.bias = constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.logits.weight = xavier_uniform_(self.logits.weight)
        self.logits.bias = constant_(self.logits.bias, 0)

    def forward(self, x, extract_cnn=False):
        x = self.backbone(x)

        if extract_cnn:
            return x

        x = self.logits(x)
        return x


class my_temporal(nn.Module):
    def __init__(self, model_name, visual_backbone_path,
                 num_inputs=192, cnn1d_channels=[128, 128, 128], cnn1d_kernel_size=5,
                 cnn1d_dropout_rate=0.1, modality=['video'],
                 embedding_dim=256, hidden_dim=128, lstm_dropout_rate=0.5, bidirectional=True, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.model_name = model_name

        # Initialize the visual backbone

        if 'video' in modality:
            spatial = my_res50()
            state_dict = torch.load(visual_backbone_path, map_location='cpu')
            spatial.load_state_dict(state_dict)

            for param in spatial.parameters():
                param.requires_grad = False

            self.spatial = spatial.backbone

        # Initialize the temporal module
        if "tcn" in model_name and "lstm" in model_name:
            raise ValueError("Please disambiguate the model name since it includes both tcn and lstm!")

        if "tcn" in model_name:
            self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=cnn1d_channels,
                                            kernel_size=cnn1d_kernel_size, dropout=cnn1d_dropout_rate)
            self.regressor = nn.Linear(cnn1d_channels[-1], output_dim)

        elif "lstm" in model_name:
            self.temporal = nn.LSTM(input_size=num_inputs, hidden_size=hidden_dim, num_layers=2,
                                    batch_first=True, bidirectional=bidirectional, dropout=lstm_dropout_rate)
            input_dim = hidden_dim
            if bidirectional:
                input_dim = hidden_dim * 2

            self.regressor = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        assert len(x.keys()) == 1, "This models is not designed for more than one modalities."

        if "video" in x:
            num_batches, length, channel, width, height = x['video'].shape
            x = x['video'].view(-1, channel, width, height)
            x = self.spatial(x)
            _, feature_dim = x.shape
            x = x.view(num_batches, length, feature_dim).contiguous()
        else:
            x = x[list(x.keys())[0]]
            x = x.squeeze(1)

        if "lstm" in self.model_name:
            x, _ = self.temporal(x)
            x = x.contiguous()
        else:
            x = x.transpose(1, 2).contiguous()
            x = self.temporal(x).transpose(1, 2).contiguous()
        feature = x

        x = self.regressor(x).contiguous()

        return x, feature

