import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from resnet import *
from densenet import *
from senet import *
from bninception import *

class Feature_Network_Wrapper(nn.Module):
    def __init__(self, net_type, network):
        super(Feature_Network_Wrapper, self).__init__()

        self.net_type = net_type
        self.network = network

        if self.net_type == 'resnet':
            self.classifier = self.network.fc

    def forward(self, x):
        if self.net_type == 'resnet':
            x = self._resnet_forward(x)
        elif self.net_type == 'densenet':
            x = self._densenet_forward(x)

        return x

    def _resnet_forward(self, x):
        x = self.network.conv1(x)
        x = self.network.bn1(x)
        x = self.network.relu(x)
        x = self.network.maxpool(x)

        x = self.network.layer1(x)
        x = self.network.layer2(x)
        x = self.network.layer3(x)
        x = self.network.layer4(x)

        x = self.network.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def _densenet_forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        return out

def create_network(name, pretrained, num_classes, drop_rate=0.):
    network_type = name.split('-')[0]
    if len(name.split('-')) > 1:
        network_depth = int(name.split('-')[1])

    if network_type == 'resnet':
        if network_depth == 18:
            backbone = resnet18(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        elif network_depth == 34:
            backbone = resnet34(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        elif network_depth == 50:
            backbone = resnet50(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        elif network_depth == 101:
            backbone = resnet101(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
        elif network_depth == 152:
            backbone = resnet152(pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    elif network_type == 'densenet':
        if network_depth == 121:
            backbone = densenet121(pretrained=pretrained, num_classes=num_classes)
        elif network_depth == 169:
            backbone = densenet169(pretrained=pretrained, num_classes=num_classes)
    elif network_type == 'seresnext':
        if network_depth == 50:
            backbone = se_resnext50_32x4d(num_classes=num_classes, pretrained=pretrained)
        elif network_depth == 101:
            backbone = se_resnext101_32x4d(num_classes=num_classes, pretrained=pretrained)
    elif network_type == 'bninception':
        backbone = bninception(num_classes=num_classes, pretrained=pretrained)

    return backbone

# wrapper for train
class Classifier_Wrapper(nn.Module):
    def __init__(self, feature_network, num_classes, label_type, dropout_ratio=0.5):
        super(Classifier_Wrapper, self).__init__()

        self.num_classes = num_classes
        self.label_type = label_type

        self.features = feature_network
        self.num_features = feature_network.classifier.in_features

        self.dropout_ratio = dropout_ratio

        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.num_classes),
            nn.Dropout(p=self.dropout_ratio, inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x

# K-fold CV for test
class Avg_Classifier_Wrapper(nn.Module):
    def __init__(self, checkpoints):
        super(Avg_Classifier_Wrapper, self).__init__()

        self.score_threshold = checkpoints['score'].mean()

        # load checkpoints
        self.networks = []

        for i, row in checkpoints.iterrows():
            backbone = create_feature_network(row['net_type'], pretrained=False)
            network = Classifier_Wrapper(backbone, row['num_classes'], row['label_type'])

            network.load_state_dict(torch.load(row['path']))

            self.networks.append(network)
        

    def forward(self, x):
        outputs = torch.stack([network(x) for network in self.networks])

        # TODO : softmax for single-label, sigmoid for multi-label

        # convert [N, BATCH, CLASSES] to [BATCH, CLASSES, N]
        outputs = torch.permute(1, 2, 0)
        output = outputs.mean(dim=-1)

        return output

# ensemble for test
class Simple_Ensemble_Classifier(nn.Module):
    def __init__(self, checkpoints, alter_threshold):
        super(Simple_Ensemble_Classifier, self).__init__()

        # alternative threshold for multi outputs with false single predictions
        self.alter_threshold = alter_threshold

        # load multi-label classifier
        self.multi_checkpoints = checkpoints[checkpoints['label_type'] == 'multi']
        self.multi_label_classifier = Avg_Classifier_Wrapper(self.multi_checkpoints)

        # load single-label classifiers
        self.single_checkpoints = checkpoints[checkpoints['label_type'] == 'single']
        self.single_groups = self.single_checkpoints.groupby(['label'])

        self.single_labels = []
        self.single_label_classifiers = []

        for name, group in self.single_groups:
            self.single_labels.append(name)

            single_label_classifier = Avg_Classifier_Wrapper(group)
            self.single_label_classifiers.append(single_label_classifier)

    def forward(self, x):
        # apply score_threshold to classifier outputs

        # [BatchSize, N_CLASSES]
        multi_output = self.multi_label_classifier(x)

        single_outputs = [single_label_classifier(x) for single_label_classifier in self.single_label_classifiers]
        single_outputs = torch.squeeze(single_outputs)
        # [N, BatchSize]
        single_results = [torch.gt(single_output, single_label_classifier.score_threshold) for single_output, single_label_classifier in zip(single_outputs, self.single_label_classifiers)]

        # TODO : ensemble multi and single outputs
        for i, single_result in enumerate(single_results):
            single_label = self.single_labels[i]
            