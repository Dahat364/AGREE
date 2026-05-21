import torch
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import clip
import math

def teacher_model():

    # model = HKD_Teacher()
    # model_path = '/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/weights/HKD-IAA/Pre-trained teacher.pth'
    # checkpoint_model = torch.load(model_path)
    # model.load_state_dict(checkpoint_model['model'])
    # return model
    model = HKD_Teacher()
    model_path = '/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/weights/HKD-IAA/Pre-trained teacher.pth'
    checkpoint = torch.load(model_path)
    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()

    # 保留那些 shape 匹配的参数
    matched_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                matched_dict[k] = v
            else:
                print(f"[跳过] 参数 {k}: 预训练 shape={v.shape}，模型 shape={model_dict[k].shape}")
        else:
            print(f"[跳过] 参数 {k} 不在当前模型中")

    # 加载匹配的部分参数
    model.load_state_dict(matched_dict, strict=False)
    print(f"✅ 已加载 {len(matched_dict)} 个匹配参数，跳过 {len(pretrained_dict) - len(matched_dict)} 个不匹配参数")
    return model

class HKD_Teacher(nn.Module):
    def __init__(self):
        super(HKD_Teacher, self).__init__()
        self.CLIP_T, _ = clip.load("/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/weights/HKD-IAA/ViT-B-16.pt")
        self.MultimodalTransformer = MultimodalTransformer(visual_dim=512, physiological_dim=512, num_heads=2, hidden_dim=512, num_layers=2,
                              num_classes=2)
        self.head_t = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self,x,text):
        image_features = self.CLIP_T.encode_image(x)
        text_features  = self.CLIP_T.encode_text(text)
        _, Fused_feature = self.MultimodalTransformer(image_features.unsqueeze(1).float(), text_features.unsqueeze(1).float())
        score = self.head_t(Fused_feature)
        return score,Fused_feature,image_features,text_features


class HKD_Student(nn.Module):
    def __init__(self):
        super(HKD_Student, self).__init__()
        self.teacher_model= teacher_model()
        for p in self.parameters():
            p.requires_grad = False
        self.student_model, _ = clip.load("/home/dmc/12TB_ZHZ68JLF/A_PROJECT_IAA/weights/HKD-IAA/RN50.pt")
        self.unimodel_fea_ext = UnimodalTransformer(F_dim=512, num_heads=2, hidden_dim=512, num_layers=2,num_classes=2)
        self.head_s = nn.Sequential(
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.fc = nn.Sequential(nn.Linear(1024, 512),)

        dim_motion = 32

        self.fc_t = Mapping(512, dim_motion)
        self.fc_s = Mapping(512, dim_motion)

        self.Dictionary = Dictionary(dim_motion)
        self.sim_matrice=CosineSimilarityMatrixTranspose()

    def forward(self,x,text):
        T_pred, Fused_feature,image_features,text_features=self.teacher_model(x,text)
        image_features_S = self.student_model.encode_image(x)
        image_features_S = self.fc(image_features_S.float())
        Inheritance_F = self.unimodel_fea_ext(image_features_S)

        weight_t = self.fc_t(Fused_feature)
        weight_s = self.fc_s(Inheritance_F)

        h_t = self.Dictionary(weight_t)
        h_s = self.Dictionary(weight_s)

        relation_t = self.sim_matrice(h_t)
        relation_s = self.sim_matrice(h_s)

        S_pred = self.head_s(h_s)

        return T_pred, S_pred, relation_t, relation_s,h_t,h_s

class Dictionary(nn.Module):
    def __init__(self, motion_dim):
        super(Dictionary, self).__init__()

        self.weight = nn.Parameter(torch.randn(512, motion_dim))

    def forward(self, input):
        # input: (bs*t) x 512

        weight = self.weight + 1e-8
        Q, R = torch.qr(weight)  # get eignvector, orthogonal [n1, n2, n3, n4]

        if input is None:
            return Q
        else:
            input_diag = torch.diag_embed(input)  # alpha, diagonal matrix
            out = torch.matmul(input_diag, Q.T)
            out = torch.sum(out, dim=1)

            return out

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')

class Mapping(nn.Module):
    def __init__(self, input_dim, output_dim, layers=4):
        super(Mapping, self).__init__()
        # Create a series of EqualLinear layers
        self.layers = nn.ModuleList([EqualLinear(input_dim, input_dim) for _ in range(layers - 1)])
        self.layers.append(EqualLinear(input_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)

        # Apply feed forward network
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x

class UnimodalTransformer(nn.Module):
    def __init__(self, F_dim, num_heads, hidden_dim, num_layers, num_classes):
        super(UnimodalTransformer, self).__init__()
        self.TB_1 = TransformerEncoderBlock(F_dim, num_heads, hidden_dim, num_layers)

    def forward(self, features):
        Inheritance_F = self.TB_1(features)
        return Inheritance_F

class MultimodalTransformer(nn.Module):
    def __init__(self, visual_dim, physiological_dim, num_heads, hidden_dim, num_layers, num_classes):
        super(MultimodalTransformer, self).__init__()
        self.visual_encoder = TransformerEncoderBlock(visual_dim, num_heads, hidden_dim, num_layers)
        self.physiological_encoder = TransformerEncoderBlock(physiological_dim, num_heads, hidden_dim, num_layers)
        self.cross_attention_v = nn.MultiheadAttention(visual_dim, num_heads)
        self.cross_attention_p = nn.MultiheadAttention(physiological_dim, num_heads)
        self.gated_attention = nn.Linear(visual_dim + physiological_dim, 1)
        self.fc = nn.Linear(visual_dim, num_classes)

    def forward(self, visual_features, physiological_features):
        visual_encoded = self.visual_encoder(visual_features)
        physiological_encoded = self.physiological_encoder(physiological_features)

        cross_attention_output_v, _ = self.cross_attention_v(physiological_encoded.permute(1, 0, 2),
                                                             visual_encoded.permute(1, 0, 2),
                                                             visual_encoded.permute(1, 0, 2))

        cross_attention_output_v = cross_attention_output_v.permute(1, 0, 2)

        cross_attention_output_p, _ = self.cross_attention_p(visual_encoded.permute(1, 0, 2),
                                                             physiological_encoded.permute(1, 0, 2),
                                                             physiological_encoded.permute(1, 0, 2))
        cross_attention_output_p = cross_attention_output_p.permute(1, 0, 2)

        gating_input = torch.cat((cross_attention_output_v, cross_attention_output_p), dim=2)
        gating_coefficients = torch.sigmoid(self.gated_attention(gating_input))
        combined_attention = (gating_coefficients * cross_attention_output_v) + (
                    (1 - gating_coefficients) * cross_attention_output_p)
        combined_attention = combined_attention.squeeze(1)

        output = self.fc(combined_attention)  # Use only the final timestep for classification
        return output, combined_attention

class CosineSimilarityMatrixTranspose(torch.nn.Module):
    def __init__(self):
        super(CosineSimilarityMatrixTranspose, self).__init__()

    def forward(self, matrix):
        transpose_matrix = matrix.t()
        similarity_matrix = torch.mm(matrix, transpose_matrix)
        norm_matrix = torch.norm(matrix, dim=1, keepdim=True)
        norm_transpose = torch.norm(transpose_matrix, dim=0, keepdim=True)
        similarity_matrix.div_(torch.mm(norm_matrix, norm_transpose))
        return similarity_matrix
