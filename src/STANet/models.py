import torch
import torch.nn as nn
import torch.nn.functional as F

class PAMBlock(nn.Module):
    """The basic implementation for self-attention block/non-local block"""
    def __init__(self, in_channels, key_channels, value_channels, scale=1, ds=1):
        super(PAMBlock, self).__init__()
        self.scale = scale
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels

        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                     kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                     kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels)
        )
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        x = input
        if self.ds != 1:
            x = self.pool(input)

        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3) // 2

        local_y = []
        local_x = []
        step_h, step_w = h // self.scale, w // self.scale
        for i in range(0, self.scale):
            for j in range(0, self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, h), min(start_y + step_w, w)
                if i == (self.scale - 1):
                    end_x = h
                if j == (self.scale - 1):
                    end_y = w
                local_x += [start_x, end_x]
                local_y += [start_y, end_y]

        value = self.f_value(x)
        query = self.f_query(x)
        key = self.f_key(x)

        value = torch.stack([value[:, :, :, :w], value[:, :, :, w:]], 4)
        query = torch.stack([query[:, :, :, :w], query[:, :, :, w:]], 4)
        key = torch.stack([key[:, :, :, :w], key[:, :, :, w:]], 4)

        local_block_cnt = 2 * self.scale * self.scale

        def self_attention(value_local, query_local, key_local):
            batch_size_new = value_local.size(0)
            h_local, w_local = value_local.size(2), value_local.size(3)
            value_local = value_local.contiguous().view(batch_size_new, self.value_channels, -1)

            query_local = query_local.contiguous().view(batch_size_new, self.key_channels, -1)
            query_local = query_local.permute(0, 2, 1)
            key_local = key_local.contiguous().view(batch_size_new, self.key_channels, -1)

            sim_map = torch.bmm(query_local, key_local)
            sim_map = (self.key_channels ** -.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

            context_local = torch.bmm(value_local, sim_map.permute(0, 2, 1))
            context_local = context_local.view(batch_size_new, self.value_channels, h_local, w_local, 2)
            return context_local

        v_list = [value[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
        v_locals = torch.cat(v_list, dim=0)
        q_list = [query[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
        q_locals = torch.cat(q_list, dim=0)
        k_list = [key[:, :, local_x[i]:local_x[i + 1], local_y[i]:local_y[i + 1]] for i in range(0, local_block_cnt, 2)]
        k_locals = torch.cat(k_list, dim=0)
        context_locals = self_attention(v_locals, q_locals, k_locals)

        context_list = []
        for i in range(0, self.scale):
            row_tmp = []
            for j in range(0, self.scale):
                left = batch_size * (j + i * self.scale)
                right = batch_size * (j + i * self.scale) + batch_size
                tmp = context_locals[left:right]
                row_tmp.append(tmp)
            context_list.append(torch.cat(row_tmp, 3))

        context = torch.cat(context_list, 2)
        context = torch.cat([context[:, :, :, :, 0], context[:, :, :, :, 1]], 3)

        if self.ds != 1:
            context = F.interpolate(context, [h * self.ds, 2 * w * self.ds])

        return context

class PAM(nn.Module):
    """PAM (Position Attention Module)"""
    def __init__(self, in_channels, out_channels, sizes=([1]), ds=1):
        super(PAM, self).__init__()
        self.group = len(sizes)
        self.stages = []
        self.ds = ds
        self.value_channels = out_channels
        self.key_channels = out_channels // 8

        self.stages = nn.ModuleList(
            [PAMBlock(in_channels, self.key_channels, self.value_channels, size, self.ds)
             for size in sizes])
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_channels * self.group, out_channels, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = []
        for i in range(0, len(priors)):
            context += [priors[i]]
        output = self.conv_bn(torch.cat(context, 1))
        return output

class BAM(nn.Module):
    """Basic self-attention module"""
    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8
        self.activation = activation
        self.ds = ds
        self.pool = nn.AvgPool2d(self.ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        x = self.pool(input)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        energy = (self.key_channel ** -.5) * energy
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)
        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input
        return out

class CDSA(nn.Module):
    """Self attention module for change detection"""
    def __init__(self, in_c, ds=1, mode='BAM'):
        super(CDSA, self).__init__()
        self.in_C = in_c
        self.ds = ds
        self.mode = mode
        if self.mode == 'BAM':
            self.Self_Att = BAM(self.in_C, ds=self.ds)
        elif self.mode == 'PAM':
            self.Self_Att = PAM(in_channels=self.in_C, out_channels=self.in_C, sizes=[1, 2, 4, 8], ds=self.ds)
        elif self.mode == 'None':
            self.Self_Att = nn.Identity()

    def forward(self, x1, x2):
        height = x1.shape[3]
        x = torch.cat((x1, x2), 3)
        x = self.Self_Att(x)
        return x[:, :, :, 0:height], x[:, :, :, height:]

class STANet(nn.Module):
    """STANet for multiclass change detection"""
    def __init__(self, input_channels=3, hidden_channels=32, num_classes=3, attention_mode='BAM'):
        super(STANet, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.attention_mode = attention_mode

        # Encoder/Backbone layers
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels*2)
        self.conv3 = nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(hidden_channels*4)

        # Self-attention module
        self.sa = CDSA(in_c=hidden_channels*4, ds=1, mode=attention_mode)

        # Decoder layers
        self.upconv1 = nn.ConvTranspose2d(hidden_channels*8, hidden_channels*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upbn1 = nn.BatchNorm2d(hidden_channels*4)
        self.upconv2 = nn.ConvTranspose2d(hidden_channels*4, hidden_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upbn2 = nn.BatchNorm2d(hidden_channels*2)
        self.upconv3 = nn.ConvTranspose2d(hidden_channels*2, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upbn3 = nn.BatchNorm2d(hidden_channels)

        # Final classification layer
        self.final_conv = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def encode(self, x):
        # Encoder path
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        return x

    def decode(self, x):
        # Decoder path
        x = F.relu(self.upbn1(self.upconv1(x)))
        x = F.relu(self.upbn2(self.upconv2(x)))
        x = F.relu(self.upbn3(self.upconv3(x)))
        return x

    def forward(self, x1, x2):
        # Encode both images
        feat1 = self.encode(x1)
        feat2 = self.encode(x2)

        # Apply self-attention
        att1, att2 = self.sa(feat1, feat2)

        # Concatenate attended features
        combined = torch.cat([att1, att2], dim=1)

        # Decode
        decoded = self.decode(combined)

        # Final classification
        out = self.final_conv(decoded)

        # Only apply softmax during inference
        if not self.training:
            out = self.softmax(out)

        return out