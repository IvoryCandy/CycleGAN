import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, activation='relu', batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.batch_norm = batch_norm
        self.bn = nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = nn.ReLU(True)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(self.conv(x))
        else:
            x = self.conv(x)

        if self.activation == 'relu':
            return self.relu(x)
        elif self.activation == 'lrelu':
            return self.lrelu(x)
        elif self.activation == 'tanh':
            return self.tanh(x)
        elif self.activation == 'no_act':
            return x


class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu',
                 batch_norm=True):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, output_padding)
        self.batch_norm = batch_norm
        self.bn = nn.InstanceNorm2d(output_size)
        self.activation = activation
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.batch_norm:
            x = self.bn(self.deconv(x))
        else:
            x = self.deconv(x)

        if self.activation == 'relu':
            return self.relu(x)
        elif self.activation == 'lrelu':
            return self.lrelu(x)
        elif self.activation == 'tanh':
            return self.tanh(x)
        elif self.activation == 'no_act':
            return x


class ResnetBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=0):
        super(ResnetBlock, self).__init__()
        self.resnet_block = nn.Sequential(nn.ReflectionPad2d(1),
                                          nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding),
                                          nn.InstanceNorm2d(num_filter),
                                          nn.ReflectionPad2d(1),
                                          nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding),
                                          nn.InstanceNorm2d(num_filter))

    def forward(self, x):
        x = self.resnet_block(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_dim, num_filter, output_dim, num_resnet):
        super(Generator, self).__init__()

        # Reflection padding
        self.padding = nn.ReflectionPad2d(3)
        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=7, stride=1, padding=0)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        # Resnet blocks
        self.resnet_blocks = []
        for i in range(num_resnet):
            self.resnet_blocks.append(ResnetBlock(num_filter * 4))
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 4, num_filter * 2)
        self.deconv2 = DeconvBlock(num_filter * 2, num_filter)
        self.deconv3 = ConvBlock(num_filter, output_dim, kernel_size=7, stride=1, padding=0, activation='tanh',
                                 batch_norm=False)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(self.padding(x))
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        # Resnet blocks
        res = self.resnet_blocks(enc3)
        # Decoder
        dec1 = self.deconv1(res)
        dec2 = self.deconv2(dec1)
        x = self.deconv3(self.padding(dec2))
        return x

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                nn.init.normal(m.deconv.weight, mean, std)
            if isinstance(m, ResnetBlock):
                nn.init.normal(m.conv.weight, mean, std)
                nn.init.constant(m.conv.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(input_dim, num_filter, kernel_size=4, stride=2, padding=1, activation='lrelu', batch_norm=False),
            ConvBlock(num_filter, num_filter * 2, kernel_size=4, stride=2, padding=1, activation='lrelu'),
            ConvBlock(num_filter * 2, num_filter * 4, kernel_size=4, stride=2, padding=1, activation='lrelu'),
            ConvBlock(num_filter * 4, num_filter * 8, kernel_size=4, stride=1, padding=1, activation='lrelu'),
            ConvBlock(num_filter * 8, output_dim, kernel_size=4, stride=1, padding=1, activation='no_act',
                      batch_norm=False))

    def forward(self, x):
        x = self.conv_blocks(x)
        return x

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                nn.init.normal(m.conv.weight, mean, std)
