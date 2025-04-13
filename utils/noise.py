import torch
import numpy as np
import torch.nn as nn

class Noise:

    def add_Poisson_noise(img, min_exponent=2.0, max_exponent=4.0):
        # First normalize the image to [0, 1] range with proper rounding
        img = np.clip((img * 255.0).round(), 0, 255) / 255.
        
        # Calculate the scaling factor (vals) based on the exponent range
        vals = 10 ** (random.random() * (max_exponent - min_exponent) + min_exponent)
        
        # Apply either color or grayscale Poisson noise based on probability
        if random.random() < 0.5:
          # Color Poisson noise (applied to each channel independently)
          img = np.random.poisson(img * vals).astype(np.float32) / vals
        else:
          # Grayscale Poisson noise
          img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
          img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
          noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
          img += noise_gray[:, :, np.newaxis]
        
        # Clip values to ensure they remain in valid range
        img = np.clip(img, 0.0, 1.0)
        return img
    
    def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
        noise_level = random.randint(noise_level1, noise_level2)
        rnum = np.random.rand()
        if rnum > 0.6:  # add color Gaussian noise
          img = img + np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
        elif rnum < 0.4:  # add grayscale Gaussian noise
          img = img + np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
        else:  # add  noise
          L = noise_level2 / 255.
          D = np.diag(np.random.rand(3))
          U = orth(np.random.rand(3, 3))
          conv = np.dot(np.dot(np.transpose(U), D), U)
          img = img + np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return img
    
    def add_speckle_noise(img, noise_level1=2, noise_level2=25):
        noise_level = random.randint(noise_level1, noise_level2)
        img = np.clip(img, 0.0, 1.0)
        rnum = random.random()
        if rnum > 0.6:
          img += img * np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
        elif rnum < 0.4:
          img += img * np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
        else:
          L = noise_level2 / 255.
          D = np.diag(np.random.rand(3))
          U = orth(np.random.rand(3, 3))
          conv = np.dot(np.dot(np.transpose(U), D), U)
          img += img * np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)
        return img
        
    def salt_and_pepper_noise(img, prob=0.05):
        """
        Add salt-and-pepper (impulse) noise to an image.
        
        Args:
          img (ndarray): Input image in [0, 1], shape HxWxC or HxW.
          prob (float): Probability of a pixel being corrupted by noise.
                        Typically in the range [0, 1]. Default 0.05.
        Returns:
          ndarray: Image with salt-and-pepper noise.
        """
        output = img.copy()
        # If single-channel, expand dims for consistent handling
        if output.ndim == 2:
          output = np.expand_dims(output, axis=2)
        
        # Generate a random matrix in [0, 1] with same height and width
        # The shape is (H, W, 1) but we broadcast for channels if needed
        rand_matrix = np.random.rand(output.shape[0], output.shape[1], 1)
        
        # Salt mask: pixels that become 1
        salt_mask = rand_matrix < (prob / 2.0)
        # Pepper mask: pixels that become 0
        pepper_mask = (rand_matrix >= (prob / 2.0)) & (rand_matrix < prob)
        
        # Fix: use the first channel index of the mask to index all channels
        output[salt_mask[:, :, 0], :] = 1.0
        output[pepper_mask[:, :, 0], :] = 0.0
        
        # If the original was single-channel, squeeze back
        if img.ndim == 2:
          output = np.squeeze(output, axis=2)
        
        return np.clip(output, 0.0, 1.0)
        
    def adjust_brightness_torch(img, factor=1.0, beta=0.0):
        """
        Adjust image brightness in a differentiable manner.
        
        Args:
          img (torch.Tensor): Input image tensor with shape (N, C, H, W) and values in [0, 1].
          factor (float): Multiplicative brightness (and contrast) factor.
          beta (float): Additive brightness offset.
        
        Returns:
          torch.Tensor: Brightness-adjusted image.
        """
        # Ensure that img is a float tensor and perform the linear transformation
        img = img.float() * factor + beta
        # Clamp to maintain valid range (for instance, [0, 1] if working in normalized space)
        img = torch.clamp(img, 0, 1)
        return img
        
    def pixelate(x, pixel_size=4, severity=None):
        """
        Apply pixelation effect to an image without changing its dimensions.
        
        Args:
          x: Input image (numpy array)
          pixel_size: Size of pixels for pixelation effect (higher = more pixelated)
          severity: Alternative way to specify pixelation level (1-5 scale, overrides pixel_size if provided)
        
        Returns:
          Pixelated image with original dimensions
        """
        # Get original dimensions
        h, w = x.shape[:2]
        
        # Convert numpy array to PIL Image
        x_pil = Image.fromarray((x * 255).astype(np.uint8))
        
        # If severity is provided, use it to determine pixel_size
        if severity is not None:
          c = [0.6, 0.5, 0.4, 0.3, 0.25][min(severity, 5) - 1]
          reduced_size = (int(w * c), int(h * c))
        else:
          # Calculate reduced size based on pixel_size
          reduced_size = (max(1, w // pixel_size), max(1, h // pixel_size))
        
        # Down-sample then up-sample to create pixelation effect
        x_pil = x_pil.resize(reduced_size, Image.BOX)
        x_pil = x_pil.resize((w, h), Image.BOX)
        
        # Convert back to numpy array in range [0,1]
        result = np.array(x_pil).astype(np.float32) / 255.0
        
        return result
        
    def saturate(x, severity=1):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
        
        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)
        
        return np.clip(x, 0, 1) * 255
    
    y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                    55], [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                    77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
    
    y_table = nn.Parameter(torch.from_numpy(y_table))
    c_table = np.empty((8, 8), dtype=np.float32)
    c_table.fill(99)
    c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                                [24, 26, 56, 99], [47, 66, 99, 99]]).T
    c_table = nn.Parameter(torch.from_numpy(c_table))

    # 1. RGB -> YCbCr
    class rgb_to_ycbcr_jpeg(nn.Module):
        """ Converts RGB image to YCbCr
        Input:
            image(tensor): batch x 3 x height x width
        Outpput:
            result(tensor): batch x height x width x 3
        """
        def __init__(self):
            super(rgb_to_ycbcr_jpeg, self).__init__()
            matrix = np.array(
                [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
                 [0.5, -0.418688, -0.081312]], dtype=np.float32).T
            self.shift = nn.Parameter(torch.tensor([0., 128., 128.]))
            self.matrix = nn.Parameter(torch.from_numpy(matrix))
    
        def forward(self, image):
            image = image.permute(0, 2, 3, 1)
            result = torch.tensordot(image, self.matrix, dims=1) + self.shift
            result.view(image.shape)
            return result
        
    # 2. Chroma subsampling
    class chroma_subsampling(nn.Module):
        """ Chroma subsampling on CbCv channels
        Input:
            image(tensor): batch x height x width x 3
        Output:
            y(tensor): batch x height x width
            cb(tensor): batch x height/2 x width/2
            cr(tensor): batch x height/2 x width/2
        """
        def __init__(self):
            super(chroma_subsampling, self).__init__()
    
        def forward(self, image):
            image_2 = image.permute(0, 3, 1, 2).clone()
            avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                    count_include_pad=False)
            cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
            cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
            cb = cb.permute(0, 2, 3, 1)
            cr = cr.permute(0, 2, 3, 1)
            return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)
    
    # 3. Block splitting
    class block_splitting(nn.Module):
        """ Splitting image into patches
        Input:
            image(tensor): batch x height x width
        Output: 
            patch(tensor):  batch x h*w/64 x h x w
        """
        def __init__(self):
            super(block_splitting, self).__init__()
            self.k = 8
    
        def forward(self, image):
            height, width = image.shape[1:3]
            batch_size = image.shape[0]
            image_reshaped = image.view(batch_size, height // self.k, self.k, -1, self.k)
            image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
            return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)
        
    # 4. DCT
    class dct_8x8(nn.Module):
        """ Discrete Cosine Transformation
        Input:
            image(tensor): batch x height x width
        Output:
            dcp(tensor): batch x height x width
        """
        def __init__(self):
            super(dct_8x8, self).__init__()
            tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
            for x, y, u, v in itertools.product(range(8), repeat=4):
                tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                    (2 * y + 1) * v * np.pi / 16)
            alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
            #
            self.tensor =  nn.Parameter(torch.from_numpy(tensor).float())
            self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float() )
            
        def forward(self, image):
            image = image - 128
            result = self.scale * torch.tensordot(image, self.tensor, dims=2)
            result.view(image.shape)
            return result

    # 5. Quantization
    class y_quantize(nn.Module):
        """ JPEG Quantization for Y channel
        Input:
            image(tensor): batch x height x width
            rounding(function): rounding function to use
            factor(float): Degree of compression
        Output:
            image(tensor): batch x height x width
        """
        def __init__(self, rounding, factor=1):
            super(y_quantize, self).__init__()
            self.rounding = rounding
            self.factor = factor
            self.y_table = y_table
    
        def forward(self, image):
            image = image.float() / (self.y_table * self.factor)
            image = self.rounding(image)
            return image


    class c_quantize(nn.Module):
        """ JPEG Quantization for CrCb channels
        Input:
            image(tensor): batch x height x width
            rounding(function): rounding function to use
            factor(float): Degree of compression
        Output:
            image(tensor): batch x height x width
        """
        def __init__(self, rounding, factor=1):
            super(c_quantize, self).__init__()
            self.rounding = rounding
            self.factor = factor
            self.c_table = c_table
    
        def forward(self, image):
            image = image.float() / (self.c_table * self.factor)
            image = self.rounding(image)
            return image

    class compress_jpeg(nn.Module):
        """ Full JPEG compression algortihm
        Input:
            imgs(tensor): batch x 3 x height x width
            rounding(function): rounding function to use
            factor(float): Compression factor
        Ouput:
            compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        """
        def __init__(self, rounding=torch.round, factor=1):
            super(compress_jpeg, self).__init__()
            self.l1 = nn.Sequential(
                rgb_to_ycbcr_jpeg(),
                chroma_subsampling()
            )
            self.l2 = nn.Sequential(
                block_splitting(),
                dct_8x8()
            )
            self.c_quantize = c_quantize(rounding=rounding, factor=factor)
            self.y_quantize = y_quantize(rounding=rounding, factor=factor)
    
        def forward(self, image):
            y, cb, cr = self.l1(image*255)
            components = {'y': y, 'cb': cb, 'cr': cr}
            for k in components.keys():
                comp = self.l2(components[k])
                if k in ('cb', 'cr'):
                    comp = self.c_quantize(comp)
                else:
                    comp = self.y_quantize(comp)
    
                components[k] = comp
    
            return components['y'], components['cb'], components['cr']
    
    # -5. Dequantization
    class y_dequantize(nn.Module):
        """ Dequantize Y channel
        Inputs:
            image(tensor): batch x height x width
            factor(float): compression factor
        Outputs:
            image(tensor): batch x height x width
        """
        def __init__(self, factor=1):
            super(y_dequantize, self).__init__()
            self.y_table = y_table
            self.factor = factor
    
        def forward(self, image):
            return image * (self.y_table * self.factor)


    class c_dequantize(nn.Module):
        """ Dequantize CbCr channel
        Inputs:
            image(tensor): batch x height x width
            factor(float): compression factor
        Outputs:
            image(tensor): batch x height x width
        """
        def __init__(self, factor=1):
            super(c_dequantize, self).__init__()
            self.factor = factor
            self.c_table = c_table
    
        def forward(self, image):
            return image * (self.c_table * self.factor)

    # -4. Inverse DCT
    class idct_8x8(nn.Module):
        """ Inverse discrete Cosine Transformation
        Input:
            dcp(tensor): batch x height x width
        Output:
            image(tensor): batch x height x width
        """
        def __init__(self):
            super(idct_8x8, self).__init__()
            alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
            self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
            tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
            for x, y, u, v in itertools.product(range(8), repeat=4):
                tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                    (2 * v + 1) * y * np.pi / 16)
            self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
    
        def forward(self, image):
            image = image * self.alpha
            result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
            result.view(image.shape)
            return result

    # -3. Block joining
    class block_merging(nn.Module):
        """ Merge pathces into image
        Inputs:
            patches(tensor) batch x height*width/64, height x width
            height(int)
            width(int)
        Output:
            image(tensor): batch x height x width
        """
        def __init__(self):
            super(block_merging, self).__init__()
            
        def forward(self, patches, height, width):
            k = 8
            batch_size = patches.shape[0]
            image_reshaped = patches.view(batch_size, height//k, width//k, k, k)
            image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
            return image_transposed.contiguous().view(batch_size, height, width)

    # -2. Chroma upsampling
    class chroma_upsampling(nn.Module):
        """ Upsample chroma layers
        Input: 
            y(tensor): y channel image
            cb(tensor): cb channel
            cr(tensor): cr channel
        Ouput:
            image(tensor): batch x height x width x 3
        """
        def __init__(self):
            super(chroma_upsampling, self).__init__()
    
        def forward(self, y, cb, cr):
            def repeat(x, k=2):
                height, width = x.shape[1:3]
                x = x.unsqueeze(-1)
                x = x.repeat(1, 1, k, k)
                x = x.view(-1, height * k, width * k)
                return x
    
            cb = repeat(cb)
            cr = repeat(cr)
            
            return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)

    # -1: YCbCr -> RGB
    class ycbcr_to_rgb_jpeg(nn.Module):
        """ Converts YCbCr image to RGB JPEG
        Input:
            image(tensor): batch x height x width x 3
        Outpput:
            result(tensor): batch x 3 x height x width
        """
        def __init__(self):
            super(ycbcr_to_rgb_jpeg, self).__init__()
    
            matrix = np.array(
                [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
                dtype=np.float32).T
            self.shift = nn.Parameter(torch.tensor([0, -128., -128.]))
            self.matrix = nn.Parameter(torch.from_numpy(matrix))
    
        def forward(self, image):
            result = torch.tensordot(image + self.shift, self.matrix, dims=1)
            result.view(image.shape)
            return result.permute(0, 3, 1, 2)

    class decompress_jpeg(nn.Module):
        """ Full JPEG decompression algortihm
        Input:
            compressed(dict(tensor)): batch x h*w/64 x 8 x 8
            rounding(function): rounding function to use
            factor(float): Compression factor
        Ouput:
            image(tensor): batch x 3 x height x width
        """
        def __init__(self, height, width, rounding=torch.round, factor=1):
            super(decompress_jpeg, self).__init__()
            self.c_dequantize = c_dequantize(factor=factor)
            self.y_dequantize = y_dequantize(factor=factor)
            self.idct = idct_8x8()
            self.merging = block_merging()
            self.chroma = chroma_upsampling()
            self.colors = ycbcr_to_rgb_jpeg()
            
            self.height, self.width = height, width
            
        def forward(self, y, cb, cr):
            components = {'y': y, 'cb': cb, 'cr': cr}
            for k in components.keys():
                if k in ('cb', 'cr'):
                    comp = self.c_dequantize(components[k])
                    height, width = int(self.height/2), int(self.width/2)                
                else:
                    comp = self.y_dequantize(components[k])
                    height, width = self.height, self.width                
                comp = self.idct(comp)
                components[k] = self.merging(comp, height, width)
                #
            image = self.chroma(components['y'], components['cb'], components['cr'])
            image = self.colors(image)
            
            image = torch.min(255*torch.ones_like(image),
                              torch.max(torch.zeros_like(image), image))
            return image/255

    def round_only_at_0(x):
        cond = (torch.abs(x) < 0.5).float()
        return cond * (x ** 3) + (1 - cond) * x
    
    def quality_to_factor(quality):
        """ Calculate factor corresponding to quality
        Input:
            quality(float): Quality for jpeg compression
        Output:
            factor(float): Compression factor
        """
        if quality < 50:
            quality = 5000. / quality
        else:
            quality = 200. - quality*2
        return quality / 100.
        
    def jpeg_compress_decompress(image,rounding=round_only_at_0,quality=80):
    
            height, width = image.shape[2:4]
            
            factor = quality_to_factor(quality)
        
            compress = compress_jpeg(rounding=rounding, factor=factor).cuda()
            decompress = decompress_jpeg(height, width, rounding=rounding, factor=factor).cuda()
        
            y, cb, cr = compress(image)
            recovered = decompress(y, cb, cr)
        
            return recovered.contiguous()
    
    def gaussian_blur(x, severity=1):
        c = [1, 2, 3, 4, 6][severity - 1]
        # Normalize input, apply gaussian filter, then scale back up.
        x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
        return np.clip(x, 0, 1) * 255
        
        
    def spatter(x, severity=1):
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
           (0.65, 0.3, 3, 0.68, 0.6, 0),
           (0.65, 0.3, 2, 0.68, 0.5, 0),
           (0.65, 0.3, 1, 0.65, 1.5, 1),
           (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
        x = np.array(x, dtype=np.float32) / 255.
        
        liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
        
        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        if c[5] == 0:
          liquid_layer = (liquid_layer * 255).astype(np.uint8)
          dist = 255 - cv2.Canny(liquid_layer, 50, 150)
          dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
          _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
          dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
          dist = cv2.equalizeHist(dist)
          ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
          dist = cv2.filter2D(dist, cv2.CV_8U, ker)
          dist = cv2.blur(dist, (3, 3)).astype(np.float32)
        
          m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
          m /= np.max(m, axis=(0, 1))
          m *= c[4]
        
          # water is pale turqouise
          color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                  238 / 255. * np.ones_like(m[..., :1]),
                                  238 / 255. * np.ones_like(m[..., :1])), axis=2)
        
          color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
          x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)
        
          return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
        else:
          m = np.where(liquid_layer > c[3], 1, 0)
          m = gaussian(m.astype(np.float32), sigma=c[4])
          m[m < 0.8] = 0
        
          # mud brown
          color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                  42 / 255. * np.ones_like(x[..., :1]),
                                  20 / 255. * np.ones_like(x[..., :1])), axis=2)
        
          color *= m[..., np.newaxis]
          x *= (1 - m[..., np.newaxis])
        
          return np.clip(x + color, 0, 1) * 255
        
    
    def disk(radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
          L = np.arange(-8, 8 + 1)
          ksize = (3, 3)
        else:
          L = np.arange(-radius, radius + 1)
          ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)
        
        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
        
        
    def defocus_blur(x, severity=1):
        
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
        
        x = np.array(x) / 255.
        kernel = disk(radius=c[0], alias_blur=c[1])
        
        channels = []
        for d in range(3):
          channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  
        
        return np.clip(channels, 0, 1) * 255
