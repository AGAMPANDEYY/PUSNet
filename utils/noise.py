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
