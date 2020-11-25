import numpy as np

class US_pattern:
    def choose_acceleration(self, seed, R):
        self.rng.seed(seed)
        if R == 4:
            num_low_freqs = 0.08
        if R == 8:
            num_low_freqs = 0.04
        acceleration = R
        return num_low_freqs, acceleration

    # ====================================================================
    def generate_US_pattern_pytorch(self, shape, R=4 ,seed=1):
        """
                Args:
                    shape (iterable[int]): The shape of the mask to be created. The shape should have
                        at least 3 dimensions. Samples are drawn along the second last dimension.
                    seed (int, optional): Seed for the random number generator. Setting the seed
                        ensures the same mask is generated each time for the same shape.
                Returns:
                    torch.Tensor: A mask of the specified shape.
        """

        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        self.rng = np.random.RandomState()
        self.rng.seed(seed)
        num_cols = shape[1]
        center_fraction, acceleration = self.choose_acceleration(seed, R)
        # Create the mask
        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = self.rng.uniform(size=num_cols) < prob

        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask = np.repeat(mask[np.newaxis,:], shape[:1], axis=0)

        return mask

    # ====================================================================
    def generate_US_pattern_1D(self, size_2D, R, no_of_training_profs=15):
        # assuming a normal distribution
        # size_2D is taken in 2D, first dimesnion is the readout direction, no undersampling there
        # second dimension gets the undersampling pattern
        # returns a full 2D pattern
        # fills the [-no_of_training_profs/2, +no_of_training_profs/2] k-space center regardless of the random sampling, to mkae sure you have the k-space center

        if R == 1:
            samp_patt = np.tile(np.ones((size_2D[1], 1)).T, (size_2D[0], 1))
            return samp_patt

        mid = np.round(size_2D[1] / 2).astype(int)

        no_of_samples = np.round(size_2D[1] / R).astype(int)
        smps = np.zeros(no_of_samples)

        for i in range(0, no_of_training_profs):
            smps[i] = - np.floor(no_of_training_profs / 2) + i

        ctr = no_of_training_profs  # you already have some samples in the k-space center, count them as well

        while (ctr < no_of_samples):
            smp = np.round(np.random.randn(1) * size_2D[1] / 6)
            if np.abs(smp) < size_2D[1] / 2 - 1:
                if not (smp in smps):
                    smps[ctr] = smp
                    ctr = ctr + 1
        # put the positions into a 1D array
        tmp = np.zeros((size_2D[1], 1))
        inxs = mid + smps.astype(int)
        tmp[inxs.astype(int)] = 1
        # replicate the array to get a 2D pattern image
        samp_patt = np.tile(tmp.T, (size_2D[0], 1))

        return samp_patt

    # ====================================================================
    def generate_opt_US_pattern_1D(self, size_2D, R, max_iter, no_of_training_profs=15):

        if R == 1:
            opt_pt = self.generate_US_pattern_1D(size_2D, R, no_of_training_profs)
            return opt_pt

        mid = np.round(size_2D[1] / 2).astype(int)

        opt_p2s = 0  # 1e10
        opt_pt = []
        opt_ptf = []

        for it in range(0, max_iter):
            pt = self.generate_US_pattern_1D(size_2D, R, no_of_training_profs)
            ptf = np.abs(np.fft.fftshift(np.fft.fft(pt[0, :])))  # just get one readout line
            peak = np.sum(ptf[mid - 1:mid + 1])
            side = np.sum(ptf[0:mid - 2]) + np.sum(ptf[mid + 2:])
            peak2side = peak / side
            #            print(peak2side)

            if peak2side > opt_p2s:
                opt_p2s = peak2side
                opt_pt = pt
                opt_ptf = ptf

        #        print("opt is: " + str(opt_p2s))

        return opt_pt

