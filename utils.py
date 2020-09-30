def FT(x, normalize=False):
    # inp: [nx, ny]
    # out: [nx, ny]
    if normalize:
        return np.fft.fftshift(np.fft.fft2(x, axes=(0, 1)), axes=(0, 1)) / np.sqrt(252 * 308)
    else:
        return np.fft.fftshift(np.fft.fft2(x, axes=(0, 1)), axes=(0, 1))


def tFT(x, normalize=False):
    # inp: [nx, ny]
    # out: [nx, ny]
    if normalize:
        return np.fft.ifft2(np.fft.ifftshift(x, axes=(0, 1)), axes=(0, 1)) * np.sqrt(252 * 308)
    else:
        return np.fft.ifft2(np.fft.ifftshift(x, axes=(0, 1)), axes=(0, 1))


def UFT(x, uspat, normalize=False):
    # inp: [nx, ny], [nx, ny]
    # out: [nx, ny]

    return uspat * FT(x, normalize)


def tUFT(x, uspat, normalize=False):
    # inp: [nx, ny], [nx, ny]
    # out: [nx, ny]
    return tFT(uspat * x, normalize)


def calc_rmse(rec, imorig):
    return 100 * np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))))

def normalize_tensor(input_tens):
    i_max = input_tens.max()
    i_min = input_tens.min()
    input_tens = (input_tens-i_min)/(i_max-i_min)
    return input_tens