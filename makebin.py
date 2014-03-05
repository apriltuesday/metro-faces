# make binary feat vector
import numpy as np
import scipy.io as io

mat = io.loadmat('../data/April_full_fixedTime.mat')
images = mat['images'][:,0]
times = mat['timestamps']
faces = mat['faces']
faces = np.ceil(faces)

io.savemat('../data/April_full_fixedTime_binary.mat', {'images': images,
                                              'timestamps': times,
                                              'faces': faces}
           )
