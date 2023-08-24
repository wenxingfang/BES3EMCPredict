import os
import random
path = '/hpcfs/juno/junogpu/junoML/fangwx/BES3/emc/recoverHitE/'
files = os.listdir(path)
with open('train.txt','w') as f:
    for file in files:
        if '.h5' not in file:continue
        file = file.replace('\n','')
        f.write('%s/%s\n'%(path,file))
