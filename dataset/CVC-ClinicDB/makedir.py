import os
import  random
dir=os.getcwd()
image_names=os.listdir(os.path.join(dir,'image'))
image_names.sort()
label_names=os.listdir(os.path.join(dir,'label'))
label_names.sort()

#print(label_names)
#print(image_names)
image_dirs=[]
label_dirs=[]
for image_name,label_name in zip(image_names,label_names):
    #print(image_name,label_name)
    image_dirs.append(os.path.join('image',image_name)+" "+os.path.join('label',label_name))
    #label_dirs.append(os.path.join('label',label_name))
#print(label_dirs)
random.shuffle(image_dirs)
random.shuffle(image_dirs)
N=len(image_dirs)
with open('all_dirs.txt','w') as f:
    for i in range(N):
        f.write(image_dirs[i])
        if i != N-1:
            f.write('\n')
with open('train.txt','w') as f:
    for i in range(int(0.8*N)):
        f.write(image_dirs[i])
        if i != int(0.8*N) - 1:
            f.write('\n')
with open('test.txt','w') as f:
    for i in range(int(0.8 * N),N):
        f.write(image_dirs[i])
        if i != N - 1:
            f.write('\n')
# with open('test.txt','r') as f:
#     lines=f.readlines()
#     for line in lines:
#         #print(line.split())


