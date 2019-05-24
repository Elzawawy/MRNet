import numpy as np
import os

def apply_three_slice_trans(data_path, trans_path, dilation=2):
    npy_files = [f for f in sorted(os.listdir(data_path)) if f[-4:]=='.npy']
    temp = []
    new_stacks = []
    print("Found files:",npy_files[0],npy_files[1],",.....")
    print("Applying Transformation:============")
    for file in npy_files:
        orig_stack=np.load(data_path+'/'+file)
        mid_slice_idx=orig_stack.shape[0]//2
        mid_slice = orig_stack[mid_slice_idx]
        lower_slice = orig_stack[mid_slice_idx-dilation]
        upper_slice = orig_stack[mid_slice_idx+dilation]
        temp.extend([lower_slice,mid_slice,upper_slice])
        new_stacks.append(np.array(temp))
        temp.clear()
    print("Done with Transformation:============")
    print("Creating",trans_path,"....")
    np.save(trans_path,np.array(new_stacks))

def apply_five_class_labels(abn_label_path, acl_label_path, men_label_path, trans_path):
    abnormal = np.genfromtxt(abn_label_path, delimiter=',')
    acl = np.genfromtxt(acl_label_path, delimiter=',')
    meniscus = np.genfromtxt(men_label_path, delimiter=',')
    labels = np.zeros((abnormal.shape[0],5))
    for i in range(labels.shape[0]):
        if(abnormal[i,1] == 1 and acl[i,1] == 1 and meniscus[i,1] == 1):
            labels[i,4] = 1
        elif(abnormal[i,1] == 1 and meniscus[i,1] == 1):
            labels[i,3] = 1
        elif(abnormal[i,1] == 1 and acl[i,1] == 1):
            labels[i,2] = 1
        elif(abnormal[i,1] == 1):
            labels[i,1] = 1
        elif(abnormal[i,1] == 0 and acl[i,1] == 0 and meniscus[i,1] == 0):
            labels[i,0] = 1
    np.save(trans_path,labels)



    
    
