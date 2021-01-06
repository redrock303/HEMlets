import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.gridspec as gridspec
joints_left = [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23]
mlineType_local = cv2.LINE_AA
def drawArrow(img, pStart, pEnd, alen, alpha, color, thickness, lineType=None):
	#print pStart,pEnd
	angle = math.atan2(pStart[1] - pEnd[1], pStart[0] - pEnd[0])
	arrowP = pEnd * 0.6 + pStart * 0.4
	cv2.line(img,(int(pStart[0]),int(pStart[1])),(int(pEnd[0]),int(pEnd[1])),color=color,thickness=thickness,lineType=lineType)
	arrow_x = arrowP[0] + alen * math.cos(angle + np.pi * alpha / 180.0)
	arrow_y = arrowP[1] + alen * math.sin(angle + np.pi * alpha / 180.0)

	cv2.line(img,(int(arrowP[0]),int(arrowP[1])),(int(arrow_x),int(arrow_y)),color=color,thickness=thickness,lineType=lineType)
	arrow_x = arrowP[0] + alen * math.cos(angle - np.pi * alpha / 180.0)
	arrow_y = arrowP[1] + alen * math.sin(angle - np.pi * alpha / 180.0)
	cv2.line(img,(int(arrowP[0]),int(arrowP[1])),(int(arrow_x),int(arrow_y)),color=color,thickness=thickness,lineType=lineType)

def drawSkeleton(img,joints,cons,cons_color,winName = 'img'):
    for idx, joint in enumerate(joints):
        x = int(joint[0])
        y = int(joint[1])
        if joint[2] == 0 :
            cv2.circle(img,(x,y),3,(255,0,0),-1)
        elif joint[2] == 1 :
            cv2.circle(img,(x,y),3,(0,0,255),-1)
        else:
            cv2.circle(img,(x,y),3,(0,128,0),-1)
    edge_color_dict = {0:(0,0,255),1:( 0,255,0),2:(255,0,0)}
    for idx,edge in enumerate(cons) :
        i = edge[0]
        j = edge[1]
        joint_i = joints[i]
        joint_j = joints[j]
        if joint_i[0] + joint_i[1] < 5 or joint_j[0] + joint_j[1] < 5:
            continue
        if cons_color is not None:
            color = edge_color_dict[cons_color[idx]]
        else:
            color = (0,0,255)

        drawArrow(img,joint_i,joint_j,7,30,color=color,thickness=1,lineType=mlineType_local)
    # cv2.imshow('img',img)
    # cv2.waitKey()
    return img 
def Draw3DSkeleton(channels,ax,edge,Name=None,fontdict=None,j18_color  = None,image = None):
    edge = np.array(edge)
    I    = edge[:,0]
    J    = edge[:,1]
    LR  = np.ones((edge.shape[0]),dtype=np.int)
    colors = [(0,0,1.0),(0,1.0,0),(1.0,0,0)]
    vals = np.reshape( channels, (-1, 3) )

    vals[:] = vals[:] - vals[0]

    ax.cla()
    ax.view_init(azim=-136,elev=-157)
    ax.invert_yaxis()

    for i in np.arange( len(I) ):
        x,y,z = [np.array([vals[I[i],j],vals[J[i],j]]) for j in range(3)]
        ax.plot(x, -z, y, lw=2, c=colors[j18_color[i]])

    for i in range(16):
        ax.plot([vals[i,0],vals[i,0]+1],[-vals[i,2],-vals[i,2]],[vals[i,1],vals[i,1]],lw=3,c=(0.0,0.8,0.0))		

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    maxAxis = np.max(vals,axis=0)
    minAxis = np.min(vals,axis=0)
    # max_size = np.max(maxAxis-minAxis) / 2 * 1.1
    
    # ax.set_xlim3d([-max_size + xroot, max_size + xroot])
    # ax.set_ylim3d([-max_size + zroot, max_size + zroot])
    # ax.set_zlim3d([-max_size + yroot, max_size + yroot])
    
    max_size = 130
    ax.set_xlim3d([-max_size , max_size ])
    ax.set_ylim3d([-max_size , max_size ])
    ax.set_zlim3d([-max_size , max_size ])
    #print max_size,vals
    if False:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # Get rid of the ticks and tick labels
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])

    # ax.get_xaxis().set_ticklabels([])
    # ax.get_yaxis().set_ticklabels([])
    # ax.set_zticklabels([])
    # ax.set_aspect('equal')



    # px = np.arange(-max_size + xroot, max_size + xroot, 3)   
    # pz = np.arange(-max_size + zroot, max_size + zroot, 3)     
    # px, pz = np.meshgrid(px,pz)   

    # py = np.zeros((px.shape[0],px.shape[1]),dtype=float)
    # py[:,:]=vals[:,1].max()

    # #print px.shape,pz.shape,py.shape
    # ax.axis('off')
    # surf = ax.plot_surface(px, pz, py,color='gray',alpha=0.5)   

    if Name is not None :
        ax.set_title(Name,fontdict=fontdict)
    # ax.set_aspect('equal')

def DrawContant(image,poselist,PoseNameList=None,cons=None,cons_color=None,protocol_data = None,gs1=None):
    
    font = {'family' : 'serif',  
        'color'  : 'darkred',  
        'weight' : 'normal',  
        'size'   : 8,  
            }
    row = 1
    col = len(poselist) + 1
    if protocol_data is not None:
        row += 1
    

    
    # plt.axis('off')

    # draw image 
    axImg=plt.subplot(gs1[0,0])
    axImg.axis('off')
    axImg.set_title('Input Image',fontdict=font)
    image_draw = drawSkeleton(image,poselist[0],cons,cons_color)
    axImg.imshow(image_draw)

    # draw 3d skeleton
    for i in range(len(poselist)):
        
        colIndex = i + 1
        axPose3d=plt.subplot(gs1[0,colIndex],projection='3d')
        Pose3d = poselist[i]
        Pose3d[:,2]=Pose3d[:,2]- 128.0
        # channels,ax,edge,Name=None,fontdict=None,j18_color  = None,image = None
        Draw3DSkeleton(Pose3d,axPose3d,cons,PoseNameList[i],j18_color=cons_color,image = None)
    plt.draw()             
    plt.pause(0.01)

    plt.savefig('./test2.png')
    img_viz = cv2.imread('./test2.png')
    #print('current idx: ',idx)
    input("Press [enter] to continue.")   
    plt.cla()
    return img_viz
    # plt.close()