import numpy as np
import cv2
# 终止标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
"""
criteria:迭代停止的模式选择元组,格式为(type,max_iter,epsilon)
其中,type又有两种选择:
--cv2.TERM_CRITERIA_EPS :精确度(误差)满足epsilon停止。
—- cv2.TERM_CRITERIA_MAX_ITER:迭代次数超过max_iter停止。
—-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,两者综合,任意一个满足结束
"""
# 角点世界坐标矩阵
objp = np.zeros((13*6,3), np.float32)

objp[:,:2] = np.mgrid[0:13,0:6].T.reshape(-1,2)
# 因为世界坐标系被放置在标定板平面上，故每个角点的世界坐标是已知的，有规律分布的（标定板黑白格等长度）

objpoints = [] # 在现实世界空间的3d点坐标，需要repmat使其与imgpoints等大
imgpoints = [] # 图像平面中的2d点坐标。
images=[]
for i in range(1,11):
    images.append('./Chessboard/'+str(i)+'.jpg')

# 该函数用于在2d图像上绘制3d坐标轴
def draw_1(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    a1=int(corner[0])
    b1=int(corner[1])
    a2=int(imgpts[0].ravel()[0])
    b2=int(imgpts[0].ravel()[1])
    a3=int(imgpts[1].ravel()[0])
    b3=int(imgpts[1].ravel()[1])
    a4=int(imgpts[2].ravel()[0])
    b4=int(imgpts[2].ravel()[1])
    img = cv2.line(img, (a1,b1), (a2,b2), (255, 0, 0), 5)#x
    img = cv2.line(img, (a1,b1), (a3,b3), (0, 255, 0), 5)#y
    img = cv2.line(img, (a1,b1), (a4,b4), (0, 0, 255), 5)#z
    """
    cv.line:
第一个参数img:要划的线所在的图像;
第二个参数pt1:直线起点
第三个参数pt2:直线终点
第四个参数color:直线的颜色 e.g:Scalor(0,0,255)
第五个参数thickness=1:线条粗细
第六个参数line_type=8,
8 (or 0)  -  8-connected line(8邻接)连接 线。
4         -  4-connected line(4邻接)连接线。
CV_AA     -  antialiased 线条。
第七个参数：坐标点的小数点位数。
    """
    return img
# 该函数用于在2d图像上绘制3d立方体
def draw_2(img,corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

axis_1 = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
axis_2 = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                   [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])

#开始处理
#1、图像识别角点，获取控制点图像像素2d坐标和物理世界3d坐标的对应关系
# 由于是初始标定，假设图像是无畸变的
index=1
for fname in images:
    #对每张图片，识别出角点，记录世界物体坐标和图像像素坐标

    print("processing img:%s"%fname)
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转灰度
    #寻找角点，存入corners，ret是是否成功找到角点的逻辑值
    ret, corners = cv2.findChessboardCorners(gray, (13, 6),None)
    #corner = tuple(corners[0].ravel())
    """
    findChessboardCorners(image,patternSize,corners,flags = None)
    此函数试图确定输入图片是否有棋盘图案，并定位棋盘板上的内角点。
    如果所有的角点被找到且以一定的顺序排列（一行接一行，从一行的左边到右边），该函数会返回一个非零值。
    如果该函数没有找到所有的角点或者重新排列他们,则返回0。
    image:输入原始的棋盘板图像。该图像必须是一张8位的灰度图或色彩图。
    patternSize:(w,h):棋盘上每一排和每一列的内角数。w=棋盘板一行上黑白块的数量-1,h=棋盘板一列上黑白块的数量-1,例如,10x6的棋盘板,则(w,h)=(9,5)
    corners:array:检测到的角点的输出数组。
    """

    # 如果找到，添加标定对象世界点，图像像素点（精炼后）
    if ret == True:
        print('chessboard detected')
        objpoints.append(objp) #在列表最后添加一个元素
        #执行亚像素级角点检测
        corners2 = cv2.cornerSubPix(gray,corners,(10,10),(-1,-1),criteria)
        #将得到的精确的角点像素坐标加入列表
        imgpoints.append(corners2)

        # 绘制并显示角点
        img = cv2.drawChessboardCorners(img, (13,6), corners2,ret)
        """
        第一个参数是棋盘格图像。
        第二个参数是棋盘格内部角点的行、列,和cv::findChessboardCorners()指定的相同。
        第三个参数是检测到的棋盘格角点。
        第四个参数是cv::findChessboardCorners()的返回值。S
        """
        cv2.namedWindow('img',0)
        cv2.resizeWindow('img', 500, 500)
        cv2.imshow('img',img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        cv2.imwrite('./DrawCorners/'+str(index)+'.jpg',img)
        index+=1
    if ret == 0:
        print('not found corners')
# 相机标定核心函数
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
'''
初始标定结果：
ret:相机重投影误差
mtx:相机内参矩阵
dist:畸变参数
rvecs:旋转向量
tvecs:平移向量
'''

#2、通过初始值和畸变原理对标定结果进行优化，减少图像的畸变
images_1=[]
for i in range(1,11):
    images_1.append('./Chessboard/'+str(i)+'.jpg')
index_1=1
for fname_1 in images_1:
    print("processing img:%s"%fname_1)
    img_1 = cv2.imread(fname_1)
    h,w = img_1.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    """
Mat cv::getOptimalNewCameraMatrix	
(	
		InputArray 	cameraMatrix,                  // 相机内参矩阵
        InputArray 	distCoeffs,                    // 相机畸变参数
        Size 	        imageSize,                     // 图像尺寸
        double 	        alpha,                         // 缩放比例
        								//当alpha=1时,所有像素均保留，但存在黑色边框。
										//当alpha=0时,损失最多的像素，没有黑色边框。
        Size 	        newImgSize = Size(),           // 校正后的图像尺寸
)	
返回新的相机内参矩阵,roi:原始图像的建议区域 
"""
#纠正畸变
    dst = cv2.undistort(img_1, mtx, dist, None, newcameramtx)


# 裁剪图像，输出纠正畸变以后的图片
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.namedWindow('img_1',0)
    cv2.resizeWindow('img_1', 500, 500)
    cv2.imshow('img_1',dst)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    cv2.imwrite('./after_correct/'+str(index_1)+'.jpg',dst)

# 记录每张图片标定参数
    f=open('./parameters/pic_0'+str(index_1)+'纠正后'+'.txt','w')
    f.write('相机内参矩阵：\n')
    with open('./parameters/pic_0'+str(index_1)+'纠正后'+'.txt','a') as f:
        np.savetxt(f,newcameramtx, fmt='%f', delimiter=' , ')
        f.write('畸变值:\n')
        np.savetxt(f,dist, fmt='%f', delimiter=' , ')
        f.write('相机外参（旋转向量）:\n')
        np.savetxt(f,rvecs[index_1-1], fmt='%f', delimiter=' , ')
        f.write('相机外参（平移向量）:\n')
        np.savetxt(f,tvecs[index_1-1], fmt='%f', delimiter=' , ')
    f.close
    index_1+=1

# 3、结果输出，标定坐标显示的图像，标定参数，标定质量评估
images_2=[]
for i in range(1,11):
    images_2.append('./Chessboard/'+str(i)+'.jpg')
index_2=1
for fname_2 in images_2:
    img_2 = cv2.imread(fname_2)
    gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    ret_2, corners_2 = cv2.findChessboardCorners(gray_2, (13, 6), None)
    if ret_2:
        print("drawing 3D:%s"%fname_2)
        corners2_2 = cv2.cornerSubPix(gray_2, corners_2, (11, 11), (-1, -1), criteria)
        # 寻找旋转和平移矢量
        ret_2, rvecs_2, tvecs_2 = cv2.solvePnP(objp, corners2_2, mtx, dist)
        # 将3D点投影到平面，输出结果：每张图片上绘有标定世界坐标系和三维立方体
        imgpts_1, jac = cv2.projectPoints(axis_1, rvecs_2, tvecs_2, mtx, dist)
        imgpts_2, jac = cv2.projectPoints(axis_2, rvecs_2, tvecs_2, mtx, dist)
        img_2 = draw_1(img_2, corners2_2, imgpts_1)
        img_2 = draw_2(img_2, corners2_2, imgpts_2)
        cv2.imshow('img_2', img_2)
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        cv2.imwrite('./result/pic_0'+str(index_2)+'.jpg',img_2)
        cv2.destroyAllWindows()

    f=open('./parameters/pic_0'+str(index_2)+'.txt','w')
    f.write('相机内参矩阵：\n')
    with open('./parameters/pic_0'+str(index_2)+'.txt','a') as f:
        np.savetxt(f,mtx, fmt='%f', delimiter=' , ')
        f.write('畸变值:\n')
        np.savetxt(f,dist, fmt='%f', delimiter=' , ')
        f.write('相机外参（旋转向量）:\n')
        np.savetxt(f,rvecs[index_2-1], fmt='%f', delimiter=' , ')
        f.write('相机外参（平移向量）:\n')
        np.savetxt(f,tvecs[index_2-1], fmt='%f', delimiter=' , ')
    f.close
    index_2+=1


#计算最终标定结果的误差（标定精度评定）,单位为像素
tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)# 重投影
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(objpoints[i])
    tot_error += error

print ("total error: ", tot_error/len(objpoints))

