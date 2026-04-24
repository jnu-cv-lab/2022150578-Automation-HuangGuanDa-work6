ORB与SIFT特征匹配对比研究
一、实验目的
理解特征点与特征描述子的基本概念；
掌握ORB特征检测、描述与匹配的实现方法；
掌握SIFT特征检测、描述与匹配的实现方法；
学会使用RANSAC剔除错误匹配并估计Homography矩阵；
对比ORB与SIFT在目标定位任务中的性能差异。

二、实验原理
2.1 特征点与特征描述子
特征点是图像中具有显著纹理变化的位置（如角点、边缘交点），包含位置、尺度和方向信息。特征描述子是对特征点周围邻域信息的量化向量，用于在不同图像中匹配同一物理点。

2.2 ORB算法
ORB（Oriented FAST and Rotated BRIEF）结合了FAST角点检测和BRIEF描述子，并加入了方向估计和图像金字塔，使其具有旋转和尺度鲁棒性。ORB描述子为256位的二进制字符串，使用汉明距离进行匹配。

2.3 SIFT算法
SIFT（Scale-Invariant Feature Transform）通过高斯差分金字塔检测关键点，并基于梯度方向直方图生成128维浮点描述子。SIFT使用欧氏距离进行匹配，对尺度、旋转和光照变化具有较好的鲁棒性。

2.4 RANSAC与Homography
RANSAC（随机采样一致）通过几何一致性检验剔除错误匹配。Homography矩阵描述两幅平面图像之间的透视变换关系，用于目标定位。

三、实验环境
项目	配置
操作系统	Ubuntu (WSL)
编程语言	Python 3
主要库	OpenCV 4.x, NumPy
测试图像	box.png, box_in_scene.png

四、实验内容与步骤
4.1 任务1：ORB特征点检测
使用cv2.ORB_create(nfeatures=1000)创建检测器，调用detectAndCompute()检测关键点和描述子。

4.2 任务2：ORB特征匹配
使用cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)创建暴力匹配器，按汉明距离排序并显示前30个匹配。

4.3 任务3：RANSAC剔除错误匹配
使用cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)估计单应矩阵，根据返回的mask筛选内点。

4.4 任务4：目标定位
将box.png的四个角点通过Homography矩阵投影到场景图中，使用cv2.polylines()绘制定位边框。

4.5 任务5：参数对比实验
改变ORB的nfeatures参数（500/1000/2000），观察匹配效果变化。

4.6 任务6：SIFT对比实验
使用SIFT算法完成相同任务，与ORB进行对比分析。

4.7 SIFT 特征匹配（选做任务）

七、实验结论
ORB算法：速度快，匹配数量多，但内点比例较低，适合实时性要求高的应用场景。
SIFT算法：内点比例极高（93.75%），匹配质量显著优于ORB，但计算速度较慢，适合精度要求高的场景。
RANSAC必要性：ORB的初始匹配中约82%是错误匹配，必须使用RANSAC剔除才能实现准确定位。
算法选择建议：
实时应用（SLAM/视频处理）：优先选择ORB；
精度优先（图像拼接/目标识别）：优先选择SIFT；
两者都需要配合RANSAC实现鲁棒定位。

八、实验心得
通过本次实验，我深入理解了特征点检测、描述子匹配、RANSAC剔除和Homography变换的完整流程。ORB与SIFT的对比实验让我认识到：没有一种算法适用于所有场景，实际应用中需要根据任务需求（速度 vs 精度）选择合适的特征匹配方法。

