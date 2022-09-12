import numpy as np
import os.path as osp

import torch

from ..utils.serialization import read_json
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms
# import

def evaluation(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """
    输入：
    1.距离矩阵，dismat
    2.query集行人ID，q_pids
    3.gallery集行人ID，g_pids
    4.query集摄像机ID，q_camids
    5.gallery集摄像机ID，g_camids
    6.计算CMC中Rank-N中的N最大值max_rank

    输出：
    1.记录CMC曲线中Rank-N的列表all_cmc
    2.记录所有符合要求的query数据的平均准确率均值mAP
    """
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    query_cams = np.asarray(q_camids)
    gallery_cams = np.asarray(g_camids)
    num_q, num_g = distmat.shape  # 获得距离矩阵的规模
    if num_g < max_rank:  # 若gallery数据少于max_rank，则将max_rank设为gallery数据数量num_g
        print("提示：gallery数据集数据不足，将参数'max_rank'从{}修改为{}。".format(max_rank, num_g))
        max_rank = num_g

    """
    np.argsort指定按行进行排序，返回每行按升序排列的元素下标
    例如有一个人列表[1,2,0]，则按行排列后，就返回[2,0,1]代表第2个元素最小，第0个元素第二小，第1个元素最大
    使用该函数便可以按照距离大小进行排序，并获取排序后的下标顺序
    """
    indices = np.argsort(distmat, axis=1)  # 按照距离进行排序

    """
    显示搜索结果
    """
    result_show_indices = indices[:, :10]
    dataset_dir = "vru"
    root = "datasets"
    dataset_dir = osp.join(root, dataset_dir)
    split_path = osp.join(dataset_dir, "test_10.json")
    splits = read_json(split_path)
    split = splits[0]
    query_indx = split["query"]
    gallery_indx = split["gallery"]
    query_dir = []
    gallery_dir = []
    for i in range(len(query_indx)):
        query_dir.append(query_indx[i][0])
    for i in range(len(gallery_indx)):
        gallery_dir.append(gallery_indx[i][0])
    result_dir = []
    result_show_indices = result_show_indices.numpy().tolist()
    for i in range(len(result_show_indices)):
        result_d = result_show_indices[i]
        re = []
        for j in range(len(result_d)):
            re.append(gallery_dir[result_d[j]])
        result_dir.append(re)
    N = len(query_dir)
    M = 11
    image_num = 0
    for i in range(len(query_dir)):
        query_img = plt.imread(query_dir[i])
        image_num += 1
        plt.subplot(N, M, image_num)
        plt.imshow(query_img)
        plt.xticks([])
        plt.yticks([])
        for j in range(10):
            result_img = plt.imread(result_dir[i][j])
            image_num += 1
            plt.subplot(N, M, image_num)
            plt.imshow(result_img)
            plt.xticks([])
            plt.yticks([])
    plt.savefig('test2.jpg')
    plt.show()
    image = plt.imread("test2.jpg")
    image = torch.from_numpy(np.transpose(image, (2,0,1)))
    writer = SummaryWriter("./result_tb")
    writer.add_image("The result of ReID", image)
    writer.close()

    """
    g_pids[indices]生成了一个与距离矩阵相同规模的矩阵，但矩阵元素是按照距离大小升序排列后对应的gallery的行人ID
    假设距离矩阵某一行为[1,0,5,6]，按行升序排列后得到的下标列表为[1,0,2,3]，gallery行人ID为[4,5,6,8]
    则可以计算得到g_pids[indices]对应的那一行为[5,4,6,8]，即该行对应的特征应该被分类到ID为5的类中

    q_pids[:, np.newaxis]将q_pids矩阵增加了一维
    将g_pids[indices]与q_pids进行匹配，对应位置相同则元素为1，否则为0
    这样，便可以获得一个对应关系矩阵matches，该矩阵与距离矩阵规模相同。

    matches矩阵第i行第j个元素代表query第i个行人ID与gallery中于其距离第j近的数据行人ID是否相同
    举例，matches[1][3]=1，说明query中第1个行人与距离第三近的gallery数据属于同一行人
    """
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # 进行ID匹配，计算匹配矩阵matched，便于计算cmc与AP
    # matches = (g_pids[indices] == q_pids[:, np.newaxis])  # 进行ID匹配，计算匹配矩阵matched，便于计算cmc与AP
    all_cmc = []  # 记录每个query数据的CMC数据
    all_AP = []  # 记录每个query数据的AP
    num_valid_q = 0.  # 记录符合CMC与mAP计算的query数据的总数，便于计算总Rank-N

    for q_idx in range(num_q):  # 对于query集中的每个数据

        q_pid = q_pids[q_idx]  # 获取该数据的行人ID
        order = indices[q_idx]  # 获得有关该数据的gallery数据距离排序

        # 删除与该query数据相同摄像机ID、行人ID的数据。相同摄像机相同行人的gallery数据不符合跨摄像机的要求
        # remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)  # 得到需要删除的元素的bool类型列表
        # keep = np.invert(remove)  # 对remove进行翻转得到可以保留的元素的bool类型列表

        orig_cmc = matches[q_idx]  # 匹配矩阵只保留对应keep中为True的元素，得到该query数据的匹配列表
        if not np.any(orig_cmc):  # 如果该query数据未在可以保留的gallery集中出现，说明该query数据不符合CMC与mAP计算要求，返回循环头
            continue

        """
        计算每个query数据的CMC数据
        """
        # 计算匹配列表的叠加和
        cmc = orig_cmc.cumsum()
        # 根据叠加和得到该query数据关于gallery数据的Rank-N
        cmc[cmc > 1] = 1 # 把cmc中大于1的数字改为1
        # 将该query数据的CMC数据加入all_AP列表便于之后计算mAP，可以通过指定max_rank来指定一行保留多少列，默认50列
        all_cmc.append(cmc[:max_rank])

        """
        计算每个query数据的AP
        """
        # 每个query数据的正确匹配总数
        num_rel = orig_cmc.sum()
        # 计算匹配列表的叠加和
        tmp_cmc = orig_cmc.cumsum()
        # 计算每次正确匹配的准确率
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        # 将错误匹配的准确率降为0
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        # 计算平均准确度
        AP = tmp_cmc.sum() / num_rel
        # 将该query数据的AP加入all_AP列表便于之后计算mAP
        all_AP.append(AP)

        # 统计符合CMC与mAP计算的query数据的总数，便于计算总Rank-N
        num_valid_q += 1.

    # 如果符合CMC计算的query数据的总数小于等于0，则报错所有query数据都不符合要求
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    all_cmc = np.asarray(all_cmc).astype(np.float32)  # 将all_cmc转换为np.array类型
    all_cmc = all_cmc.sum(0) / num_valid_q  # 将所有符合条件的query数据的Rank-N按列求和并取平均数，即可计算总CMC曲线中的Rank-N
    mAP = np.mean(all_AP)  # 平均准确率均值就是所有符合条件的query数据平均准确率的平均数

    return all_cmc, mAP
