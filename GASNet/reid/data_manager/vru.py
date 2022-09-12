import os.path as osp
import numpy as np

from ..utils.serialization import write_json, read_json

class VRU(object):
    dataset_dir = 'VRU'

    def __init__(self, root='datasets', split_id=0, verbose=True, **kwargs):
        super(VRU, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.label_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.imgs_dir = osp.join(self.dataset_dir, 'Pic')
        # self.split_labeled_json_path = osp.join(self.dataset_dir, 'train_validation.json')
        self.split_labeled_json_path = osp.join(self.dataset_dir, 'test_1200.json')
        # self.split_labeled_json_path = osp.join(self.dataset_dir, 'test_1200_big_gallery.json')
        # self.split_labeled_json_path = osp.join(self.dataset_dir, 'test_2400.json')
        # self.split_labeled_json_path = osp.join(self.dataset_dir, 'test_2400_big_gallery.json')
        # self.split_labeled_json_path = osp.join(self.dataset_dir, 'test_8000.json')
        # self.split_labeled_json_path = osp.join(self.dataset_dir, 'test_8000_big_gallery.json')
        # self.split_labeled_json_path = osp.join(self.dataset_dir, 'validation.json')
        # self.split_labeled_json_path = osp.join(self.dataset_dir, 'validation_big_gallery.json')


        self._check_before_run()
        self._preprocess()

        split_path = self.split_labeled_json_path

        splits = read_json(split_path)
        assert split_id < len(splits), "Condition split_id ({}) < len(splits) ({}) is false".format(split_id, len(splits))
        split = splits[split_id]
        print("Split index = {}".format(split_id))

        train = split['train']  # list
        query = split['query']  # list
        gallery = split['gallery']  # list

        num_train_cids = split['num_train_pids']  # int: 车辆实例数
        num_query_cids = split['num_query_pids']  # int
        num_gallery_cids = split['num_gallery_pids']  # int
        num_total_cids = num_train_cids + num_query_cids

        num_train_imgs = split['num_train_imgs']  # int
        num_query_imgs = split['num_query_imgs']  # int
        num_gallery_imgs = split['num_gallery_imgs']  # int
        num_total_imgs = num_train_imgs + num_query_imgs

        if verbose:
            print("=> VRU loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_cids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_cids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_cids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_cids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_cids = num_train_cids
        self.num_query_cids = num_query_cids
        self.num_gallery_cids = num_gallery_cids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.label_dir):
            raise RuntimeError("'{}' is not available".format(self.label_dir))
        if not osp.exists(self.imgs_dir):
            raise RuntimeError("'{}' is not available".format(self.imgs_dir))
        # if not osp.exists(self.split_labeled_json_path):
        #     raise RuntimeError("'{}' is not available".format(self.split_labeled_json_path))

    def _preprocess(self):
        """
        This function is a bit complex and ugly, what it does is
        1. Extract data from cuhk-03.mat and save as png images.
        2. Create 20 classic splits. (Li et al. CVPR'14)
        3. Create new split. (Zhong et al. CVPR'17)
        """
        print("Note: if root path is changed, the previously generated json files need to be re-generated (delete them first)")

        def _extract_split(label_dir, label_file_name, pic_dir, split_name):  #
            split_name_path = osp.join(label_dir, label_file_name)
            split_file = open(split_name_path)
            line = split_file.readline()
            train, query, gallery = [], [], []
            car_dic = {}
            # 处理训练集
            if split_name == "train":
                car_count = 0
                while line:
                    line_list = line.split()  # str转list
                    if line_list[1] in car_dic:
                        car_dic[line_list[1]] = car_dic[line_list[1]] + 1
                    else:
                        car_dic[line_list[1]] = 1
                        car_id = car_count
                        car_count += 1
                    line_list[0] = osp.join(pic_dir, line_list[0]+".jpg")
                    line_list[1] = int(car_id)
                    line_list.append(0)  # camera ID ,没啥用 占个位而已
                    train.append(line_list)
                    line = split_file.readline()
                split_file.close()
                return train, len(car_dic), len(train)
            # 处理查询集和图库集
            else:
                while line:
                    line_list = line.split()
                    if line_list[1] in car_dic:
                        car_dic[line_list[1]].append(int(line_list[0]))
                    else:
                        car_dic[line_list[1]] = [int(line_list[0])]
                    line = split_file.readline()
                num_cids = 0
                car_id = 0
                for key, value in car_dic.items():
                    if len(value) > 1:
                        num_cids += 1
                        gallery_list = []
                        # choose a random image to set the gallery
                        gallery_index = np.random.randint(0, len(value))
                        gallery_list.append(osp.join(pic_dir, str(value[gallery_index]) + ".jpg"))
                        gallery_list.append(car_id)
                        gallery_list.append(0)
                        # gallery.append(gallery_list)  # 图库集数量少
                        query.append(gallery_list)  # 查询集数量少

                        # put the rest images into the query
                        value.pop(gallery_index)
                        for i in range(len(value)):
                            query_list = []
                            query_list.append(osp.join(pic_dir, str(value[i])+".jpg"))
                            query_list.append(car_id)
                            query_list.append(0)
                            # query.append(query_list)
                            gallery.append(query_list)
                    car_id += 1
                return query, gallery, num_cids, num_cids, len(query), len(gallery)

        train, num_train_cids, num_train_imgs = _extract_split(self.label_dir, "train_list.txt", self.imgs_dir, "train")
        query, gallery, num_query_cids, num_gallery_cids, num_query_imgs, num_gallery_imgs = _extract_split(self.label_dir, "test_list_1200.txt", self.imgs_dir, "test")

        splits = [{
            'train': train, 'query': query, 'gallery': gallery,
            'num_train_pids': num_train_cids, 'num_train_imgs': num_train_imgs,
            'num_query_pids': num_query_cids, 'num_query_imgs': num_query_imgs,
            'num_gallery_pids': num_gallery_cids, 'num_gallery_imgs': num_gallery_imgs,
        }]
        write_json(splits, self.split_labeled_json_path)