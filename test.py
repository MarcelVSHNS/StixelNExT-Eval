from resultloader.cityscapes import CityscapesDataLoader


city = CityscapesDataLoader(root_dir="C:\\Users\\marce\\Documents\\datasets\\cityscapes")
data = city[101]
print(len(data.gt_obstacles))
