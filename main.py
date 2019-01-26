import utils

utils.plot_cloud("bed_0001.off", 3)

voxels = utils.voxelize("bed_0001.off", 2)

utils.plot_mesh("bed_0001.off", 2, view='yz')

pixels = utils.pixelize("bed_0001.off", 1, view='yz')
