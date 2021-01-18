import os

class Path(object):
    @staticmethod
    def root_dir(name):
        if name == 'img':
            return os.path.join('D:\\','PycharmProjects','aSegNet','cdw2014_dataset')
        elif name == 'gt':
            return os.path.join('D:\\','PycharmProjects','aSegNet','cdw2014_train')
        elif name =='model':
            return os.path.join('D:\\','PycharmProjects','aSegNet','run','deeplab-cdw2014')
        elif name =='result':
            return os.path.join('D:\\','PycharmProjects','aSegNet','test','results')
        else:
            print('{} not available.'.format(name))
            raise NotImplementedError