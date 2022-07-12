import os
import time
import pickle


class Tools:

    @classmethod
    def print(cls, info=None, txt_path=None):
        info = "" if info is None else "{} {}".format(cls.get_format_time(), info)
        print(info)

        if txt_path is not None:
            cls.write_to_txt(txt_path, "{}\n".format(info), reset=False)
            pass
        pass

    @staticmethod
    def get_format_time():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    @staticmethod
    def write_to_txt(_path, _txt, reset=False):
        with open(_path, "w" if reset else "a") as f:
            f.writelines(_txt)
        pass

    # save file
    @staticmethod
    def write_to_pkl(_path, _data):
        with open(_path, "wb") as f:
            pickle.dump(_data, f)
        pass

    # read file
    @staticmethod
    def read_from_pkl(_path):
        with open(_path, "rb") as f:
            return pickle.load(f)
        pass

    @staticmethod
    def new_dir(_path_or_file):
        if "." in os.path.basename(_path_or_file):
            new_dir = os.path.split(_path_or_file)[0]
        else:
            new_dir = _path_or_file
            pass
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            pass
        return _path_or_file

    pass


